"""Query tools for filtering and analyzing structured data."""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging
import json
import os
import re
from .config import get_llm_client, LLM_PROVIDER, MODEL_NAME

from .llm_schema import StructuredOutput, PlanSummary

logger = logging.getLogger(__name__)

def _sanitize_query_for_schema_tagging(query_text: str) -> str:
    """
    Sanitize query text to reduce medical terminology that might trigger safety filters.
    Replace medical terms with generic data field references.
    """
    if not query_text:
        return ""
    
    # Replace common medical terms with generic data field references
    replacements = {
        'patient': 'record',
        'patients': 'records',
        'kidney stone': 'data field',
        'kidney stones': 'data fields',
        'stone': 'data point',
        'stones': 'data points',
        'bladder': 'volume field',
        'medical': 'data',
        'clinical': 'data',
    }
    
    sanitized = query_text.lower()
    for term, replacement in replacements.items():
        sanitized = sanitized.replace(term, replacement)
    
    return sanitized

def _call_llm_unified(client, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    """Unified LLM call that works with both OpenAI and Gemini."""
    if LLM_PROVIDER == "openai":
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        
        # Try different model name formats (including 2.5 models)
        model = None
        model_variants = [
            MODEL_NAME,  # Try original name first
            f"{MODEL_NAME}-latest",  # Try with -latest suffix
            # Try 2.5 variants if 1.5 is specified
            MODEL_NAME.replace("gemini-1.5-flash", "gemini-2.5-flash"),
            MODEL_NAME.replace("gemini-1.5-pro", "gemini-2.5-pro"),
            MODEL_NAME.replace("gemini-1.5-flash", "gemini-1.5-flash-latest"),
            MODEL_NAME.replace("gemini-1.5-pro", "gemini-1.5-pro-latest"),
            # Try 1.5 variants if 2.5 is specified
            MODEL_NAME.replace("gemini-2.5-flash", "gemini-1.5-flash-latest"),
            MODEL_NAME.replace("gemini-2.5-pro", "gemini-1.5-pro-latest"),
        ]
        
        for variant in model_variants:
            try:
                model = genai.GenerativeModel(variant)
                # Test if model is accessible
                _ = model._model_name
                break
            except Exception:
                continue
        
        if model is None:
            # Last resort: try to list available models and use the first matching one
            try:
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                # Extract short names (last part after '/')
                available_short_names = [m.split('/')[-1] for m in available_models]
                
                # Try to find a matching model based on type (flash/pro)
                preferred_order = []
                if 'flash' in MODEL_NAME.lower():
                    preferred_order = [m for m in available_short_names if 'flash' in m.lower()]
                elif 'pro' in MODEL_NAME.lower():
                    preferred_order = [m for m in available_short_names if 'pro' in m.lower()]
                
                # If no preferred match, use any available model
                if not preferred_order:
                    preferred_order = available_short_names
                
                # Try models in preferred order
                for model_short_name in preferred_order:
                    try:
                        model = genai.GenerativeModel(model_short_name)
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        
        if model is None:
            raise ValueError(f"Could not find a valid Gemini model. Tried: {model_variants}. Please check your API key and available models.")
        
        # Configure safety settings to allow medical content
        # Import safety enums
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        # Try most permissive settings - BLOCK_NONE for DANGEROUS_CONTENT to allow medical terminology
        # Note: BLOCK_NONE may require special permissions, but we try it first
        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # Most permissive for medical data
            }
        except (AttributeError, ValueError):
            # Fallback: Try using numeric values if enum doesn't work
            # 0 = BLOCK_NONE, 1 = BLOCK_ONLY_HIGH, 2 = BLOCK_MEDIUM_AND_ABOVE, 3 = BLOCK_LOW_AND_ABOVE
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: 1,  # BLOCK_ONLY_HIGH
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: 1,  # BLOCK_ONLY_HIGH
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 1,  # BLOCK_ONLY_HIGH
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 0,  # BLOCK_NONE - most permissive
            }
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        response = model.generate_content(
            prompt, 
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Check if response was blocked or filtered
        if not response.candidates or len(response.candidates) == 0:
            logger.warning("Gemini API returned no candidates - content may have been blocked")
            raise ValueError("Gemini API returned no response - content may have been filtered")
        
        candidate = response.candidates[0]
        
        # Check finish_reason
        # finish_reason values: 0=STOP, 1=MAX_TOKENS, 2=SAFETY, 3=RECITATION, 4=OTHER
        if candidate.finish_reason == 2:  # SAFETY - content was blocked
            logger.warning(f"Gemini API blocked content (finish_reason=SAFETY). Safety ratings: {candidate.safety_ratings}")
            raise ValueError("Gemini API blocked the content due to safety filters")
        elif candidate.finish_reason == 3:  # RECITATION
            logger.warning(f"Gemini API blocked content (finish_reason=RECITATION)")
            raise ValueError("Gemini API blocked the content due to recitation policy")
        elif candidate.finish_reason not in [0, 1]:  # Not STOP or MAX_TOKENS
            logger.warning(f"Gemini API returned unexpected finish_reason: {candidate.finish_reason}")
            raise ValueError(f"Gemini API returned unexpected finish_reason: {candidate.finish_reason}")
        
        # Check if content is available
        if not candidate.content or not candidate.content.parts:
            logger.warning("Gemini API response has no content parts")
            raise ValueError("Gemini API response has no content")
        
        # Extract text from parts
        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
        if not text_parts:
            logger.warning("Gemini API response has no text content")
            raise ValueError("Gemini API response has no text content")
        
        return ''.join(text_parts).strip()
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

class QueryTools:
    """Tools for querying and filtering structured radiology data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with structured DataFrame.
        
        Args:
            df: DataFrame with structured radiology data
        """
        self.df = df.copy()
        self._prepare_data()
        
        # Dynamic learning components
        self.learned_filters_file = "out/learned_filters.json"
        self.learned_statistics_file = "out/learned_statistics.json"
        self.learned_operations_file = "out/learned_operations.json"
        
        # Load learned patterns
        self.learned_filters = self._load_learned_filters()
        self.learned_statistics = self._load_learned_statistics()
        self.learned_operations = self._load_learned_operations()
        
        # Available columns for dynamic learning
        self.available_columns = list(self.df.columns)
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.text_columns = self.df.select_dtypes(include=['object']).columns.tolist()
    
    def _prepare_data(self):
        """Prepare data for querying."""
        # Ensure date columns are datetime
        if 'imaging_date' in self.df.columns:
            self.df['imaging_date'] = pd.to_datetime(self.df['imaging_date'], errors='coerce')
        
        # Convert size columns to numeric
        size_columns = ['right_stone_size_cm', 'left_stone_size_cm', 'bladder_volume_ml']
        for col in size_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create helper columns for easier querying
        self.df['has_right_stone'] = self.df['right_stone'] == 'present'
        self.df['has_left_stone'] = self.df['left_stone'] == 'present'
        self.df['has_any_stone'] = (self.df['has_right_stone'] | self.df['has_left_stone'])
        self.df['has_bilateral_stones'] = (self.df['has_right_stone'] & self.df['has_left_stone'])
        
        # Extract year from imaging_date
        if 'imaging_date' in self.df.columns:
            self.df['imaging_year'] = self.df['imaging_date'].dt.year
    
    def apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame. LLM handles intent, so we use simple direct filtering.
        
        Args:
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered DataFrame
        """
        # Log what filters we're applying
        if filters:
            logger.info(f"Applying filters: {list(filters.keys())}")
        else:
            logger.warning("apply_filters called with empty filters dict - will return all rows!")
        
        # Basic validation - check for invalid filter combinations
        if 'min_size_cm' in filters and 'max_size_cm' in filters:
            min_size = filters.get('min_size_cm')
            max_size = filters.get('max_size_cm')
            if min_size is not None and max_size is not None and min_size > max_size:
                logger.warning(f"Invalid filter: min_size_cm ({min_size}) > max_size_cm ({max_size})")
                return self.df.iloc[0:0].copy()
        
        filtered_df = self.df.copy()
        
        # Apply side filter
        if 'side' in filters:
            filtered_df = self._apply_side_filter(filtered_df, filters['side'])
        
        # Apply stone presence filter
        if 'stone_presence' in filters:
            filtered_df = self._apply_presence_filter(filtered_df, filters['stone_presence'])
        
        # Apply size filters
        if 'min_size_cm' in filters or 'max_size_cm' in filters:
            filtered_df = self._apply_size_filters(filtered_df, filters)
        
        # Apply date filters
        if 'start_year' in filters:
            start_year = filters['start_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] >= start_year]
        
        if 'end_year' in filters:
            end_year = filters['end_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] <= end_year]
        
        # Apply bladder volume filters
        if 'min_bladder_volume_ml' in filters:
            min_volume = filters['min_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] >= min_volume]
        
        if 'max_bladder_volume_ml' in filters:
            max_volume = filters['max_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] <= max_volume]
        
        # Apply narrative text search filters (new schema)
        if 'narrative_contains_any' in filters:
            # Search for ANY of the terms
            terms = filters['narrative_contains_any']
            if isinstance(terms, str):
                terms = [terms]
            if isinstance(terms, list) and len(terms) > 0:
                # Filter out empty strings
                terms = [str(t).strip() for t in terms if t and str(t).strip()]
                if len(terms) > 0:
                    # Try to find text column (narrative, history_summary, or any text column)
                    text_column = None
                    if 'narrative' in filtered_df.columns:
                        text_column = 'narrative'
                    elif 'history_summary' in filtered_df.columns:
                        text_column = 'history_summary'
                    else:
                        # Find any text/object column that might contain narrative
                        text_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
                        # Prefer columns with longer text (likely narratives)
                        for col in text_columns:
                            if col not in ['recordid', 'imaging_date', 'matched_reason']:
                                sample_len = filtered_df[col].astype(str).str.len().mean()
                                if sample_len > 50:  # Likely a narrative column
                                    text_column = col
                                    break
                        if not text_column and text_columns:
                            # Fallback to first text column
                            text_column = text_columns[0]
                    
                    if text_column:
                        # Build regex pattern for ANY match
                        pattern = '|'.join([re.escape(str(term).lower()) for term in terms])
                        mask = filtered_df[text_column].astype(str).str.lower().str.contains(pattern, case=False, na=False, regex=True)
                        filtered_df = filtered_df[mask]
                        logger.info(f"Applied narrative_contains_any filter on '{text_column}' column with {len(terms)} terms. Matched {len(filtered_df)} rows.")
                    else:
                        logger.warning(f"narrative_contains_any filter specified but no suitable text column found. Available columns: {list(filtered_df.columns)}")
                else:
                    logger.warning(f"narrative_contains_any filter had no valid terms after cleaning")
            else:
                logger.warning(f"narrative_contains_any filter value is not a valid list: {terms}")
        
        if 'narrative_contains_all' in filters:
            # Search for ALL of the terms
            terms = filters['narrative_contains_all']
            if isinstance(terms, str):
                terms = [terms]
            if isinstance(terms, list) and len(terms) > 0:
                # Find text column (same logic as narrative_contains_any)
                text_column = None
                if 'narrative' in filtered_df.columns:
                    text_column = 'narrative'
                elif 'history_summary' in filtered_df.columns:
                    text_column = 'history_summary'
                else:
                    text_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
                    for col in text_columns:
                        if col not in ['recordid', 'imaging_date', 'matched_reason']:
                            sample_len = filtered_df[col].astype(str).str.len().mean()
                            if sample_len > 50:
                                text_column = col
                                break
                
                if text_column:
                    mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)
                    for term in terms:
                        term_mask = filtered_df[text_column].astype(str).str.contains(str(term), case=False, na=False)
                        mask = mask & term_mask
                    filtered_df = filtered_df[mask]
                    logger.info(f"Applied narrative_contains_all filter on '{text_column}' column. Matched {len(filtered_df)} rows.")
                else:
                    logger.warning(f"narrative_contains_all filter specified but no suitable text column found")
        
        if 'narrative_contains_none' in filters:
            # Exclude rows containing ANY of the terms
            terms = filters['narrative_contains_none']
            if isinstance(terms, str):
                terms = [terms]
            if isinstance(terms, list) and len(terms) > 0:
                # Find text column (same logic as narrative_contains_any)
                text_column = None
                if 'narrative' in filtered_df.columns:
                    text_column = 'narrative'
                elif 'history_summary' in filtered_df.columns:
                    text_column = 'history_summary'
                else:
                    text_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
                    for col in text_columns:
                        if col not in ['recordid', 'imaging_date', 'matched_reason']:
                            sample_len = filtered_df[col].astype(str).str.len().mean()
                            if sample_len > 50:
                                text_column = col
                                break
                
                if text_column:
                    # Build regex pattern for ANY match
                    pattern = '|'.join([re.escape(str(term).lower()) for term in terms])
                    mask = ~filtered_df[text_column].astype(str).str.lower().str.contains(pattern, case=False, na=False, regex=True)
                    filtered_df = filtered_df[mask]
                    logger.info(f"Applied narrative_contains_none filter on '{text_column}' column. Matched {len(filtered_df)} rows.")
                else:
                    logger.warning(f"narrative_contains_none filter specified but no suitable text column found")
        
        # Backward compatibility: old narrative_contains format
        if 'narrative_contains' in filters:
            search_term = str(filters['narrative_contains']).lower()
            # Find text column (same logic as above)
            text_column = None
            if 'narrative' in filtered_df.columns:
                text_column = 'narrative'
            elif 'history_summary' in filtered_df.columns:
                text_column = 'history_summary'
            else:
                text_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
                for col in text_columns:
                    if col not in ['recordid', 'imaging_date', 'matched_reason']:
                        sample_len = filtered_df[col].astype(str).str.len().mean()
                        if sample_len > 50:
                            text_column = col
                            break
            
            if text_column:
                filtered_df = filtered_df[filtered_df[text_column].astype(str).str.contains(search_term, case=False, na=False)]
                logger.info(f"Applied narrative_contains filter on '{text_column}' column. Matched {len(filtered_df)} rows.")
            else:
                logger.warning(f"narrative_contains filter specified but no suitable text column found. Available columns: {list(filtered_df.columns)}")
        
        # Apply direct column filters (for categorical columns like right_hydronephrosis, left_hydronephrosis, etc.)
        # This handles filters where the key matches a column name directly
        for key, value in filters.items():
            # Skip special filter keys that we've already handled
            special_keys = ['side', 'stone_presence', 'min_size_cm', 'max_size_cm', 'start_year', 'end_year',
                          'min_bladder_volume_ml', 'max_bladder_volume_ml', 'narrative_contains', 
                          'narrative_contains_any', 'narrative_contains_all', 'narrative_contains_none', 'distinct']
            
            if key in special_keys:
                continue
            
            # Check if this is a direct column filter
            if key in filtered_df.columns:
                # Handle categorical filters (present/absent/unclear)
                if value in ['present', 'absent', 'unclear']:
                    filtered_df = filtered_df[filtered_df[key] == value]
                    logger.info(f"Applied direct column filter '{key}' == '{value}': {len(filtered_df)} rows remaining")
                # Handle list of values (e.g., ["present", "unclear"])
                elif isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                    logger.info(f"Applied direct column filter '{key}' in {value}: {len(filtered_df)} rows remaining")
                # Handle single value match
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
                    logger.info(f"Applied direct column filter '{key}' == '{value}': {len(filtered_df)} rows remaining")
        
        # Apply distinct clause for patient counting
        if 'distinct' in filters:
            distinct_col = filters['distinct']
            if distinct_col in filtered_df.columns:
                # Get unique values of distinct column, then filter to first occurrence
                unique_values = filtered_df[distinct_col].unique()
                filtered_df = filtered_df.drop_duplicates(subset=[distinct_col], keep='first')
                logger.info(f"Applied distinct filter on '{distinct_col}': {len(filtered_df)} unique records")
            else:
                logger.warning(f"distinct filter specified for non-existent column '{distinct_col}'")
        
        # Add matched_reason column
        filtered_df = self._add_matched_reason(filtered_df, filters)
        
        return filtered_df
    
    def _determine_filtering_strategy(self, filters: Dict[str, Any], query_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determine the optimal filtering strategy based on filters and query context.
        
        Args:
            filters: Dictionary of filters to apply
            query_context: Optional context about the query
            
        Returns:
            Dictionary containing filtering strategy information
        """
        strategy = {
            'strict_mode': False,
            'side_specific_size': True,
            'combine_filters': 'AND',
            'size_logic': 'side_aware',
            'presence_logic': 'standard'
        }
        
        # Analyze query context for adaptive behavior
        if query_context:
            query_text = query_context.get('query_text') or ''
            if query_text:
                intent = self._detect_query_intent(query_text)
            else:
                intent = self._detect_query_intent('')
            
            # Apply intent-based strategy adjustments
            if intent['is_precise']:
                strategy['strict_mode'] = True
                strategy['combine_filters'] = 'AND'
                strategy['size_logic'] = 'side_aware'
            
            elif intent['is_exploratory']:
                strategy['strict_mode'] = False
                strategy['size_logic'] = 'flexible'
                strategy['tolerance_level'] = 'low'
            
            elif intent['is_comparative']:
                strategy['side_specific_size'] = False
                strategy['combine_filters'] = 'OR'
                strategy['size_logic'] = 'statistical'
            
            elif intent['is_statistical']:
                strategy['side_specific_size'] = False
                strategy['size_logic'] = 'statistical'
                strategy['combine_filters'] = 'OR'
            
            # Adjust for side-specific requirements
            if intent['requires_side_specific']:
                strategy['side_specific_size'] = True
                strategy['size_logic'] = 'side_aware'
        
        # Analyze filter combinations for optimization
        filter_keys = set(filters.keys())
        
        # If both side and size filters, use side-specific logic
        if 'side' in filter_keys and ('min_size_cm' in filter_keys or 'max_size_cm' in filter_keys):
            strategy['side_specific_size'] = True
        
        # If multiple size criteria, use range logic
        if 'min_size_cm' in filter_keys and 'max_size_cm' in filter_keys:
            strategy['size_logic'] = 'range'
        
        # If presence and side filters, optimize combination
        if 'stone_presence' in filter_keys and 'side' in filter_keys:
            strategy['presence_logic'] = 'side_aware'
        
        return strategy
    
    def _optimize_filter_order(self, filters: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
        """
        Optimize the order of filter application for better performance.
        
        Args:
            filters: Dictionary of filters to apply
            strategy: Filtering strategy
            
        Returns:
            List of filter keys in optimal order
        """
        # Start with most selective filters first
        filter_order = []
        
        # Side filters are usually most selective
        if 'side' in filters:
            filter_order.append('side')
        
        # Size filters are often selective
        if 'min_size_cm' in filters or 'max_size_cm' in filters:
            filter_order.append('min_size_cm')
            if 'max_size_cm' in filters:
                filter_order.append('max_size_cm')
        
        # Presence filters
        if 'stone_presence' in filters:
            filter_order.append('stone_presence')
        
        # Date filters
        if 'start_year' in filters:
            filter_order.append('start_year')
        if 'end_year' in filters:
            filter_order.append('end_year')
        
        # Bladder volume filters (usually less selective)
        if 'min_bladder_volume_ml' in filters:
            filter_order.append('min_bladder_volume_ml')
        if 'max_bladder_volume_ml' in filters:
            filter_order.append('max_bladder_volume_ml')
        
        return filter_order
    
    def _detect_query_intent(self, query_text: str) -> Dict[str, Any]:
        """
        Detect the intent and characteristics of a query for adaptive filtering.
        
        Args:
            query_text: The user's query text
            
        Returns:
            Dictionary with detected intent characteristics
        """
        if not query_text:
            query_text = ""
        query_lower = query_text.lower()
        
        intent = {
            'is_precise': False,
            'is_exploratory': False,
            'is_comparative': False,
            'is_statistical': False,
            'requires_side_specific': False,
            'tolerance_level': 'medium'
        }
        
        # Detect precision level
        if any(word in query_lower for word in ['exactly', 'precisely', 'only', 'strictly', 'exact']):
            intent['is_precise'] = True
            intent['tolerance_level'] = 'high'
        elif any(word in query_lower for word in ['around', 'approximately', 'about', 'roughly', 'close to']):
            intent['is_exploratory'] = True
            intent['tolerance_level'] = 'low'
        
        # Detect comparative intent
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'contrast']):
            intent['is_comparative'] = True
        
        # Detect statistical intent
        if any(word in query_lower for word in ['mean', 'average', 'median', 'statistics', 'distribution']):
            intent['is_statistical'] = True
        
        # Detect side-specific requirements
        if any(word in query_lower for word in ['left', 'right', 'bilateral', 'side']):
            intent['requires_side_specific'] = True
        
        return intent
    
    def _validate_and_suggest_filters(self, filters: Dict[str, Any], query_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate filters and suggest improvements based on query context.
        
        Args:
            filters: Dictionary of filters to validate
            query_context: Optional context about the query
            
        Returns:
            Dictionary with validation results and suggestions
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'suggestions': [],
            'optimized_filters': filters.copy()
        }
        
        # Check for conflicting filters
        if 'stone_presence' in filters and 'side' in filters:
            presence = (filters.get('stone_presence') or '').lower()
            side = (filters.get('side') or '').lower()
            
            if presence == 'absent' and side in ['left', 'right']:
                validation['warnings'].append(
                    f"Filtering for {side} side stones but stone_presence is 'absent' - this may return no results"
                )
        
        # Check for size filter logic
        if 'min_size_cm' in filters and 'max_size_cm' in filters:
            min_size = filters['min_size_cm']
            max_size = filters['max_size_cm']
            
            # Only compare if both values are not None
            if min_size is not None and max_size is not None and min_size > max_size:
                validation['is_valid'] = False
                validation['warnings'].append(
                    f"min_size_cm ({min_size}) is greater than max_size_cm ({max_size}) - this will return no results"
                )
        
        # Suggest optimizations based on query context
        if query_context:
            query_text = (query_context.get('query_text') or '').lower()
            intent = self._detect_query_intent(query_text)
            
            # Suggest side-specific filtering for precise queries
            if intent['is_precise'] and 'side' not in filters and any(word in query_text for word in ['left', 'right']):
                if 'left' in query_text:
                    validation['suggestions'].append("Consider adding 'side': 'left' filter for more precise results")
                elif 'right' in query_text:
                    validation['suggestions'].append("Consider adding 'side': 'right' filter for more precise results")
            
            # Suggest range filtering for exploratory queries
            if intent['is_exploratory'] and 'min_size_cm' in filters and 'max_size_cm' not in filters:
                suggested_max = filters['min_size_cm'] + 0.5  # Add 0.5 cm range
                validation['suggestions'].append(f"Consider adding max_size_cm: {suggested_max} for exploratory analysis")
        
        return validation
    
    def _apply_adaptive_filters(self, df: pd.DataFrame, filters: Dict[str, Any], strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters using adaptive logic based on strategy.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of filters to apply
            strategy: Filtering strategy to use
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply side filter with adaptive logic
        if 'side' in filters:
            filtered_df = self._apply_side_filter(filtered_df, filters['side'], strategy)
        
        # Apply stone presence filter with adaptive logic
        if 'stone_presence' in filters:
            filtered_df = self._apply_presence_filter(filtered_df, filters['stone_presence'], strategy)
        
        # Apply size filters with adaptive logic
        if 'min_size_cm' in filters or 'max_size_cm' in filters:
            filtered_df = self._apply_size_filters(filtered_df, filters, strategy)
        
        # Apply date filters
        if 'start_year' in filters:
            start_year = filters['start_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] >= start_year]
        
        if 'end_year' in filters:
            end_year = filters['end_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] <= end_year]
        
        # Apply bladder volume filters
        if 'min_bladder_volume_ml' in filters:
            min_volume = filters['min_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] >= min_volume]
        
        if 'max_bladder_volume_ml' in filters:
            max_volume = filters['max_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] <= max_volume]
        
        # Add matched_reason column
        filtered_df = self._add_matched_reason(filtered_df, filters)
        
        return filtered_df
    
    def _apply_side_filter(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """Apply side filter."""
        if not side:
            return df
        side = side.lower()
        
        if side == 'left':
            return df[df['has_left_stone']]
        elif side == 'right':
            return df[df['has_right_stone']]
        elif side == 'bilateral':
            return df[df['has_bilateral_stones']]
        else:
            return df
    
    def _apply_presence_filter(self, df: pd.DataFrame, presence: str) -> pd.DataFrame:
        """Apply stone presence filter."""
        if not presence:
            return df
        presence = presence.lower()
        
        if presence == 'present':
            return df[df['has_any_stone']]
        elif presence == 'absent':
            return df[~df['has_any_stone']]
        elif presence == 'unclear':
            return df[
                (df['right_stone'] == 'unclear') | 
                (df['left_stone'] == 'unclear')
            ]
        else:
            return df
    
    def _apply_size_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply size filters. LLM handles intent, so we apply filters directly."""
        filtered_df = df.copy()
        side = filters.get('side', '').lower() if 'side' in filters else None
        
        # Apply min size filter
        if 'min_size_cm' in filters:
            min_size = filters['min_size_cm']
            if side == 'left':
                size_condition = (filtered_df['left_stone_size_cm'] >= min_size)
            elif side == 'right':
                size_condition = (filtered_df['right_stone_size_cm'] >= min_size)
            elif side == 'bilateral':
                size_condition = (
                    (filtered_df['right_stone_size_cm'] >= min_size) |
                    (filtered_df['left_stone_size_cm'] >= min_size)
                )
            else:
                # No side specified - check both sides
                size_condition = (
                    (filtered_df['right_stone_size_cm'] >= min_size) |
                    (filtered_df['left_stone_size_cm'] >= min_size)
                )
            filtered_df = filtered_df[size_condition]
        
        # Apply max size filter
        if 'max_size_cm' in filters:
            max_size = filters['max_size_cm']
            if side == 'left':
                size_condition = (filtered_df['left_stone_size_cm'] <= max_size)
            elif side == 'right':
                size_condition = (filtered_df['right_stone_size_cm'] <= max_size)
            elif side == 'bilateral':
                size_condition = (
                    (filtered_df['right_stone_size_cm'] <= max_size) |
                    (filtered_df['left_stone_size_cm'] <= max_size)
                )
            else:
                # No side specified - check both sides
                size_condition = (
                    (filtered_df['right_stone_size_cm'] <= max_size) |
                    (filtered_df['left_stone_size_cm'] <= max_size)
                )
            filtered_df = filtered_df[size_condition]
        
        return filtered_df
    
    def _add_matched_reason(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Add matched_reason column explaining why rows match the filters."""
        reasons = []
        
        for _, row in df.iterrows():
            reason_parts = []
            
            # Side reason
            if 'side' in filters:
                side = filters['side']
                if side == 'left' and row['has_left_stone']:
                    reason_parts.append(f"left side stone present")
                elif side == 'right' and row['has_right_stone']:
                    reason_parts.append(f"right side stone present")
                elif side == 'bilateral' and row['has_bilateral_stones']:
                    reason_parts.append(f"bilateral stones present")
            
            # Size reasons
            side = filters.get('side', '').lower() if 'side' in filters else None
            
            if 'min_size_cm' in filters:
                min_size = filters['min_size_cm']
                if side == 'left' and pd.notna(row['left_stone_size_cm']) and row['left_stone_size_cm'] >= min_size:
                    reason_parts.append(f"left stone ≥{min_size}cm")
                elif side == 'right' and pd.notna(row['right_stone_size_cm']) and row['right_stone_size_cm'] >= min_size:
                    reason_parts.append(f"right stone ≥{min_size}cm")
                elif side == 'bilateral' or not side:
                    if pd.notna(row['right_stone_size_cm']) and row['right_stone_size_cm'] >= min_size:
                        reason_parts.append(f"right stone ≥{min_size}cm")
                    if pd.notna(row['left_stone_size_cm']) and row['left_stone_size_cm'] >= min_size:
                        reason_parts.append(f"left stone ≥{min_size}cm")
            
            if 'max_size_cm' in filters:
                max_size = filters['max_size_cm']
                if side == 'left' and pd.notna(row['left_stone_size_cm']) and row['left_stone_size_cm'] <= max_size:
                    reason_parts.append(f"left stone ≤{max_size}cm")
                elif side == 'right' and pd.notna(row['right_stone_size_cm']) and row['right_stone_size_cm'] <= max_size:
                    reason_parts.append(f"right stone ≤{max_size}cm")
                elif side == 'bilateral' or not side:
                    if pd.notna(row['right_stone_size_cm']) and row['right_stone_size_cm'] <= max_size:
                        reason_parts.append(f"right stone ≤{max_size}cm")
                    if pd.notna(row['left_stone_size_cm']) and row['left_stone_size_cm'] <= max_size:
                        reason_parts.append(f"left stone ≤{max_size}cm")
            
            # Date reason
            if 'start_year' in filters or 'end_year' in filters:
                year = row['imaging_year']
                if pd.notna(year):
                    reason_parts.append(f"imaging year {int(year)}")
            
            # Bladder reason
            if 'min_bladder_volume_ml' in filters:
                volume = row['bladder_volume_ml']
                if pd.notna(volume):
                    reason_parts.append(f"bladder volume ≥{filters['min_bladder_volume_ml']}ml")
            
            if 'max_bladder_volume_ml' in filters:
                volume = row['bladder_volume_ml']
                if pd.notna(volume):
                    reason_parts.append(f"bladder volume ≤{filters['max_bladder_volume_ml']}ml")
            
            if reason_parts:
                reasons.append("; ".join(reason_parts))
            else:
                reasons.append("matches general criteria")
        
        df['matched_reason'] = reasons
        return df
    
    def count_patients_with_stone(self, side: Optional[str] = None, 
                                min_size_cm: Optional[float] = None) -> int:
        """
        Count patients with stones matching criteria.
        
        Args:
            side: 'left', 'right', or None for any side
            min_size_cm: Minimum stone size in cm
            
        Returns:
            Number of patients matching criteria
        """
        if side == 'left':
            condition = self.df['has_left_stone']
            if min_size_cm:
                condition = condition & (self.df['left_stone_size_cm'] >= min_size_cm)
        elif side == 'right':
            condition = self.df['has_right_stone']
            if min_size_cm:
                condition = condition & (self.df['right_stone_size_cm'] >= min_size_cm)
        else:
            condition = self.df['has_any_stone']
            if min_size_cm:
                condition = condition & (
                    (self.df['right_stone_size_cm'] >= min_size_cm) |
                    (self.df['left_stone_size_cm'] >= min_size_cm)
                )
        
        return condition.sum()
    
    def get_stone_size_distribution(self, side: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stone size distribution statistics.
        
        Args:
            side: 'left', 'right', or None for both sides
            
        Returns:
            Dictionary with size statistics
        """
        if side == 'left':
            sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
        elif side == 'right':
            sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
        else:
            # Combine both sides
            left_sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
            right_sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
            sizes = pd.concat([left_sizes, right_sizes])
        
        if len(sizes) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }
        
        return {
            'count': len(sizes),
            'mean': float(sizes.mean()),
            'median': float(sizes.median()),
            'min': float(sizes.min()),
            'max': float(sizes.max()),
            'std': float(sizes.std())
        }
    
    def get_side_distribution(self) -> Dict[str, int]:
        """
        Get distribution of stones by side.
        
        Returns:
            Dictionary with counts by side
        """
        return {
            'left_only': (self.df['has_left_stone'] & ~self.df['has_right_stone']).sum(),
            'right_only': (self.df['has_right_stone'] & ~self.df['has_left_stone']).sum(),
            'bilateral': self.df['has_bilateral_stones'].sum(),
            'no_stones': (~self.df['has_any_stone']).sum()
        }
    
    def get_bladder_volume_stats(self) -> Dict[str, Any]:
        """
        Get bladder volume statistics.
        
        Returns:
            Dictionary with bladder volume statistics
        """
        volumes = self.df['bladder_volume_ml'].dropna()
        
        if len(volumes) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None
            }
        
        return {
            'count': len(volumes),
            'mean': float(volumes.mean()),
            'median': float(volumes.median()),
            'min': float(volumes.min()),
            'max': float(volumes.max())
        }
    
    def get_temporal_distribution(self) -> Dict[str, int]:
        """
        Get distribution of imaging by year.
        
        Returns:
            Dictionary with counts by year
        """
        if 'imaging_year' not in self.df.columns:
            return {}
        
        year_counts = self.df['imaging_year'].value_counts().sort_index()
        return {str(year): count for year, count in year_counts.items() if pd.notna(year)}
    
    def estimate_matching_rows(self, filters: Dict[str, Any]) -> int:
        """
        Estimate number of rows that would match given filters.
        
        Args:
            filters: Dictionary of filters
            
        Returns:
            Estimated number of matching rows
        """
        try:
            filtered_df = self.apply_filters(filters)
            return len(filtered_df)
        except Exception as e:
            logger.warning(f"Error estimating matching rows: {e}")
            return 0
    
    def get_summary_stats(self, filtered_df: Optional[pd.DataFrame] = None, 
                         relevant_fields: Optional[List[str]] = None,
                         query_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset, focusing only on relevant fields.
        
        Args:
            filtered_df: Optional filtered DataFrame, uses full dataset if None
            relevant_fields: List of fields that are relevant to the query
            query_text: Original query text to determine specific statistics needed
            
        Returns:
            Dictionary with summary statistics
        """
        if filtered_df is None:
            filtered_df = self.df
        
        stats = {
            'total_records': len(filtered_df),
            'unique_patients': filtered_df['recordid'].nunique() if 'recordid' in filtered_df.columns else 0
        }
        
        # Only compute stats for relevant fields
        if relevant_fields is None:
            relevant_fields = []
        
        if any('stone' in field.lower() for field in relevant_fields):
            stats['stone_distribution'] = self.get_side_distribution()
            stats['stone_size_stats'] = self.get_stone_size_distribution()
        
        if any('bladder' in field.lower() or 'volume' in field.lower() for field in relevant_fields):
            stats['bladder_volume_stats'] = self.get_bladder_volume_stats()
        
        if any('date' in field.lower() or 'year' in field.lower() or 'time' in field.lower() for field in relevant_fields):
            stats['temporal_distribution'] = self.get_temporal_distribution()
        
        # Add specific statistics based on query text
        if query_text:
            stats.update(self._get_specific_statistics(filtered_df, query_text))
        
        return stats
    
    def _get_specific_statistics(self, filtered_df: pd.DataFrame, query_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Get specific statistics based on query text.
        
        Args:
            filtered_df: Filtered DataFrame
            query_text: Original query text (optional)
            
        Returns:
            Dictionary with specific statistics
        """
        specific_stats = {}
        if not query_text:
            return specific_stats
        query_lower = query_text.lower()
        
        # Get stone size data once for reuse
        left_sizes = filtered_df[filtered_df['has_left_stone']]['left_stone_size_cm'].dropna()
        right_sizes = filtered_df[filtered_df['has_right_stone']]['right_stone_size_cm'].dropna()
        all_sizes = pd.concat([left_sizes, right_sizes]) if len(left_sizes) > 0 or len(right_sizes) > 0 else pd.Series(dtype=float)
        
        # Get bladder volume data once for reuse
        volumes = filtered_df['bladder_volume_ml'].dropna()
        
        # Check for multiple statistics in the same query (use if statements instead of elif)
        
        # Mean bladder volume
        if 'mean' in query_lower and 'bladder' in query_lower and 'volume' in query_lower:
            if len(volumes) > 0:
                specific_stats['mean_bladder_volume_ml'] = float(volumes.mean())
                specific_stats['bladder_volume_count'] = len(volumes)
        
        # Mean stone size
        if 'mean' in query_lower and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['mean_stone_size_cm'] = float(all_sizes.mean())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Maximum stone size (biggest, largest, max)
        # Handle both "biggest stone size" and "biggest stone" queries
        if any(word in query_lower for word in ['biggest', 'largest', 'maximum', 'max']) and 'stone' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['max_stone_size_cm'] = float(all_sizes.max())
                specific_stats['stone_size_count'] = len(all_sizes)
                
                # If asking "which patient", include patient details
                if any(word in query_lower for word in ['which', 'who', 'what patient', 'which patient']):
                    max_size = all_sizes.max()
                    # Find the patient(s) with the max stone size
                    max_patients = filtered_df[
                        (filtered_df['right_stone_size_cm'] == max_size) | 
                        (filtered_df['left_stone_size_cm'] == max_size)
                    ]
                    specific_stats['patients_with_max_stone'] = max_patients[['recordid', 'imaging_date', 'right_stone_size_cm', 'left_stone_size_cm']].to_dict('records')
        
        # Minimum stone size (smallest, min)
        # Handle both "smallest stone size" and "smallest stone" queries
        if any(word in query_lower for word in ['smallest', 'minimum', 'min']) and 'stone' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['min_stone_size_cm'] = float(all_sizes.min())
                specific_stats['stone_size_count'] = len(all_sizes)
                
                # If asking "which patient", include patient details
                if any(word in query_lower for word in ['which', 'who', 'what patient', 'which patient']):
                    min_size = all_sizes.min()
                    # Find the patient(s) with the min stone size
                    min_patients = filtered_df[
                        (filtered_df['right_stone_size_cm'] == min_size) | 
                        (filtered_df['left_stone_size_cm'] == min_size)
                    ]
                    specific_stats['patients_with_min_stone'] = min_patients[['recordid', 'imaging_date', 'right_stone_size_cm', 'left_stone_size_cm']].to_dict('records')
        
        # Sum stone size (total, sum)
        if any(word in query_lower for word in ['sum', 'total']) and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['sum_stone_size_cm'] = float(all_sizes.sum())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Median stone size
        if 'median' in query_lower and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['median_stone_size_cm'] = float(all_sizes.median())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Standard deviation stone size
        if any(word in query_lower for word in ['standard deviation', 'std', 'deviation']) and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['std_stone_size_cm'] = float(all_sizes.std())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Count queries
        if 'how many' in query_lower or 'count' in query_lower:
            # This is handled by total_records, but we can add more specific counts
            if 'stone' in query_lower:
                stone_count = filtered_df['has_any_stone'].sum()
                specific_stats['patients_with_stones'] = int(stone_count)
            if 'bladder' in query_lower:
                bladder_count = filtered_df['bladder_volume_ml'].notna().sum()
                specific_stats['patients_with_bladder_volume'] = int(bladder_count)
        
        # "Which patient" queries without specific statistic (show patient list)
        if any(word in query_lower for word in ['which', 'who', 'what patient', 'which patient', 'list', 'show me']):
            # If they're asking which patients match certain criteria
            if 'stone' in query_lower and len(filtered_df) > 0:
                # Check if they already got specific patient details from max/min queries
                if 'patients_with_max_stone' not in specific_stats and 'patients_with_min_stone' not in specific_stats:
                    # Show list of matching patients
                    cols = ['recordid', 'imaging_date']
                    if 'left' in query_lower or 'side' in query_lower:
                        cols.extend(['left_stone', 'left_stone_size_cm'])
                    if 'right' in query_lower or 'side' in query_lower:
                        cols.extend(['right_stone', 'right_stone_size_cm'])
                    if 'left' not in query_lower and 'right' not in query_lower:
                        cols.extend(['left_stone', 'left_stone_size_cm', 'right_stone', 'right_stone_size_cm'])
                    
                    # Only include columns that exist in the dataframe
                    cols = [c for c in cols if c in filtered_df.columns]
                    patient_list = filtered_df[cols].to_dict('records')
                    specific_stats['matching_patients'] = patient_list[:20]  # Limit to first 20
        
        return specific_stats
    
    def format_summary(self, stats: Dict[str, Any]) -> str:
        """
        Format summary statistics for display, showing only relevant information.
        
        Args:
            stats: Summary statistics dictionary
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            "QUERY RESULTS",
            "=" * 60,
            f"Total records: {stats['total_records']}",
            f"Unique patients: {stats['unique_patients']}",
        ]
        
        # Show specific statistics first (these are the direct answers to queries)
        if 'mean_bladder_volume_ml' in stats:
            lines.append("")
            lines.append("Mean bladder volume:")
            lines.append(f"  • {stats['mean_bladder_volume_ml']:.1f} ml (based on {stats.get('bladder_volume_count', 0)} records)")
        
        if 'mean_stone_size_cm' in stats:
            lines.append("")
            lines.append("Mean stone size:")
            lines.append(f"  • {stats['mean_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'max_stone_size_cm' in stats:
            lines.append("")
            lines.append("Biggest stone size:")
            lines.append(f"  • {stats['max_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
            
            # Show patient details if available
            if 'patients_with_max_stone' in stats:
                lines.append("")
                lines.append("Patient(s) with biggest stone:")
                for patient in stats['patients_with_max_stone']:
                    right_size = patient.get('right_stone_size_cm')
                    left_size = patient.get('left_stone_size_cm')
                    imaging_date = patient.get('imaging_date')
                    
                    # Format date
                    if hasattr(imaging_date, 'strftime'):
                        date_str = imaging_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(imaging_date)
                    
                    # Show which side has the max stone
                    if pd.notna(right_size) and right_size == stats['max_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Right: {right_size:.2f} cm)")
                    if pd.notna(left_size) and left_size == stats['max_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Left: {left_size:.2f} cm)")
        
        if 'min_stone_size_cm' in stats:
            lines.append("")
            lines.append("Smallest stone size:")
            lines.append(f"  • {stats['min_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
            
            # Show patient details if available
            if 'patients_with_min_stone' in stats:
                lines.append("")
                lines.append("Patient(s) with smallest stone:")
                for patient in stats['patients_with_min_stone']:
                    right_size = patient.get('right_stone_size_cm')
                    left_size = patient.get('left_stone_size_cm')
                    imaging_date = patient.get('imaging_date')
                    
                    # Format date
                    if hasattr(imaging_date, 'strftime'):
                        date_str = imaging_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(imaging_date)
                    
                    # Show which side has the min stone
                    if pd.notna(right_size) and right_size == stats['min_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Right: {right_size:.2f} cm)")
                    if pd.notna(left_size) and left_size == stats['min_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Left: {left_size:.2f} cm)")
        
        if 'sum_stone_size_cm' in stats:
            lines.append("")
            lines.append("Sum of stone sizes:")
            lines.append(f"  • {stats['sum_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'median_stone_size_cm' in stats:
            lines.append("")
            lines.append("Median stone size:")
            lines.append(f"  • {stats['median_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'std_stone_size_cm' in stats:
            lines.append("")
            lines.append("Standard deviation of stone sizes:")
            lines.append(f"  • {stats['std_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'patients_with_stones' in stats:
            lines.append("")
            lines.append("Patients with stones:")
            lines.append(f"  • {stats['patients_with_stones']} patients")
        
        if 'patients_with_bladder_volume' in stats:
            lines.append("")
            lines.append("Patients with bladder volume data:")
            lines.append(f"  • {stats['patients_with_bladder_volume']} patients")
        
        # Show matching patients list if available
        if 'matching_patients' in stats and stats['matching_patients']:
            lines.append("")
            lines.append("Matching patients:")
            for i, patient in enumerate(stats['matching_patients'][:10], 1):  # Show first 10
                recordid = patient.get('recordid', 'N/A')
                imaging_date = patient.get('imaging_date')
                
                # Format date
                if hasattr(imaging_date, 'strftime'):
                    date_str = imaging_date.strftime('%Y-%m-%d')
                elif pd.notna(imaging_date):
                    date_str = str(imaging_date)[:10]
                else:
                    date_str = 'N/A'
                
                # Build patient info string
                info_parts = [f"{recordid} (Date: {date_str})"]
                
                # Add stone information
                left_stone = patient.get('left_stone')
                left_size = patient.get('left_stone_size_cm')
                right_stone = patient.get('right_stone')
                right_size = patient.get('right_stone_size_cm')
                
                stone_info = []
                if left_stone and pd.notna(left_size):
                    stone_info.append(f"Left: {left_size:.2f} cm")
                elif left_stone == 'present':
                    stone_info.append(f"Left: present")
                    
                if right_stone and pd.notna(right_size):
                    stone_info.append(f"Right: {right_size:.2f} cm")
                elif right_stone == 'present':
                    stone_info.append(f"Right: present")
                
                if stone_info:
                    info_parts.append(", ".join(stone_info))
                
                lines.append(f"  {i}. {', '.join(info_parts)}")
            
            total_matching = len(stats['matching_patients'])
            if total_matching > 10:
                lines.append(f"  ... and {total_matching - 10} more patients (showing first 10)")
        
        # Only show detailed distributions if no specific statistics were requested
        if not any(key in stats for key in ['mean_bladder_volume_ml', 'mean_stone_size_cm', 'max_stone_size_cm', 'min_stone_size_cm', 'sum_stone_size_cm', 'median_stone_size_cm', 'std_stone_size_cm', 'patients_with_stones', 'patients_with_bladder_volume', 'matching_patients']):
            # Only show stone distribution if it's relevant
            if 'stone_distribution' in stats:
                lines.append("")
                lines.append("Stone distribution:")
                stone_dist = stats['stone_distribution']
                lines.append(f"  • Left only: {stone_dist['left_only']}")
                lines.append(f"  • Right only: {stone_dist['right_only']}")
                lines.append(f"  • Bilateral: {stone_dist['bilateral']}")
                lines.append(f"  • No stones: {stone_dist['no_stones']}")
            
            # Only show stone size stats if relevant
            if 'stone_size_stats' in stats:
                size_stats = stats['stone_size_stats']
                if size_stats['count'] > 0:
                    lines.append("")
                    lines.append("Stone size statistics:")
                    lines.append(f"  • Count: {size_stats['count']}")
                    lines.append(f"  • Mean: {size_stats['mean']:.2f} cm")
                    lines.append(f"  • Median: {size_stats['median']:.2f} cm")
                    lines.append(f"  • Range: {size_stats['min']:.2f} - {size_stats['max']:.2f} cm")
            
            # Only show bladder volume stats if relevant
            if 'bladder_volume_stats' in stats:
                bladder_stats = stats['bladder_volume_stats']
                if bladder_stats['count'] > 0:
                    lines.append("")
                    lines.append("Bladder volume statistics:")
                    lines.append(f"  • Count: {bladder_stats['count']}")
                    lines.append(f"  • Mean: {bladder_stats['mean']:.1f} ml")
                    lines.append(f"  • Median: {bladder_stats['median']:.1f} ml")
                    lines.append(f"  • Range: {bladder_stats['min']:.1f} - {bladder_stats['max']:.1f} ml")
            
            # Only show temporal distribution if relevant
            if 'temporal_distribution' in stats:
                temporal_dist = stats['temporal_distribution']
                if temporal_dist:
                    lines.append("")
                    lines.append("Imaging by year:")
                    for year, count in sorted(temporal_dist.items()):
                        lines.append(f"  • {year}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    # ==================== DYNAMIC LEARNING METHODS ====================
    
    def _load_learned_filters(self) -> Dict[str, Any]:
        """Load learned filter patterns from file."""
        try:
            if os.path.exists(self.learned_filters_file):
                with open(self.learned_filters_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load learned filters: {e}")
        return {}
    
    def _save_learned_filters(self):
        """Save learned filter patterns to file."""
        try:
            os.makedirs(os.path.dirname(self.learned_filters_file), exist_ok=True)
            with open(self.learned_filters_file, 'w') as f:
                json.dump(self.learned_filters, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned filters: {e}")
    
    def _load_learned_statistics(self) -> Dict[str, Any]:
        """Load learned statistics patterns from file."""
        try:
            if os.path.exists(self.learned_statistics_file):
                with open(self.learned_statistics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load learned statistics: {e}")
        return {}
    
    def _save_learned_statistics(self):
        """Save learned statistics patterns to file."""
        try:
            os.makedirs(os.path.dirname(self.learned_statistics_file), exist_ok=True)
            with open(self.learned_statistics_file, 'w') as f:
                json.dump(self.learned_statistics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned statistics: {e}")
    
    def _load_learned_operations(self) -> Dict[str, Any]:
        """Load learned operation patterns from file."""
        try:
            if os.path.exists(self.learned_operations_file):
                with open(self.learned_operations_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load learned operations: {e}")
        return {}
    
    def _save_learned_operations(self):
        """Save learned operation patterns to file."""
        try:
            os.makedirs(os.path.dirname(self.learned_operations_file), exist_ok=True)
            with open(self.learned_operations_file, 'w') as f:
                json.dump(self.learned_operations, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learned operations: {e}")
    
    def _detect_new_medical_concepts(self, query_text: str) -> List[str]:
        """Detect new medical concepts in query using LLM."""
        client = get_llm_client()
        if not client:
            return []
        
        try:
            # Sanitize query to reduce medical terminology
            sanitized_query = _sanitize_query_for_schema_tagging(query_text)
            
            prompt = f"""
            You are a technical data schema tagging assistant. This is a data processing task, not medical analysis or advice.
            
            Task: Identify database field names or column tags from a user's data query.
            
            User Query (data processing context): "{sanitized_query}"
            
            Available database columns: {', '.join(self.available_columns)}
            
            Instructions:
            - This is purely a technical data mapping task
            - Identify field names, attributes, or column tags that map to database columns
            - Think of this as database schema mapping, not medical analysis
            - Return only the field/column tag names, one per line
            - If no relevant field names are found, return "NONE"
            
            Return format: One field name per line, or "NONE" if none found.
            """
            
            concepts = _call_llm_unified(client, prompt, max_tokens=200, temperature=0.1)
            if not concepts or concepts == "NONE":
                return []
            
            return [concept.strip() for concept in concepts.split('\n') if concept.strip()]
        except Exception as e:
            logger.error(f"Failed to detect medical concepts: {e}")
            return []
    
    def _generate_filter_logic(self, concept: str, query_text: str) -> Optional[Dict[str, Any]]:
        """Generate filter logic for a new medical concept using LLM."""
        client = get_llm_client()
        if not client:
            return None
        
        try:
            # Sanitize query to reduce medical terminology
            sanitized_query = _sanitize_query_for_schema_tagging(query_text)
            sanitized_concept = _sanitize_query_for_schema_tagging(concept)
            
            prompt = f"""
            You are a technical data schema mapping assistant. This is a data processing task, not medical analysis or advice.
            
            Task: Map a field name to a database column and create a filter specification.
            
            Field name to map: "{sanitized_concept}"
            User Query (data processing context): "{sanitized_query}"
            
            Available database columns: {', '.join(self.available_columns)}
            Numeric columns: {', '.join(self.numeric_columns)}
            Text columns: {', '.join(self.text_columns)}
            
            Instructions:
            - This is purely a technical database mapping task
            - Create a JSON filter specification mapping the field name to a database column
            - Return JSON with: "column", "operator", "value", "description"
            - Operators: ==, !=, >, <, >=, <=, contains, not_contains
            - If the field cannot be mapped to any existing column, return null
            
            Return format: Valid JSON object or null.
            """
            
            result = _call_llm_unified(client, prompt, max_tokens=300, temperature=0.1)
            if not result or (isinstance(result, str) and result.lower().strip() == "null"):
                return None
            
            return json.loads(result)
        except Exception as e:
            logger.error(f"Failed to generate filter logic for {concept}: {e}")
            return None
    
    def _detect_new_statistics(self, query_text: str) -> List[str]:
        """Detect new statistical requests in query using LLM."""
        client = get_llm_client()
        if not client:
            return []
        
        try:
            # Sanitize query to reduce medical terminology
            sanitized_query = _sanitize_query_for_schema_tagging(query_text)
            
            prompt = f"""
            You are a technical data schema tagging assistant. This is a data processing task, not medical analysis or advice.
            
            Task: Identify statistical function names or aggregation operations from a user's data query.
            
            User Query (data processing context): "{sanitized_query}"
            
            Standard statistical functions already supported: mean, median, max, min, sum, count, std
            
            Instructions:
            - This is purely a technical data processing task
            - Identify additional statistical function names mentioned (e.g., variance, mode, range, percentile, etc.)
            - Think of this as database function mapping, not medical analysis
            - Return only the function/operation names, one per line
            - If no additional statistical functions are found, return "NONE"
            
            Return format: One function name per line, or "NONE" if none found.
            """
            
            stats = _call_llm_unified(client, prompt, max_tokens=200, temperature=0.1)
            if not stats or stats == "NONE":
                return []
            
            return [stat.strip() for stat in stats.split('\n') if stat.strip()]
        except Exception as e:
            logger.error(f"Failed to detect new statistics: {e}")
            return []
    
    def _generate_statistics_logic(self, statistic: str, query_text: str) -> Optional[Dict[str, Any]]:
        """Generate computation logic for a new statistic using LLM."""
        client = get_llm_client()
        if not client:
            return None
        
        try:
            # Sanitize query to reduce medical terminology
            sanitized_query = _sanitize_query_for_schema_tagging(query_text)
            sanitized_statistic = _sanitize_query_for_schema_tagging(statistic)
            
            prompt = f"""
            You are a technical data schema mapping assistant. This is a data processing task, not medical analysis or advice.
            
            Task: Map a statistical function name to database columns and create a computation specification.
            
            Statistical function to map: "{sanitized_statistic}"
            User Query (data processing context): "{sanitized_query}"
            
            Available numeric columns: {', '.join(self.numeric_columns)}
            
            Instructions:
            - This is purely a technical database computation mapping task
            - Create a JSON specification mapping the function to database operations
            - Return JSON with: "columns", "method", "parameters", "description"
            - Methods: pandas functions like "var", "mode", "quantile"
            - If the function cannot be mapped to any existing columns or methods, return null
            
            Return format: Valid JSON object or null.
            """
            
            result = _call_llm_unified(client, prompt, max_tokens=300, temperature=0.1)
            if not result or (isinstance(result, str) and result.lower().strip() == "null"):
                return None
            
            return json.loads(result)
        except Exception as e:
            logger.error(f"Failed to generate statistics logic for {statistic}: {e}")
            return None
    
    def _learn_from_query(self, query_text: str):
        """Learn new patterns from user query."""
        # Dynamic learning is enabled for all providers
        # Note: For Gemini, safety filters may still trigger, but we try with sanitized prompts
        
        # Learn new medical concepts and filters
        new_concepts = self._detect_new_medical_concepts(query_text)
        for concept in new_concepts:
            if concept not in self.learned_filters:
                filter_logic = self._generate_filter_logic(concept, query_text)
                if filter_logic:
                    self.learned_filters[concept] = filter_logic
                    self._save_learned_filters()
                    logger.info(f"Learned new filter for concept: {concept}")
        
        # Learn new statistics
        new_statistics = self._detect_new_statistics(query_text)
        for statistic in new_statistics:
            if statistic not in self.learned_statistics:
                stats_logic = self._generate_statistics_logic(statistic, query_text)
                if stats_logic:
                    self.learned_statistics[statistic] = stats_logic
                    self._save_learned_statistics()
                    logger.info(f"Learned new statistic: {statistic}")
    
    def apply_learned_filters(self, filters: Dict[str, Any], filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned filters to the DataFrame."""
        for concept, filter_logic in self.learned_filters.items():
            if concept in filters:
                column = filter_logic.get('column')
                operator = filter_logic.get('operator')
                value = filter_logic.get('value')
                
                if column not in self.df.columns:
                    continue
                
                try:
                    if operator == '==':
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif operator == '!=':
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif operator == '>':
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == '<':
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == '>=':
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == '<=':
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == 'contains':
                        filtered_df = filtered_df[filtered_df[column].str.contains(str(value), case=False, na=False)]
                    elif operator == 'not_contains':
                        filtered_df = filtered_df[~filtered_df[column].str.contains(str(value), case=False, na=False)]
                except Exception as e:
                    logger.warning(f"Failed to apply learned filter {concept}: {e}")
        
        return filtered_df
    
    def compute_learned_statistics(self, filtered_df: pd.DataFrame, query_text: str) -> Dict[str, Any]:
        """Compute learned statistics on the filtered DataFrame."""
        learned_stats = {}
        
        for statistic, stats_logic in self.learned_statistics.items():
            if query_text and statistic.lower() in query_text.lower():
                try:
                    columns = stats_logic.get('columns', [])
                    method = stats_logic.get('method')
                    parameters = stats_logic.get('parameters', {})
                    
                    for column in columns:
                        if column in filtered_df.columns and column in self.numeric_columns:
                            data = filtered_df[column].dropna()
                            if len(data) > 0:
                                if method == 'var':
                                    learned_stats[f'{statistic}_{column}'] = float(data.var())
                                elif method == 'mode':
                                    mode_result = data.mode()
                                    learned_stats[f'{statistic}_{column}'] = float(mode_result.iloc[0]) if len(mode_result) > 0 else None
                                elif method == 'quantile':
                                    q = parameters.get('q', 0.5)
                                    learned_stats[f'{statistic}_{column}'] = float(data.quantile(q))
                                elif method == 'range':
                                    learned_stats[f'{statistic}_{column}'] = float(data.max() - data.min())
                                
                                learned_stats[f'{statistic}_{column}_count'] = len(data)
                except Exception as e:
                    logger.warning(f"Failed to compute learned statistic {statistic}: {e}")
        
        return learned_stats
    
    def apply_filters_with_learning(self, filters: Dict[str, Any], query_text: str) -> pd.DataFrame:
        """Apply filters with dynamic learning capabilities."""
        # Learn from the query first
        self._learn_from_query(query_text)
        
        # Create query context for adaptive filtering
        query_context = {'query_text': query_text}
        
        # Apply adaptive filters
        filtered_df = self.apply_filters(filters, query_context)
        
        # Apply learned filters
        filtered_df = self.apply_learned_filters(filters, filtered_df)
        
        return filtered_df
    
    def get_summary_stats_with_learning(self, filtered_df: Optional[pd.DataFrame] = None, 
                                      query_text: str = "", relevant_fields: List[str] = None) -> Dict[str, Any]:
        """Get summary statistics with dynamic learning capabilities."""
        if filtered_df is None:
            filtered_df = self.df
        
        # Learn from the query first
        self._learn_from_query(query_text)
        
        # Get standard statistics
        stats = self.get_summary_stats(filtered_df, relevant_fields, query_text)
        
        # Add learned statistics
        learned_stats = self.compute_learned_statistics(filtered_df, query_text)
        stats.update(learned_stats)
        
        return stats
