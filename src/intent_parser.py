"""Intent parser for natural language queries using LLM-only approach."""

import json
from typing import Dict, Any, List, Optional
import logging
import pandas as pd

from .llm_schema import UserQuery, PlanSummary
from .config import get_llm_client, LLM_PROVIDER, MODEL_NAME

logger = logging.getLogger(__name__)

class IntentParser:
    """Parse natural language queries into structured filters and goals using LLM-only approach."""
    
    def __init__(self):
        """Initialize intent parser."""
        pass
    
    # ==================== LLM-BASED SCHEMA ANALYSIS & PROMPT GENERATION ====================
    
    def _generate_dataset_schema_info(self, df: pd.DataFrame) -> str:
        """
        Generate dataset schema information string for LLM analysis.
        
        Args:
            df: Structured DataFrame to analyze
            
        Returns:
            String describing the dataset schema with basic statistics
        """
        schema_info = []
        schema_info.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.\n")
        schema_info.append("Column information:")
        
        for col in df.columns:
            col_info = f"\n- {col}:"
            col_info += f"\n  Type: {df[col].dtype}"
            col_info += f"\n  Nullable: {df[col].isna().any()}"
            
            # Get basic statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_info += f"\n  Min: {non_null.min()}, Max: {non_null.max()}, Mean: {non_null.mean():.2f}"
                    col_info += f"\n  Sample values: {non_null.head(3).tolist()}"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    col_info += f"\n  Date range: {dates.min()} to {dates.max()}"
                    years = dates.dt.year
                    col_info += f"\n  Year range: {years.min()} to {years.max()}"
            else:
                # Categorical/text
                unique_vals = df[col].dropna().unique()
                unique_count = len(unique_vals)
                col_info += f"\n  Unique values: {unique_count}"
                if unique_count <= 10:
                    col_info += f"\n  Values: {list(unique_vals)}"
                else:
                    col_info += f"\n  Sample values: {list(unique_vals[:10])} (showing first 10)"
                col_info += f"\n  Sample data: {df[col].dropna().head(3).tolist()}"
            
            schema_info.append(col_info)
        
        return "\n".join(schema_info)
    
    def _generate_prompt_with_llm(self, query: str, df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate comprehensive prompt for query parsing using LLM-based schema analysis.
        
        Args:
            query: User's natural language query
            df: Optional structured DataFrame for schema analysis
            
        Returns:
            Complete prompt string for LLM to parse the query
        """
        client = get_llm_client()
        if not client:
            logger.warning(f"{LLM_PROVIDER.upper()} client not available for prompt generation")
            return self._generate_fallback_prompt(query)
        
        # If DataFrame is provided, analyze schema with LLM
        if df is not None and len(df) > 0:
            logger.info("Using LLM to analyze dataset schema and generate prompt")
            
            # Generate basic schema info for LLM
            schema_info = self._generate_dataset_schema_info(df)
            
            # Use LLM to generate schema description and filter suggestions
            schema_analysis_prompt = """
            Analyze this clinical/medical dataset schema and generate a comprehensive description for query parsing.
            
            Dataset Schema Information:
            """ + schema_info + """
            
            Generate two sections:
            
            1. SCHEMA DESCRIPTION: 
            Create a clear description of available data columns. For each column, include:
            - Column name
            - Data type and description
            - Value ranges (for numeric) or possible values (for categorical)
            - Any relevant context about the column's meaning
            
            IMPORTANT: If there are text/narrative columns (like "narrative"), emphasize that these columns contain unstructured text that may include medical concepts, findings, conditions, or observations mentioned in the original reports. Medical queries about conditions, findings, or concepts NOT in structured columns should search in narrative/text columns.
            
            Format: List each column with "- column_name: description (additional context)"
            
            2. FILTER SUGGESTIONS:
            Suggest ONLY these allowed filter keys:
            
            REQUIRED FILTER KEYS (always available):
            - "start_year": integer, or null (earliest year filter)
            - "end_year": integer, or null (latest year filter)
            - "narrative_contains_any": array of strings, or null (search narrative for ANY of these terms)
            - "narrative_contains_all": array of strings, or null (search narrative for ALL of these terms)
            - "narrative_contains_none": array of strings, or null (exclude rows with ANY of these terms)
            - "distinct": string ("recordid" for patient counting), or null
            
            OPTIONAL FILTER KEYS (only if column exists in dataset):
            - For each numeric column: "min_[column_name]": numeric, or null
            - For each numeric column: "max_[column_name]": numeric, or null
            - For each categorical column: "[column_name]": value (actual values from dataset), or null
            
            CRITICAL RULES:
            - Do NOT invent filter keys like "stone_presence", "has_hydronephrosis", etc.
            - If a medical concept is not in structured columns, use "narrative_contains_any": ["concept"]
            - Always suggest "distinct": "recordid" if query asks about "patients"
            
            Format: List each filter key with "- filter_key: description (type), or null"
            
            Return a JSON object with this structure:
            {
                "schema_description": "Available data columns in the dataset:\\n- column1: description...\\n- column2: description...",
                "filter_suggestions": "Available filter keys:\\n- filter_key1: description (type), or null\\n- filter_key2: description (type), or null"
            }
            """
            
            # Initialize variables
            schema_description = ''
            filter_suggestions = ''
            
            try:
                # Call LLM to analyze schema
                if LLM_PROVIDER == "openai":
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": schema_analysis_prompt}],
                        temperature=0.1,
                        max_tokens=2000
                    )
                    schema_response = response.choices[0].message.content.strip()
                elif LLM_PROVIDER == "gemini":
                    import google.generativeai as genai
                    
                    # Get model (similar to query parsing)
                    model = None
                    model_variants = [
                        MODEL_NAME,
                        f"{MODEL_NAME}-latest",
                        MODEL_NAME.replace("gemini-1.5-flash", "gemini-2.5-flash"),
                        MODEL_NAME.replace("gemini-1.5-pro", "gemini-2.5-pro"),
                        MODEL_NAME.replace("gemini-1.5-flash", "gemini-1.5-flash-latest"),
                        MODEL_NAME.replace("gemini-1.5-pro", "gemini-1.5-pro-latest"),
                        MODEL_NAME.replace("gemini-2.5-flash", "gemini-1.5-flash-latest"),
                        MODEL_NAME.replace("gemini-2.5-pro", "gemini-1.5-pro-latest"),
                    ]
                    
                    for variant in model_variants:
                        try:
                            model = genai.GenerativeModel(variant)
                            _ = model._model_name
                            logger.info(f"Using Gemini model for schema analysis: {variant}")
                            break
                        except Exception:
                            continue
                    
                    if model is None:
                        # Fallback to available models
                        try:
                            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            available_short_names = [m.split('/')[-1] for m in available_models]
                            preferred_order = []
                            if 'flash' in MODEL_NAME.lower():
                                preferred_order = [m for m in available_short_names if 'flash' in m.lower()]
                            elif 'pro' in MODEL_NAME.lower():
                                preferred_order = [m for m in available_short_names if 'pro' in m.lower()]
                            if not preferred_order:
                                preferred_order = available_short_names
                            
                            for model_short_name in preferred_order:
                                try:
                                    model = genai.GenerativeModel(model_short_name)
                                    logger.info(f"Using auto-detected model for schema analysis: {model_short_name}")
                                    break
                                except Exception:
                                    continue
                        except Exception as e:
                            logger.warning(f"Could not list models: {e}")
                    
                    if model is None:
                        raise ValueError(f"Could not find a valid Gemini model for schema analysis")
                    
                    from google.generativeai.types import HarmCategory, HarmBlockThreshold
                    try:
                        safety_settings = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    except (AttributeError, ValueError):
                        safety_settings = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT: 1,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: 1,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 1,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 0,
                        }
                    
                    generation_config = {
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    }
                    response = model.generate_content(
                        schema_analysis_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    schema_response = response.text.strip()
                else:
                    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
                
                # Parse LLM response
                schema_analysis = json.loads(schema_response)
                schema_description = schema_analysis.get('schema_description', '')
                filter_suggestions = schema_analysis.get('filter_suggestions', '')
                
                logger.info("LLM schema analysis successful")
                
            except Exception as e:
                logger.error(f"LLM schema analysis failed: {e}, using fallback")
                return self._generate_fallback_prompt(query, df)
        else:
            # No DataFrame provided, use fallback
            logger.info("No dataset provided, using fallback prompt")
            return self._generate_fallback_prompt(query)
        
        # Build final prompt with LLM-generated schema sections
        prompt = f"""
        You are parsing a medical/clinical query and converting it into structured filters based on the dataset schema.
        
        {schema_description}
        
        User Query: "{query}"
        
        Extract the following information:
        1. Goal: What is the user trying to accomplish? (e.g., "Count patients", "Find patients", "Calculate statistics")
        2. Filters: Create appropriate filters based on the query. {filter_suggestions}
           
           General filter patterns:
           - For numeric columns: use "min_[column_name]" and "max_[column_name]" for range filtering
           - For categorical columns: use the column name directly or a presence/status filter
           - For date columns: use "start_year" and "end_year" for year-based filtering
           - For text/narrative columns: If a medical concept (like hydronephrosis, infection, etc.) is mentioned in the query but not in structured columns, search in narrative/text columns. 
             The structured DataFrame may have "history_summary" column (which contains extracted narrative text) instead of "narrative" column.
             Use "narrative_contains_any": ["concept"] filter - the system will automatically search in the appropriate text column (narrative, history_summary, etc.)
           - Only include filters that are explicitly mentioned or clearly implied in the query
           
           CRITICAL: If the query mentions a medical condition or concept (e.g., "hydronephrosis", "infection", "pain") that is not in the structured columns list above, you MUST use "narrative_contains_any": ["concept"] filter. The system will search in the appropriate text column (narrative, history_summary, or other text columns).
        3. Outputs: What data columns should be returned? List actual column names from the available columns above.
        4. Input fields: What data fields are referenced in the query? List actual column names from available columns.
        5. Assumptions: What assumptions are being made? (optional)
        
        Return ONLY a JSON object with this exact structure:
        {{
            "goal": "string",
            "filters": {{"key": "value"}},
            "outputs": ["field1", "field2"],
            "input_fields": ["field1", "field2"],
            "assumptions": ["assumption1", "assumption2"]
        }}
        
        CRITICAL FILTER SCHEMA (MUST FOLLOW):
        Only these filter keys are allowed. If concept is not a dataframe column, use narrative filters:
        
        ALLOWED FILTER KEYS:
        - "start_year": integer (earliest year, >=)
        - "end_year": integer (latest year, <=)
        - "narrative_contains_any": array of strings (search for ANY of these terms in narrative)
        - "narrative_contains_all": array of strings (search for ALL of these terms in narrative)
        - "narrative_contains_none": array of strings (exclude rows containing ANY of these terms)
        - "distinct": string (column name to deduplicate by, usually "recordid" for patient counting)
        
        For structured columns that actually exist in the dataset:
        - "[column_name]": value (direct column filter, only if column exists)
        - "min_[column_name]": numeric (for numeric columns only)
        - "max_[column_name]": numeric (for numeric columns only)
        
        RULES:
        1. If a medical concept is NOT in the structured columns list above, you MUST use "narrative_contains_any": ["concept"]
        2. If query asks for "patients" or "patient", always include "distinct": "recordid" in filters
        3. Do NOT invent filter keys like "stone_presence", "has_hydronephrosis", etc. Use only the ALLOWED FILTER KEYS above
        4. For medical synonyms, include all variants in "narrative_contains_any": ["term1", "term2", ...]
        
        Examples (CORRECT):
        - "how many patients have hydronephrosis?" → 
          If "right_hydronephrosis" and "left_hydronephrosis" columns exist: {{"goal": "Count patients", "filters": {{"right_hydronephrosis": "present", "left_hydronephrosis": "present", "distinct": "recordid"}}, "outputs": ["recordid"], "input_fields": ["right_hydronephrosis", "left_hydronephrosis"], "assumptions": ["Using structured hydronephrosis columns"]}}
          If columns don't exist: {{"goal": "Count patients", "filters": {{"narrative_contains_any": ["hydronephrosis"], "distinct": "recordid"}}, "outputs": ["recordid"], "input_fields": ["narrative"], "assumptions": ["Hydronephrosis not in structured columns, searching narrative"]}}
        - "which patient has left hydronephrosis?" → 
          If "left_hydronephrosis" column exists: {{"goal": "Find patients", "filters": {{"left_hydronephrosis": "present", "distinct": "recordid"}}, "outputs": ["recordid"], "input_fields": ["left_hydronephrosis"], "assumptions": ["Using structured left_hydronephrosis column"]}}
        - "find records from 2023" → {{"goal": "Find records", "filters": {{"start_year": 2023, "end_year": 2023}}, "outputs": ["recordid", "imaging_date"], "input_fields": ["imaging_date"], "assumptions": []}}
        """
        
        return prompt
    
    def _generate_fallback_prompt(self, query: str, df: Optional[pd.DataFrame] = None) -> str:
        """Generate fallback prompt when LLM schema analysis is unavailable."""
        if df is not None and len(df.columns) > 0:
            # Build basic schema from DataFrame columns
            schema_lines = ["Available data columns in the dataset:"]
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        schema_lines.append(f"  - {col}: Numeric column (range: {non_null.min()} to {non_null.max()})")
                    else:
                        schema_lines.append(f"  - {col}: Numeric column")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    schema_lines.append(f"  - {col}: Date/time column")
                else:
                    unique_count = df[col].nunique()
                    schema_lines.append(f"  - {col}: Categorical/text column ({unique_count} unique values)")
            schema_description = "\n".join(schema_lines)
        else:
            schema_description = """
            Available data columns in the dataset:
            - recordid: Patient record ID
            - imaging_date: Date of imaging
            """
        
        prompt = f"""
        You are parsing a medical/clinical query and converting it into structured filters based on the dataset schema.
        
        {schema_description}
        
        User Query: "{query}"
        
        Extract the following information:
        1. Goal: What is the user trying to accomplish? (e.g., "Count patients", "Find patients", "Calculate statistics")
        2. Filters: Create appropriate filters based on the query and available columns.
           - For numeric columns: use "min_[column_name]" and "max_[column_name]" for range filtering
           - For categorical columns: use the column name directly
           - For date columns: use "start_year" and "end_year" for year-based filtering
        3. Outputs: What data columns should be returned? List actual column names from available columns.
        4. Input fields: What data fields are referenced in the query? List actual column names.
        5. Assumptions: What assumptions are being made? (optional)
        
        Return ONLY a JSON object with this exact structure:
        {{
            "goal": "string",
            "filters": {{"key": "value"}},
            "outputs": ["field1", "field2"],
            "input_fields": ["field1", "field2"],
            "assumptions": ["assumption1", "assumption2"]
        }}
        """
        return prompt
    
    # ==================== LLM-ONLY PARSING ====================
    
    def parse_query(self, query: str, structured_df: Optional[pd.DataFrame] = None) -> UserQuery:
        """
        Parse a natural language query using LLM-only approach.
        
        Args:
            query: Natural language query string
            structured_df: Optional structured DataFrame. If provided, schema will be analyzed
                          using LLM and used for prompt generation.
            
        Returns:
            UserQuery object with parsed components
        """
        logger.info(f"Parsing query with LLM: {query}")
        
        return self._parse_query_with_llm(query, structured_df)
    
    def _parse_query_with_llm(self, query: str, df: Optional[pd.DataFrame] = None) -> UserQuery:
        """
        Parse query using LLM for better accuracy and natural language understanding.
        
        Args:
            query: User's natural language query
            df: Optional structured DataFrame for schema-aware prompt generation
        """
        logger.info(f"Using LLM to parse query: {query}")
        
        # Check if we have LLM client available
        client = get_llm_client()
        if not client:
            logger.warning(f"{LLM_PROVIDER.upper()} client not available, falling back to empty query")
            return self._create_empty_user_query(query)
        
        try:
            # Generate prompt using LLM-based schema analysis
            prompt = self._generate_prompt_with_llm(query, df)
            
            # Call appropriate API based on provider to parse the query
            if LLM_PROVIDER == "openai":
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
            elif LLM_PROVIDER == "gemini":
                import google.generativeai as genai
                
                # Try different model name formats (including 2.5 models)
                model = None
                model_variants = [
                    MODEL_NAME,
                    f"{MODEL_NAME}-latest",
                    MODEL_NAME.replace("gemini-1.5-flash", "gemini-2.5-flash"),
                    MODEL_NAME.replace("gemini-1.5-pro", "gemini-2.5-pro"),
                    MODEL_NAME.replace("gemini-1.5-flash", "gemini-1.5-flash-latest"),
                    MODEL_NAME.replace("gemini-1.5-pro", "gemini-1.5-pro-latest"),
                    MODEL_NAME.replace("gemini-2.5-flash", "gemini-1.5-flash-latest"),
                    MODEL_NAME.replace("gemini-2.5-pro", "gemini-1.5-pro-latest"),
                ]
                
                for variant in model_variants:
                    try:
                        model = genai.GenerativeModel(variant)
                        _ = model._model_name
                        logger.info(f"Using Gemini model for intent parsing: {variant}")
                        break
                    except Exception as e:
                        logger.debug(f"Model variant {variant} failed: {e}")
                        continue
                
                if model is None:
                    # Last resort: try to list available models and use the first matching one
                    try:
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        logger.info(f"Available Gemini models: {available_models}")
                        available_short_names = [m.split('/')[-1] for m in available_models]
                        
                        preferred_order = []
                        if 'flash' in MODEL_NAME.lower():
                            preferred_order = [m for m in available_short_names if 'flash' in m.lower()]
                        elif 'pro' in MODEL_NAME.lower():
                            preferred_order = [m for m in available_short_names if 'pro' in m.lower()]
                        
                        if not preferred_order:
                            preferred_order = available_short_names
                        
                        for model_short_name in preferred_order:
                            try:
                                model = genai.GenerativeModel(model_short_name)
                                logger.info(f"Using auto-detected model for intent parsing: {model_short_name}")
                                break
                            except Exception as e:
                                logger.debug(f"Auto-detected model {model_short_name} failed: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Could not list models: {e}")
                
                if model is None:
                    raise ValueError(f"Could not find a valid Gemini model. Tried: {model_variants}. Please check your API key and available models.")
                
                # Configure safety settings to allow medical content
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                
                try:
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                except (AttributeError, ValueError):
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: 1,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: 1,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 1,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 0,
                    }
                
                generation_config = {
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
                response = model.generate_content(
                    prompt, 
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                content = response.text.strip()
            else:
                raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
            
            # Parse and validate JSON response
            import json
            import re
            
            # Try to extract JSON from response (handle markdown, extra text)
            content_clean = content.strip()
            if content_clean.startswith('```'):
                # Extract from markdown code block
                match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content_clean, re.DOTALL)
                if match:
                    content_clean = match.group(1)
            elif content_clean.startswith('{'):
                # Extract first JSON object
                match = re.search(r'\{.*\}', content_clean, re.DOTALL)
                if match:
                    content_clean = match.group(0)
            
            # Try to parse JSON
            try:
                result = json.loads(content_clean)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues (trailing commas, etc.)
                content_clean = re.sub(r',\s*}', '}', content_clean)
                content_clean = re.sub(r',\s*]', ']', content_clean)
                try:
                    result = json.loads(content_clean)
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse JSON response after repair attempt")
                    logger.error(f"Raw content (first 500 chars): {content[:500]}")
                    logger.error(f"Cleaned content (first 500 chars): {content_clean[:500]}")
                    logger.error(f"JSON decode error: {e2}")
                    raise ValueError(f"Invalid JSON response from LLM: {e2}")
            
            # Validate and normalize filters
            raw_filters = result.get('filters', {})
            normalized_filters = self._validate_and_normalize_filters(raw_filters, df, query)
            
            # Create UserQuery object
            user_query = UserQuery(
                goal=result.get('goal', 'Extract and filter data based on criteria'),
                filters=normalized_filters,
                outputs=result.get('outputs', ['recordid', 'imaging_date']),
                input_fields=result.get('input_fields', ['narrative']),
                assumptions=result.get('assumptions', [])
            )
            
            logger.info(f"LLM parsing successful: {user_query}")
            return user_query
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._create_empty_user_query(query)
    
    def _validate_and_normalize_filters(self, raw_filters: Dict[str, Any], df: Optional[pd.DataFrame], query: str) -> Dict[str, Any]:
        """
        Validate and normalize filters from LLM output.
        
        Fixes common issues:
        1. Invented filter keys → normalize to allowed schema
        2. Medical concepts → convert to narrative_contains_any
        3. Missing distinct clause for patient counting
        4. Synonym expansion for medical terms
        """
        if not raw_filters:
            return {}
        
        # Whitelist of allowed filter keys
        ALLOWED_STRUCTURED_KEYS = {
            'start_year', 'end_year',
            'narrative_contains_any', 'narrative_contains_all', 'narrative_contains_none',
            'distinct'
        }
        
        # Get actual DataFrame columns if available
        df_columns = set(df.columns) if df is not None else set()
        
        # Add min/max patterns for numeric columns
        numeric_columns = set()
        if df is not None:
            numeric_columns = set(df.select_dtypes(include=['number']).columns)
            for col in numeric_columns:
                ALLOWED_STRUCTURED_KEYS.add(f'min_{col}')
                ALLOWED_STRUCTURED_KEYS.add(f'max_{col}')
        
        # Add direct column filters for existing columns
        ALLOWED_STRUCTURED_KEYS.update(df_columns)
        
        normalized = {}
        medical_concepts = []
        
        # Process each filter key
        for key, value in raw_filters.items():
            if key in ALLOWED_STRUCTURED_KEYS:
                # Valid key, keep as-is
                normalized[key] = value
            elif key.startswith('min_') or key.startswith('max_'):
                # Check if column exists
                col_name = key.replace('min_', '').replace('max_', '')
                if col_name in df_columns and pd.api.types.is_numeric_dtype(df[col_name]):
                    normalized[key] = value
                else:
                    logger.warning(f"Filter key {key} references non-existent or non-numeric column, skipping")
            elif key in df_columns:
                # Direct column filter
                normalized[key] = value
            elif key in ['narrative_contains', 'hydronephrosis', 'has_hydronephrosis', 
                        'stone_presence', 'side', 'size', 'has_stone', 'has_stones']:
                # Known invented keys - normalize to narrative search
                if key == 'narrative_contains':
                    # Convert to narrative_contains_any
                    if isinstance(value, str):
                        medical_concepts.append(value)
                    elif isinstance(value, list):
                        medical_concepts.extend([str(v) for v in value])
                elif isinstance(value, (bool, str, int, float)) and value:
                    # Medical concept key (e.g., "hydronephrosis": true)
                    concept = key.replace('has_', '').replace('_presence', '').replace('_', ' ')
                    medical_concepts.append(concept)
                    logger.info(f"Normalized invented filter key '{key}' = {value} to medical concept: '{concept}'")
                elif value is False or value == 0 or value == '':
                    # Explicitly false/empty - skip this filter
                    logger.info(f"Skipping filter '{key}' with false/empty value: {value}")
                else:
                    # Value might be the concept itself
                    medical_concepts.append(str(value))
                    logger.info(f"Normalized invented filter key '{key}' to narrative search with value: {value}")
            else:
                # Unknown key - treat as potential medical concept
                if isinstance(value, (bool, int, float)) and value:
                    medical_concepts.append(key.replace('_', ' '))
                elif isinstance(value, str):
                    medical_concepts.append(value)
                logger.warning(f"Unknown filter key '{key}' treated as medical concept for narrative search")
        
        # Add medical concepts to narrative_contains_any with synonym expansion
        if medical_concepts:
            expanded_concepts = self._expand_medical_synonyms(medical_concepts)
            if 'narrative_contains_any' in normalized:
                # Merge with existing
                existing = normalized['narrative_contains_any']
                if isinstance(existing, list):
                    expanded_concepts.extend(existing)
                else:
                    expanded_concepts.append(str(existing))
            normalized['narrative_contains_any'] = list(set(expanded_concepts))  # Deduplicate
        
        # Add distinct clause if query asks about "patients" but distinct not specified
        if 'distinct' not in normalized:
            query_lower = query.lower()
            if any(word in query_lower for word in ['patient', 'patients', 'how many', 'count']):
                normalized['distinct'] = 'recordid'
                logger.info("Added 'distinct: recordid' for patient counting")
        
        # Normalize narrative_contains_any to always be a list
        if 'narrative_contains_any' in normalized:
            value = normalized['narrative_contains_any']
            if isinstance(value, str):
                normalized['narrative_contains_any'] = [value]
            elif not isinstance(value, list):
                normalized['narrative_contains_any'] = [str(value)]
            # Remove empty strings from list
            normalized['narrative_contains_any'] = [v for v in normalized['narrative_contains_any'] if v and str(v).strip()]
            # If list is empty after cleaning, remove the key
            if not normalized['narrative_contains_any']:
                del normalized['narrative_contains_any']
                logger.warning("narrative_contains_any became empty after normalization, removing filter")
        
        # Final validation: if no filters remain, try to extract concept from query
        if not normalized:
            logger.warning(f"Filter normalization resulted in empty filters! Raw filters were: {raw_filters}")
            # Try to extract medical concept directly from query as last resort
            query_lower = query.lower()
            # Look for common medical terms in query
            medical_terms = ['hydronephrosis', 'infection', 'stone', 'calculus', 'pain', 'fever']
            for term in medical_terms:
                if term in query_lower:
                    normalized['narrative_contains_any'] = [term]
                    logger.warning(f"Created fallback filter from query: narrative_contains_any=['{term}']")
                    break
        
        logger.info(f"Filter normalization complete. Normalized filters: {normalized}")
        return normalized
    
    def _expand_medical_synonyms(self, concepts: List[str]) -> List[str]:
        """
        Expand medical concepts to include common synonyms.
        
        This handles cases like hydronephrosis → [hydronephrosis, hydroureteronephrosis, pelviectasis, etc.]
        """
        expanded = []
        
        # Medical concept synonym dictionary
        SYNONYMS = {
            'hydronephrosis': [
                'hydronephrosis', 'hydroureteronephrosis', 'pelviectasis', 
                'caliectasis', 'collecting system dilation', 'collecting system dilatation',
                'dilation of the renal pelvis', 'renal pelvis dilation', 'pelvic dilation'
            ],
            'infection': ['infection', 'infectious', 'infected', 'bacteriuria', 'pyuria'],
            'stone': ['stone', 'stones', 'calculus', 'calculi', 'nephrolithiasis', 'urolithiasis'],
            'kidney stone': ['kidney stone', 'kidney stones', 'renal stone', 'renal stones', 'nephrolithiasis'],
        }
        
        for concept in concepts:
            concept_lower = concept.lower().strip()
            
            # Check for exact match in synonyms
            found = False
            for key, synonyms in SYNONYMS.items():
                if concept_lower == key or concept_lower in synonyms:
                    expanded.extend(synonyms)
                    found = True
                    break
            
            # If no synonyms found, add the concept as-is
            if not found:
                expanded.append(concept)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for term in expanded:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                result.append(term)
        
        return result
    
    def _create_empty_user_query(self, query: str) -> UserQuery:
        """Create an empty UserQuery object for fallback scenarios."""
        return UserQuery(
            goal="Extract and filter data based on criteria",
            filters={},
            outputs=['recordid', 'imaging_date'],
            input_fields=['narrative'],
            assumptions=[]
        )
    
    # ==================== PLAN CREATION ====================
    
    def create_plan_summary(self, user_query: UserQuery, 
                          estimated_rows: Optional[int] = None) -> PlanSummary:
        """
        Create a formatted plan summary for user confirmation.
        
        Args:
            user_query: Parsed user query
            estimated_rows: Estimated number of matching rows
            
        Returns:
            PlanSummary object
        """
        processing_time_estimate = "1-2 minutes" if estimated_rows and estimated_rows > 100 else "30-60 seconds"
        
        return PlanSummary(
            goal=user_query.goal,
            input_fields=user_query.input_fields,
            filters=user_query.filters,
            outputs=user_query.outputs,
            assumptions=user_query.assumptions,
            estimated_rows=estimated_rows,
            processing_time_estimate=processing_time_estimate
        )
    
    def format_plan_summary(self, plan: PlanSummary) -> str:
        """
        Format plan summary for display to user.
        
        Args:
            plan: PlanSummary object
            
        Returns:
            Formatted string for display
        """
        lines = [
            "=" * 60,
            "PLAN SUMMARY",
            "=" * 60,
            f"Goal: {plan.goal}",
            "",
            "Input fields detected:",
        ]
        
        for field in plan.input_fields:
            lines.append(f"  • {field}")
        
        lines.append("")
        lines.append("Filters:")
        if plan.filters:
            for key, value in plan.filters.items():
                lines.append(f"  • {key}: {value}")
        else:
            lines.append("  • No specific filters applied")
        
        lines.append("")
        lines.append("Outputs:")
        for output in plan.outputs:
            lines.append(f"  • {output}")
        
        lines.append("")
        lines.append("Assumptions:")
        for assumption in plan.assumptions:
            lines.append(f"  • {assumption}")
        
        if plan.estimated_rows:
            lines.append("")
            lines.append(f"Estimated matching rows: {plan.estimated_rows}")
        
        lines.append("")
        lines.append(f"Estimated processing time: {plan.processing_time_estimate}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
