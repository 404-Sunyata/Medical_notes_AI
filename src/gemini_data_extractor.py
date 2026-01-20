"""Gemini-based data extraction module for general-purpose variable extraction.

This module uses Google Gemini to:
1. Parse user intent from a query
2. Identify variables of interest from a dataset
3. Extract and organize those variables into a structured table
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import GEMINI_API_KEY, GEMINI_MODEL_NAME, MAX_RETRIES, TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class GeminiDataExtractor:
    """Gemini-based data extractor for extracting variables of interest from datasets."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Gemini data extractor.
        
        Args:
            model_name: Optional Gemini model name (defaults to GEMINI_MODEL_NAME from config)
        """
        import google.generativeai as genai
        
        if not GEMINI_API_KEY or GEMINI_API_KEY == "dummy_key_for_testing":
            raise ValueError("GEMINI_API_KEY not set. Please set it in your environment or .env file.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = model_name or GEMINI_MODEL_NAME
        self.genai = genai
        
        # Try to initialize the model
        self.model = self._get_model()
        
        if self.model is None:
            logger.warning(f"Could not initialize Gemini model {self.model_name}")
    
    def _get_model(self):
        """Get Gemini model with fallback options."""
        import google.generativeai as genai
        
        model_variants = [
            self.model_name,
            f"{self.model_name}-latest",
            self.model_name.replace("gemini-1.5-flash", "gemini-2.5-flash"),
            self.model_name.replace("gemini-1.5-pro", "gemini-2.5-pro"),
            self.model_name.replace("gemini-1.5-flash", "gemini-1.5-flash-latest"),
            self.model_name.replace("gemini-1.5-pro", "gemini-1.5-pro-latest"),
            self.model_name.replace("gemini-2.5-flash", "gemini-1.5-flash-latest"),
            self.model_name.replace("gemini-2.5-pro", "gemini-1.5-pro-latest"),
        ]
        
        for variant in model_variants:
            try:
                model = genai.GenerativeModel(variant)
                _ = model._model_name
                logger.info(f"Using Gemini model: {variant}")
                return model
            except Exception as e:
                logger.debug(f"Model variant {variant} failed: {e}")
                continue
        
        # Last resort: try to list available models
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            available_short_names = [m.split('/')[-1] for m in available_models]
            
            preferred_order = []
            if 'flash' in self.model_name.lower():
                preferred_order = [m for m in available_short_names if 'flash' in m.lower()]
            elif 'pro' in self.model_name.lower():
                preferred_order = [m for m in available_short_names if 'pro' in m.lower()]
            
            if not preferred_order:
                preferred_order = available_short_names
            
            for model_short_name in preferred_order:
                try:
                    model = genai.GenerativeModel(model_short_name)
                    logger.info(f"Using auto-detected model: {model_short_name}")
                    return model
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
        
        return None
    
    def _create_schema_analysis_prompt(self, df: pd.DataFrame) -> str:
        """Create prompt for analyzing dataset schema."""
        schema_info = []
        schema_info.append("Dataset Schema:")
        schema_info.append(f"- Total rows: {len(df)}")
        schema_info.append(f"- Total columns: {len(df.columns)}")
        schema_info.append("\nColumn Details:")
        
        for col in df.columns:
            dtype = df[col].dtype
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    schema_info.append(
                        f"  - {col}: Numeric ({dtype}) | "
                        f"Range: {non_null_values.min():.2f} to {non_null_values.max():.2f} | "
                        f"Non-null: {non_null_count}/{len(df)}"
                    )
                else:
                    schema_info.append(f"  - {col}: Numeric ({dtype}) | All null")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                schema_info.append(
                    f"  - {col}: DateTime | "
                    f"Range: {df[col].min()} to {df[col].max()} | "
                    f"Non-null: {non_null_count}/{len(df)}"
                )
            else:
                unique_count = df[col].nunique()
                sample_values = df[col].dropna().head(5).tolist()
                schema_info.append(
                    f"  - {col}: Text/Categorical ({dtype}) | "
                    f"Unique values: {unique_count} | "
                    f"Non-null: {non_null_count}/{len(df)} | "
                    f"Sample: {sample_values[:3]}"
                )
        
        return "\n".join(schema_info)
    
    def _detect_narrative_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect narrative/text columns that likely contain unstructured medical notes."""
        narrative_columns = []
        
        # Check for common narrative column names
        common_names = ['narrative', 'notes', 'report', 'text', 'description', 
                       'history_summary', 'clinical_notes', 'findings', 'summary']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name suggests narrative content
            if any(name in col_lower for name in common_names):
                narrative_columns.append(col)
            # Check if it's a text column with long average text length
            elif pd.api.types.is_object_dtype(df[col]):
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 100:  # Likely a narrative column if average length > 100 chars
                    narrative_columns.append(col)
        
        return narrative_columns
    
    def _create_extraction_prompt(self, query: str, df: pd.DataFrame) -> str:
        """Create prompt for intent parsing and variable extraction."""
        schema_info = self._create_schema_analysis_prompt(df)
        
        # Detect narrative columns
        narrative_columns = self._detect_narrative_columns(df)
        narrative_info = ""
        if narrative_columns:
            # Check for negation and bilateral conditions in query
            query_lower = query.lower()
            has_negation = any(neg_word in query_lower for neg_word in ['no', 'not', "don't", "doesn't", "didn't", 'without', 'absence', 'absent'])
            has_both_sides = any(phrase in query_lower for phrase in ['both sides', 'bilateral', 'either side', 'any side'])
            
            negation_note = ""
            if has_negation:
                negation_note = "\n\nIMPORTANT - Query contains negation: The user is asking for patients who DO NOT have the condition. Make sure to extract fields that can identify absence (e.g., both left and right kidney stone status for bilateral queries)."
            
            bilateral_note = ""
            if has_both_sides:
                bilateral_note = "\n\nIMPORTANT - Query mentions 'both sides' or bilateral: Extract status for BOTH left and right sides separately (e.g., 'left_kidney_stone_present' AND 'right_kidney_stone_present'), so we can check that BOTH are absent."
            
            narrative_info = f"""
IMPORTANT - Narrative Text Columns Detected:
The dataset contains the following narrative/text columns that contain unstructured medical notes:
{', '.join(narrative_columns)}

These columns may contain information not present in structured columns. When the user query asks for:
- Medical conditions (e.g., hydronephrosis, infection, pain)
- Findings or observations
- Measurements or values mentioned in text
- Any information that might be in the narrative text
{negation_note}
{bilateral_note}

You should:
1. Include these narrative columns in "variables_of_interest" if they contain relevant information
2. Specify in "extraction_plan" that narrative parsing is needed
3. Add "narrative_extraction" to extraction_plan with:
   - "narrative_columns": [{', '.join([f'"{col}"' for col in narrative_columns])}]
   - "extract_from_narrative": true
   - "extracted_fields": ["field1", "field2"] - list of fields to extract from narrative
   - If query mentions "both sides" or "bilateral", extract BOTH left AND right status fields
   - If query contains negation ("no", "not", "don't"), still extract the status fields so filtering can find absent cases
"""
        
        prompt = f"""You are a data extraction and NLP medical data assistant. Your task is to:
1. Parse the user's query to understand their intent
2. Identify which variables/columns from the dataset are relevant
3. If the query asks for information that might be in narrative/text columns, specify that narrative parsing is needed
4. Extract those variables and organize them into a structured table

Dataset Information:
{schema_info}
{narrative_info}

User Query: "{query}"

Please analyze the query and extract the relevant information. Return a JSON object with this structure:
{{
    "intent": "Description of what the user wants to extract",
    "variables_of_interest": ["column1", "column2", ...],
    "extraction_plan": {{
        "primary_columns": ["column1", "column2"],
        "narrative_extraction": {{
            "narrative_columns": ["narrative", "notes"],
            "extract_from_narrative": true/false,
            "extracted_fields": ["field1", "field2"],
            "field_descriptions": {{
                "field1": "description of what to extract"
            }}
        }} or null,
        "derived_columns": ["new_column_name"],
        "filters": {{
            "column_name": "value or condition"
        }},
        "grouping": ["column_name"] or null,
        "aggregations": {{
            "column_name": "function (count, sum, mean, etc.)"
        }} or null
    }},
    "output_schema": {{
        "column_name": "description and data type"
    }}
}}

Important:
- Only include columns that actually exist in the dataset
- If the query asks for information that might be in narrative text (e.g., medical conditions, findings), set "extract_from_narrative": true
- Specify which fields to extract from narrative in "extracted_fields"
- If aggregation is needed (count, sum, average), specify it
- If grouping is needed, specify the grouping columns
- The output schema should describe what each column in the final table represents
"""
        return prompt
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def parse_intent(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse user intent and identify variables of interest.
        
        Args:
            query: User's query/question
            df: Input DataFrame
            
        Returns:
            Dictionary with intent, variables_of_interest, and extraction_plan
        """
        if self.model is None:
            raise ValueError("Gemini model not initialized")
        
        prompt = self._create_extraction_prompt(query, df)
        
        try:
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
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            content = response.text.strip()
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                content = content.replace(",\n}", "\n}").replace(",\n]", "\n]")
                result = json.loads(content)
            
            logger.info(f"Intent parsed successfully: {result.get('intent', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse intent: {e}")
            raise
    
    def extract_variables(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract variables of interest from the dataset based on user query.
        
        Args:
            query: User's query/question
            df: Input DataFrame
            
        Returns:
            New DataFrame with extracted variables organized into columns
        """
        # Step 1: Parse intent
        intent_result = self.parse_intent(query, df)
        
        variables = intent_result.get("variables_of_interest", [])
        extraction_plan = intent_result.get("extraction_plan", {})
        filters = extraction_plan.get("filters", {})
        grouping = extraction_plan.get("grouping")
        aggregations = extraction_plan.get("aggregations")
        
        # Step 2: Filter data if needed
        filtered_df = df.copy()
        if filters:
            for col, condition in filters.items():
                if col in filtered_df.columns:
                    if isinstance(condition, dict):
                        # Handle range conditions
                        if "min" in condition:
                            filtered_df = filtered_df[filtered_df[col] >= condition["min"]]
                        if "max" in condition:
                            filtered_df = filtered_df[filtered_df[col] <= condition["max"]]
                        if "equals" in condition:
                            filtered_df = filtered_df[filtered_df[col] == condition["equals"]]
                        if "contains" in condition:
                            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(condition["contains"], case=False, na=False)]
                    else:
                        # Simple equality filter
                        filtered_df = filtered_df[filtered_df[col] == condition]
        
        # Step 3: Select variables of interest
        available_vars = [v for v in variables if v in filtered_df.columns]
        if not available_vars:
            logger.warning(f"None of the requested variables {variables} found in dataset. Available columns: {list(df.columns)}")
            # Fallback: return all columns
            available_vars = list(df.columns)
        
        result_df = filtered_df[available_vars].copy()
        
        # Step 4: Apply grouping and aggregation if needed
        if grouping and aggregations:
            grouping_cols = [g for g in grouping if g in result_df.columns]
            if grouping_cols:
                agg_dict = {}
                for col, func in aggregations.items():
                    if col in result_df.columns:
                        if func.lower() == "count":
                            agg_dict[col] = "count"
                        elif func.lower() == "sum":
                            agg_dict[col] = "sum"
                        elif func.lower() == "mean" or func.lower() == "average":
                            agg_dict[col] = "mean"
                        elif func.lower() == "min":
                            agg_dict[col] = "min"
                        elif func.lower() == "max":
                            agg_dict[col] = "max"
                
                if agg_dict:
                    result_df = result_df.groupby(grouping_cols).agg(agg_dict).reset_index()
        
        # Step 5: Extract information from narrative columns if needed
        narrative_extraction = extraction_plan.get("narrative_extraction")
        if narrative_extraction and narrative_extraction.get("extract_from_narrative"):
            result_df = self._extract_from_narrative(
                result_df, 
                filtered_df,  # Use original filtered_df to access narrative columns
                narrative_extraction
            )
        
        # Step 6: Apply filters based on extracted narrative values
        # If the query asks for specific conditions (e.g., "which patients have X"),
        # filter the results based on extracted narrative fields
        result_df = self._apply_narrative_filters(result_df, query, intent_result)
        
        # Step 7: Add derived columns if specified
        derived_columns = extraction_plan.get("derived_columns", [])
        if derived_columns:
            # For now, we'll just log that derived columns were requested
            # In a full implementation, you might use LLM to compute these
            logger.info(f"Derived columns requested: {derived_columns}")
        
        logger.info(f"Extracted {len(result_df)} rows with {len(result_df.columns)} columns")
        return result_df
    
    def _apply_narrative_filters(self, result_df: pd.DataFrame, query: str, 
                                 intent_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters based on extracted narrative values.
        
        If the query asks for specific conditions (e.g., "which patients have X"),
        filter the results to show only matching rows.
        
        Args:
            result_df: DataFrame with extracted narrative fields
            query: Original user query
            intent_result: Parsed intent result
            
        Returns:
            Filtered DataFrame
        """
        query_lower = query.lower()
        
        # Check if query asks for filtering (e.g., "which", "who has", "patients with")
        filter_indicators = ['which', 'who has', 'patients with', 'find patients', 
                            'show patients', 'list patients', 'extract patients', 'have', 
                            'don\'t', "doesn't", "didn't", 'without']
        needs_filtering = any(indicator in query_lower for indicator in filter_indicators)
        
        if not needs_filtering:
            return result_df
        
        # Detect negation in query
        negation_words = ['no', 'not', "don't", "doesn't", "didn't", 'without', 'absence', 'absent', "haven't"]
        has_negation = any(neg_word in query_lower for neg_word in negation_words)
        
        # Detect bilateral/both sides queries
        has_both_sides = any(phrase in query_lower for phrase in ['both sides', 'bilateral', 'either side', 'any side', 'both kidney'])
        
        # Get extracted fields from narrative
        extraction_plan = intent_result.get("extraction_plan", {})
        narrative_extraction = extraction_plan.get("narrative_extraction", {})
        extracted_fields = narrative_extraction.get("extracted_fields", [])
        
        if not extracted_fields:
            return result_df
        
        # Determine what to filter on based on query
        filtered_result = result_df.copy()
        original_count = len(filtered_result)
        
        # Extract key terms from query to match against field names
        query_terms = set(query_lower.split())
        # Add common medical term variations
        medical_terms_map = {
            'stone': ['stone', 'stones', 'calculus', 'calculi'],
            'left': ['left'],
            'right': ['right'],
            'hydronephrosis': ['hydronephrosis', 'hydroureteronephrosis'],
            'kidney': ['kidney', 'renal']
        }
        
        # Find the best matching field to filter on
        best_match_field = None
        best_match_score = 0
        
        for field in extracted_fields:
            if field not in filtered_result.columns:
                continue
                
            field_lower = field.lower().replace('_', ' ')
            field_words = set(field_lower.split())
            
            # Calculate match score
            score = len(query_terms.intersection(field_words))
            
            # Boost score for medical terms
            for term, variations in medical_terms_map.items():
                if term in query_lower and any(var in field_lower for var in variations):
                    score += 2
            
            # Boost score for presence/status fields when query asks "have" or "with"
            if ('have' in query_lower or 'with' in query_lower) and \
               any(word in field_lower for word in ['present', 'status', 'has', 'stone', 'hydronephrosis']):
                score += 1
            
            if score > best_match_score:
                best_match_score = score
                best_match_field = field
        
        # Handle bilateral queries (both sides)
        if has_both_sides and 'stone' in query_lower:
            # Look for both left and right kidney stone fields
            left_field = None
            right_field = None
            for field in extracted_fields:
                field_lower = field.lower()
                if 'left' in field_lower and 'stone' in field_lower and field in filtered_result.columns:
                    left_field = field
                elif 'right' in field_lower and 'stone' in field_lower and field in filtered_result.columns:
                    right_field = field
            
            if left_field and right_field:
                logger.info(f"Bilateral query detected - filtering by both {left_field} and {right_field}")
                
                if has_negation:
                    # Negation + bilateral: both sides must be absent
                    mask = (
                        (filtered_result[left_field].astype(str).str.lower() == 'absent') &
                        (filtered_result[right_field].astype(str).str.lower() == 'absent')
                    )
                    logger.info(f"Negation + bilateral: Filtering for patients where BOTH {left_field}='absent' AND {right_field}='absent'")
                else:
                    # Positive bilateral: either side present
                    mask = (
                        (filtered_result[left_field].astype(str).str.lower() == 'present') |
                        (filtered_result[right_field].astype(str).str.lower() == 'present')
                    )
                    logger.info(f"Bilateral query: Filtering for patients where {left_field}='present' OR {right_field}='present'")
                
                if mask.any():
                    filtered_result = filtered_result[mask]
                    logger.info(f"Filtered bilateral: {original_count} -> {len(filtered_result)} rows")
                else:
                    logger.warning(f"Bilateral filter found no matching rows")
                
                return filtered_result
        
        # Apply filter using the best matching field
        if best_match_field and best_match_field in filtered_result.columns:
            field = best_match_field
            logger.info(f"Applying filter based on extracted field: {field}")
            
            # Determine filter criteria based on field type and query intent
            if filtered_result[field].dtype == 'object':
                if has_negation:
                    # Negation query: filter for "absent"
                    absent_values = ['absent', 'no', 'false', '0', 'negative']
                    mask = filtered_result[field].astype(str).str.lower().isin(absent_values)
                    
                    if mask.any():
                        filtered_result = filtered_result[mask]
                        logger.info(f"Filtered by {field}='absent' (negation query): {original_count} -> {len(filtered_result)} rows")
                    else:
                        logger.warning(f"Negation filter found no rows with {field}='absent'")
                else:
                    # Positive query: filter for "present", "yes", or positive indicators
                    present_values = ['present', 'yes', 'true', '1', 'positive']
                    mask = filtered_result[field].astype(str).str.lower().isin(present_values)
                    
                    if mask.any():
                        filtered_result = filtered_result[mask]
                        logger.info(f"Filtered by {field} in {present_values}: {original_count} -> {len(filtered_result)} rows")
                    else:
                        # Try filtering for non-absent values
                        absent_values = ['absent', 'no', 'false', '0', 'negative', 'null', 'none', '']
                        mask = ~filtered_result[field].astype(str).str.lower().isin(absent_values)
                        mask = mask & filtered_result[field].notna()
                        
                        if mask.any() and mask.sum() < original_count:
                            filtered_result = filtered_result[mask]
                            logger.info(f"Filtered by {field} (non-absent): {original_count} -> {len(filtered_result)} rows")
            else:
                # For numeric fields, filter for non-null values
                mask = filtered_result[field].notna()
                if mask.any() and mask.sum() < original_count:
                    filtered_result = filtered_result[mask]
                    logger.info(f"Filtered by {field} (non-null): {original_count} -> {len(filtered_result)} rows")
        
        # If still no filtering applied, try filtering on any extracted field with "present" values
        if len(filtered_result) == original_count and extracted_fields:
            for field in extracted_fields:
                if field in filtered_result.columns and filtered_result[field].dtype == 'object':
                    # Check if field name suggests it's a status/presence field
                    if any(word in field.lower() for word in ['present', 'status', 'has', 'stone', 'hydronephrosis', 'infection']):
                        present_mask = filtered_result[field].astype(str).str.lower() == 'present'
                        if present_mask.any() and present_mask.sum() < original_count:
                            filtered_result = filtered_result[present_mask]
                            logger.info(f"Filtered by {field}='present' (fallback): {original_count} -> {len(filtered_result)} rows")
                            break
        
        return filtered_result
    
    def _extract_from_narrative(self, result_df: pd.DataFrame, source_df: pd.DataFrame, 
                                narrative_extraction: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract structured information from narrative text columns.
        
        Args:
            result_df: Current result DataFrame
            source_df: Source DataFrame with narrative columns
            narrative_extraction: Extraction plan for narrative parsing
            
        Returns:
            DataFrame with extracted narrative fields added as new columns
        """
        narrative_columns = narrative_extraction.get("narrative_columns", [])
        extracted_fields = narrative_extraction.get("extracted_fields", [])
        field_descriptions = narrative_extraction.get("field_descriptions", {})
        
        if not narrative_columns or not extracted_fields:
            logger.warning("Narrative extraction requested but no columns or fields specified")
            return result_df
        
        # Find narrative columns that exist in source_df
        available_narrative_cols = [col for col in narrative_columns if col in source_df.columns]
        if not available_narrative_cols:
            logger.warning(f"Narrative columns {narrative_columns} not found in dataset")
            return result_df
        
        # Use the first available narrative column (or combine them)
        narrative_col = available_narrative_cols[0]
        logger.info(f"Extracting from narrative column: {narrative_col}")
        logger.info(f"Fields to extract: {extracted_fields}")
        
        # Merge source_df narrative data with result_df
        # Ensure we have the same index/keys to merge properly
        merge_keys = ['recordid'] if 'recordid' in result_df.columns else result_df.columns[:1].tolist()
        
        # Create a mapping to preserve original result_df order
        result_df = result_df.reset_index(drop=True)  # Reset index to ensure alignment
        
        if merge_keys and merge_keys[0] in source_df.columns:
            # Merge and maintain order
            merged_df = result_df.merge(
                source_df[[merge_keys[0], narrative_col]],
                on=merge_keys[0],
                how='left',
                suffixes=('', '_source')
            )
            # If narrative column already exists, use the merged one
            if f'{narrative_col}_source' in merged_df.columns:
                merged_df[narrative_col] = merged_df[f'{narrative_col}_source']
                merged_df = merged_df.drop(columns=[f'{narrative_col}_source'])
        else:
            # If no merge key, just add narrative column by index
            merged_df = result_df.copy()
            if narrative_col in source_df.columns:
                # Align by index position
                merged_df = merged_df.reset_index(drop=True)
                source_df_aligned = source_df.reset_index(drop=True)
                merged_df[narrative_col] = source_df_aligned[narrative_col].values[:len(merged_df)]
        
        # Extract fields from narrative using LLM
        extracted_data = []
        total_rows = len(merged_df)
        logger.info(f"Starting narrative extraction for {total_rows} rows...")
        logger.info(f"result_df length: {len(result_df)}, merged_df length: {len(merged_df)}")
        
        # Iterate through merged_df in order, preserving result_df order
        # Use enumerate to track position, but iterate by integer index to ensure order
        for position in range(total_rows):
            row = merged_df.iloc[position]
            
            # Progress indicator
            if (position + 1) % 10 == 0 or (position + 1) == total_rows:
                logger.info(f"Processing narrative {position + 1}/{total_rows}...")
            
            narrative_text = str(row.get(narrative_col, ''))
            if pd.isna(narrative_text) or narrative_text == '' or narrative_text == 'nan':
                # Create empty dict for this row
                extracted_data.append({field: None for field in extracted_fields})
                logger.debug(f"Row {position}: Empty narrative, skipping extraction")
                continue
            
            # Extract information from this narrative
            try:
                logger.info(f"Row {position}: Extracting from narrative (first 100 chars): {narrative_text[:100]}...")
                extracted = self._parse_single_narrative(
                    narrative_text, 
                    extracted_fields, 
                    field_descriptions
                )
                # Ensure all fields are present
                for field in extracted_fields:
                    if field not in extracted:
                        extracted[field] = None
                extracted_data.append(extracted)
                logger.info(f"Row {position}: Successfully extracted: {extracted}")
            except Exception as e:
                logger.warning(f"Failed to extract from narrative at row {position}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                extracted_data.append({field: None for field in extracted_fields})
        
        # Add extracted fields as new columns
        # The extracted_data list should be in the same order as result_df
        if len(extracted_data) != len(result_df):
            logger.error(f"CRITICAL: Length mismatch! extracted_data={len(extracted_data)}, result_df={len(result_df)}")
            # Pad or truncate to match
            if len(extracted_data) < len(result_df):
                extracted_data.extend([{field: None for field in extracted_fields}] * (len(result_df) - len(extracted_data)))
            else:
                extracted_data = extracted_data[:len(result_df)]
        
        for field in extracted_fields:
            values = [extracted.get(field, None) for extracted in extracted_data]
            result_df[field] = values
            non_null_count = pd.Series(values).notna().sum()
            logger.info(f"Added extracted column: {field} ({non_null_count}/{len(values)} non-null values)")
        
        return result_df
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((Exception,))
    )
    def _parse_single_narrative(self, narrative_text: str, fields_to_extract: List[str],
                                field_descriptions: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse a single narrative text and extract specified fields.
        
        Args:
            narrative_text: The narrative text to parse
            fields_to_extract: List of field names to extract
            field_descriptions: Descriptions of what each field represents
            
        Returns:
            Dictionary with extracted field values
        """
        if self.model is None:
            raise ValueError("Gemini model not initialized")
        
        # Build field descriptions with more context
        fields_desc_detailed = []
        for field in fields_to_extract:
            desc = field_descriptions.get(field, f"Extract {field} from the narrative")
            # Add specific guidance for common medical fields
            field_lower = field.lower()
            if 'left' in field_lower and 'stone' in field_lower:
                desc += ". Look for mentions of 'left kidney stone', 'left stone', 'bilateral stones' with left side, etc. Return 'present' if found, 'absent' if explicitly stated as absent (e.g., 'left kidney clear', 'no left stone', 'no kidney stones', 'both kidneys normal'), or null if not mentioned."
            elif 'right' in field_lower and 'stone' in field_lower:
                desc += ". Look for mentions of 'right kidney stone', 'right stone', 'bilateral stones' with right side, etc. Return 'present' if found, 'absent' if explicitly stated as absent (e.g., 'right kidney clear', 'no right stone', 'no kidney stones', 'both kidneys normal'), or null if not mentioned."
            elif 'hydronephrosis' in field_lower:
                desc += ". Look for mentions of hydronephrosis, hydroureteronephrosis, pelviectasis, or collecting system dilation. Return 'present' if found, 'absent' if explicitly stated as absent, or null if not mentioned."
            fields_desc_detailed.append(f"- {field}: {desc}")
        
        prompt = f"""Extract the following information from this medical narrative text:

Narrative Text:
```text
{narrative_text}
```

Fields to Extract:
{chr(10).join(fields_desc_detailed)}

Return a JSON object with this exact structure:
{{
{chr(10).join([f'    "{field}": "extracted value or null"' for field in fields_to_extract])}
}}

Rules:
- Extract only information that is explicitly mentioned in the narrative
- For presence/status fields (like "has_left_kidney_stone"):
  * Return "present" if the condition is mentioned as present (e.g., "left kidney stone", "left stone present", "bilateral stones: right 1.1 cm, left 1.3 cm")
  * Return "absent" if explicitly stated as absent (e.g., "no left kidney stone", "left kidney clear", "left kidney normal", "no kidney stones seen", "both kidneys normal", "no stones identified")
  * Return null only if not mentioned at all
- IMPORTANT: Phrases like "no kidney stones", "no stones identified", "both kidneys normal" mean BOTH left AND right are absent
- For bilateral stone mentions (e.g., "bilateral stones: right X cm, left Y cm"), both sides should be marked as "present"
- For numeric values, extract the number only (no units unless part of the value)
- Be accurate - carefully read the narrative to determine presence/absence
- For "left kidney stone" queries: check for "left kidney stone", "left stone", "bilateral stones" mentioning left side
"""
        
        try:
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
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            content = response.text.strip()
            
            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Try to fix common JSON issues
                logger.debug(f"JSON decode error, attempting to fix: {json_err}")
                logger.debug(f"Content: {content[:200]}")
                content_fixed = content.replace(",\n}", "\n}").replace(",\n]", "\n]")
                try:
                    result = json.loads(content_fixed)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON even after fixing. Content: {content[:500]}")
                    return {field: None for field in fields_to_extract}
            
            # Ensure all fields are present and convert values appropriately
            for field in fields_to_extract:
                if field not in result:
                    result[field] = None
                else:
                    # Convert string "null" to actual None
                    if isinstance(result[field], str) and result[field].lower() in ['null', 'none', '']:
                        result[field] = None
                    # Normalize boolean-like values
                    elif isinstance(result[field], str):
                        val_lower = result[field].lower()
                        if val_lower in ['true', 'yes', '1']:
                            result[field] = 'present'
                        elif val_lower in ['false', 'no', '0']:
                            result[field] = 'absent'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse narrative: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {field: None for field in fields_to_extract}
    
    def extract_and_organize(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete extraction pipeline: parse intent, extract variables, and organize into table.
        
        Args:
            query: User's query/question
            df: Input DataFrame
            
        Returns:
            Dictionary with:
            - intent: Parsed intent information
            - extracted_data: DataFrame with extracted variables
            - summary: Summary statistics
        """
        try:
            # Parse intent
            intent_result = self.parse_intent(query, df)
            
            # Extract variables
            extracted_df = self.extract_variables(query, df)
            
            # Create summary
            summary = {
                "original_rows": len(df),
                "extracted_rows": len(extracted_df),
                "original_columns": len(df.columns),
                "extracted_columns": len(extracted_df.columns),
                "variables_extracted": list(extracted_df.columns),
                "intent": intent_result.get("intent", "N/A")
            }
            
            return {
                "intent": intent_result,
                "extracted_data": extracted_df,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

