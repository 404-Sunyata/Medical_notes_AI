"""Intent parser for natural language queries."""

import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging

from .llm_schema import UserQuery, PlanSummary
from .config import get_llm_client, LLM_PROVIDER, MODEL_NAME

logger = logging.getLogger(__name__)

class IntentParser:
    """Parse natural language queries into structured filters and goals."""
    
    def __init__(self):
        # Common patterns for different types of queries
        self.patterns = {
            'count_queries': [
                r'how many',
                r'count',
                r'number of',
                r'total'
            ],
            'side_queries': [
                r'\b(left|right|bilateral|both)\b',
                r'\b(left|right)\s+(kidney|renal|side)\b'
            ],
            'size_queries': [
                r'>\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
                r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)\s*(?:or\s+)?(?:larger|greater|bigger|>)',
                r'(?:larger|greater|bigger|>)\s*than\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
                r'(?:size|diameter)\s*(?:of\s+)?(?:at\s+least|>=?)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
                r'<\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
                r'(?:smaller|less)\s*than\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)'
            ],
            'date_queries': [
                r'(?:between|from)\s+(\d{4})\s*(?:and|to|-)\s*(\d{4})',
                r'(?:in|during)\s+(\d{4})',
                r'(?:since|after)\s+(\d{4})',
                r'(?:before|prior\s+to)\s+(\d{4})'
            ],
            'stone_queries': [
                r'\bstone[s]?\b',
                r'\bcalculi\b',
                r'\bnephrolithiasis\b',
                r'\bkidney\s+stone[s]?\b',
                r'\brenal\s+stone[s]?\b'
            ],
            'bladder_queries': [
                r'\bbladder\b',
                r'\bvolume\b',
                r'\bcapacity\b'
            ]
        }
        
        # Dynamic pattern storage
        self.dynamic_patterns_file = "out/dynamic_patterns.json"
        self.dynamic_patterns = self._load_dynamic_patterns()
        
        # Merge dynamic patterns with static patterns
        self.patterns.update(self.dynamic_patterns)
    
    # ==================== STAGE 1: DOMAIN VALIDATION ====================
    
    def parse_query_with_domain_validation(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query with domain validation and fallback options.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with parsing results and domain validation
        """
        logger.info(f"Checking domain relevance and data availability for query: {query}")
        
        # Use combined method for medical relevance and data availability
        result = self._check_medical_relevance_and_data_availability(query)
        
        # If query is medical and data is available, proceed with parsing
        if result.get('is_medical', False) and result.get('data_available', True):
            logger.info(f"Query '{query}' is relevant to medical radiology domain and data is available")
            
            # LLM-FIRST APPROACH: Try LLM parsing first for better accuracy
            logger.info("Attempting LLM-based parsing (primary method)")
            user_query = self._parse_query_with_llm(query)
            
            if self._is_parsing_successful(user_query, query):
                result['user_query'] = user_query
                result['parsing_method'] = 'llm_primary'
                logger.info("Successfully parsed with LLM")
                return result
            else:
                logger.warning("LLM parsing failed, falling back to pattern matching")
            
            # FALLBACK: Try pattern matching if LLM fails
            logger.info("Attempting pattern-based parsing (fallback)")
            user_query = self.parse_query(query)
            
            if self._is_parsing_successful(user_query, query):
                result['user_query'] = user_query
                result['parsing_method'] = 'pattern_fallback'
                logger.info("Successfully parsed with pattern matching")
                return result
            else:
                logger.warning("Pattern matching also failed")
            
            # FINAL FALLBACK: Try learned patterns
            logger.info("Attempting learned pattern parsing (final fallback)")
            user_query = self.parse_query_with_learning(query)
            
            if self._is_parsing_successful(user_query, query):
                result['user_query'] = user_query
                result['parsing_method'] = 'learned_patterns'
                logger.info("Successfully parsed with learned patterns")
                return result
            else:
                logger.warning("All parsing methods failed")
            
            # All methods failed
            logger.error(f"All parsing methods failed for query: {query}")
            result['error'] = "Unable to parse query with available methods. The query may be too complex or contain unrecognized medical concepts."
            result['suggestion'] = "Please try rephrasing your query with simpler medical terms or contact support if the issue persists."
            result['parsing_method'] = 'failed'
        
        return result
    
    def _parse_query_with_llm(self, query: str) -> UserQuery:
        """
        Parse query using LLM for better accuracy and natural language understanding.
        """
        logger.info(f"Using LLM to parse query: {query}")
        
        # Check if we have LLM client available
        client = get_llm_client()
        if not client:
            logger.warning(f"{LLM_PROVIDER.upper()} client not available, falling back to pattern matching")
            return self._create_empty_user_query(query)
        
        try:
            # Create a comprehensive prompt for LLM parsing
            prompt = f"""
            Parse this medical radiology query into structured components:
            
            Query: "{query}"
            
            Extract the following information:
            1. Goal: What is the user trying to accomplish? (e.g., "Count patients", "Find patients", "Calculate statistics")
            2. Filters: What criteria should be applied? Include:
               - stone_presence: "present", "absent", or null
               - side: "left", "right", "bilateral", or null  
               - min_size_cm: minimum stone size in cm, or null
               - max_size_cm: maximum stone size in cm, or null
               - min_bladder_volume_ml: minimum bladder volume in ml, or null
               - start_year: earliest year, or null
               - end_year: latest year, or null
            3. Outputs: What information should be returned? (e.g., ["recordid", "imaging_date", "stone_size"])
            4. Input fields: What data fields are needed? (e.g., ["narrative", "imaging_date"])
            5. Assumptions: What assumptions are being made?
            
            Return ONLY a JSON object with this structure:
            {{
                "goal": "string",
                "filters": {{"key": "value"}},
                "outputs": ["field1", "field2"],
                "input_fields": ["field1", "field2"],
                "assumptions": ["assumption1", "assumption2"]
            }}
            
            Examples:
            - "how many patients have no kidney stone?" → {{"goal": "Count patients", "filters": {{"stone_presence": "absent"}}, "outputs": ["recordid"], "input_fields": ["narrative"], "assumptions": ["Querying for patients without stones"]}}
            - "find patients with left kidney stones > 1cm" → {{"goal": "Find patients", "filters": {{"side": "left", "min_size_cm": 1.0}}, "outputs": ["recordid", "left_stone_size_cm"], "input_fields": ["narrative"], "assumptions": ["Querying for left side stones larger than 1cm"]}}
            """
            
            # Call appropriate API based on provider
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
                
                # Try most permissive settings - BLOCK_NONE for DANGEROUS_CONTENT
                try:
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # Most permissive for medical data
                    }
                except (AttributeError, ValueError):
                    # Fallback: Use numeric values
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: 1,  # BLOCK_ONLY_HIGH
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: 1,  # BLOCK_ONLY_HIGH
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 1,  # BLOCK_ONLY_HIGH
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 0,  # BLOCK_NONE
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
            
            # Parse the JSON response
            import json
            result = json.loads(content)
            
            # Create UserQuery object
            user_query = UserQuery(
                goal=result.get('goal', 'Extract and filter data based on criteria'),
                filters=result.get('filters', {}),
                outputs=result.get('outputs', ['recordid', 'imaging_date']),
                input_fields=result.get('input_fields', ['narrative']),
                assumptions=result.get('assumptions', [])
            )
            
            logger.info(f"LLM parsing successful: {user_query}")
            return user_query
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._create_empty_user_query(query)
    
    def _create_empty_user_query(self, query: str) -> UserQuery:
        """Create an empty UserQuery object for fallback scenarios."""
        return UserQuery(
            goal="Extract and filter data based on criteria",
            filters={},
            outputs=['recordid', 'imaging_date'],
            input_fields=['narrative'],
            assumptions=[]
        )
    
    def _is_parsing_successful(self, user_query, original_query: str) -> bool:
        """
        Check if the parsing was successful by examining the results.
        
        Args:
            user_query: Parsed UserQuery object
            original_query: Original query string
            
        Returns:
            True if parsing was successful, False otherwise
        """
        # Check if we have meaningful filters (most important indicator)
        if user_query.filters and len(user_query.filters) > 0:
            return True
        
        # Check if the goal extraction was successful and meaningful
        if user_query.goal and user_query.goal != "Extract and filter data based on criteria":
            return True
        
        # Check if we have meaningful outputs (but not just default ones)
        if (user_query.outputs and len(user_query.outputs) > 0 and 
            user_query.outputs != ['recordid', 'imaging_date']):
            return True
        
        # Check if we have meaningful input fields (but not just default ones)
        if (user_query.input_fields and len(user_query.input_fields) > 0 and 
            user_query.input_fields != ['narrative']):
            return True
        
        # If none of the above, parsing was not successful
        return False
    
    def _check_medical_relevance_and_data_availability(self, query: str) -> Dict[str, Any]:
        """Combined check for medical relevance and data availability."""
        query_lower = query.lower()
        
        # Define keywords for medical relevance
        medical_keywords = [
            'patient', 'stone', 'kidney', 'bladder', 'renal', 'imaging', 
            'radiology', 'scan', 'ultrasound', 'ct', 'mri', 'x-ray',
            'hydronephrosis', 'calculi', 'nephrolithiasis', 'volume',
            'size', 'cm', 'mm', 'ml', 'anatomy', 'medical',
            # Date/year keywords - these are available in the dataset
            'date', 'year', 'imaging date', 'imaging year', 'in year', 'from year', 'since year', 'images'
        ]
        
        non_medical_keywords = [
            'weather', 'recipe', 'sports', 'movie', 'music', 'game',
            'cooking', 'travel', 'shopping', 'fashion', 'news', 'stock',
            'bitcoin', 'crypto', 'food', 'restaurant', 'hotel', 'flight'
        ]
        
        # Define keywords for unavailable data fields
        unavailable_fields = {
            'gender': [
                'male patients', 'female patients', 'male patient', 'female patient',
                'men with', 'women with', 'man with', 'woman with',
                'gender', 'sex', 'demographic'
            ],
            'age': [
                # More specific patterns to avoid matching date years
                'years old', 'year old', 'old patients', 'young patients',
                'elderly', 'pediatric', 'adult', 'senior',
                # Only match "age" when it's clearly about patient age, not dates
                'patient age', 'age of patient', 'patient\'s age'
            ],
            'weight': [
                'weight', 'kg', 'pounds', 'lbs', 'heavy', 'light', 'obese', 'overweight'
            ],
            'height': [
                'height', 'tall', 'short', 'inches', 'feet'
            ],
            'blood_pressure': [
                'blood pressure', 'bp', 'hypertension', 'hypotension', 'pressure'
            ],
            'medication': [
                'medication', 'drug', 'medicine', 'prescription', 'treatment', 'therapy'
            ],
            'symptoms': [
                'symptom', 'pain', 'ache', 'hurt', 'complaint'
            ],
            'diagnosis': [
                'diagnosis', 'condition', 'disease', 'illness', 'disorder'
            ],
            'treatment': [
                'surgery', 'operation', 'procedure', 'intervention'
            ]
        }
        
        known_unavailable_medical_fields = {
            'vital_signs': [
                'heart rate', 'pulse', 'temperature', 'fever', 'blood pressure',
                'respiratory rate', 'oxygen saturation', 'spo2', 'vitals'
            ],
            'lab_values': [
                'lab values', 'blood test', 'urine test', 'creatinine', 'bun',
                'glucose', 'hemoglobin', 'white blood cell', 'platelet'
            ],
            'demographics': [
                'race', 'ethnicity', 'insurance', 'zip code', 'address'
            ]
        }
        
        # Step 1: Fast path - Check for clearly medical keywords
        is_medical = any(keyword in query_lower for keyword in medical_keywords)
        
        # Step 2: Fast path - Check for clearly non-medical keywords
        is_non_medical = any(keyword in query_lower for keyword in non_medical_keywords)
        
        # Step 3: Check for unavailable data fields
        # First, check for date/year context - if query mentions dates/years, it's about imaging dates, not age
        has_date_context = any(phrase in query_lower for phrase in [
            'in year', 'from year', 'since year', 'year 20', 'imaging date', 
            'date', 'images in', 'have images', 'imaging year'
        ])
        
        missing_fields = []
        for field_type, keywords in unavailable_fields.items():
            # Special handling for age - don't match if it's clearly a date/year query
            if field_type == 'age' and has_date_context:
                # Skip age check if query is about dates/years
                continue
            if any(keyword in query_lower for keyword in keywords):
                missing_fields.append(field_type)
        
        for field_type, keywords in known_unavailable_medical_fields.items():
            if any(keyword in query_lower for keyword in keywords):
                missing_fields.append(field_type)
        
        # Step 4: Extract potential variable names from the query
        variable_patterns = [
            r'average\s+(\w+)', r'mean\s+(\w+)', r'(\w+)\s+of\s+patients',
            r'(\w+)\s+rate', r'(\w+)\s+level', r'(\w+)\s+count',
            r'(\w+)\s+size', r'(\w+)\s+volume'
        ]
        
        found_variables = []
        for pattern in variable_patterns:
            matches = re.findall(pattern, query_lower)
            found_variables.extend(matches)
        
        # Check for unknown variables
        all_available_keywords = []
        for keywords in medical_keywords:
            all_available_keywords.extend([keywords])
        
        # Add date/year related keywords as available
        date_year_keywords = ['date', 'year', 'imaging', 'time', 'when', 'images', 'image']
        all_available_keywords.extend(date_year_keywords)
        
        all_unavailable_keywords = []
        for keywords in unavailable_fields.values():
            all_unavailable_keywords.extend(keywords)
        for keywords in known_unavailable_medical_fields.values():
            all_unavailable_keywords.extend(keywords)
        
        unknown_variables = []
        for variable in found_variables:
            # Skip date/year related variables - these are available
            if variable in ['date', 'year', 'imaging', 'time', 'when', 'images', 'image']:
                continue
            if (variable not in all_available_keywords and 
                variable not in all_unavailable_keywords and
                variable not in ['patients', 'stone', 'kidney', 'bladder', 'cm', 'ml']):
                unknown_variables.append(variable)
        
        # Step 5: Determine final result
        if is_non_medical:
            logger.info("Fast path: Non-medical keywords detected")
            return {
                'is_medical': False,
                'domain_relevant': False,
                'data_available': True,
                'message': f"The query '{query}' is not related to medical radiology data. This system is designed to work with radiology datasets containing patient records, kidney stones, bladder conditions, and medical imaging findings.",
                'suggestion': "Would you like me to provide a general answer using my knowledge base instead?",
                'general_response': self._get_general_llm_response(query),
                'user_query': None
            }
        
        if is_medical:
            logger.info("Fast path: Medical keywords detected")
            # Check if data is available
            if missing_fields or unknown_variables:
                # Medical query but data not available
                if unknown_variables:
                    message = f"The following variables are not available in this dataset: {', '.join(unknown_variables)}. "
                    if missing_fields:
                        message += f"Additionally, {', '.join(missing_fields)} information is not available. "
                    message += "This dataset only contains radiology imaging data (stone presence, stone size, kidney size, bladder volume, and medical history)."
                elif 'gender' in missing_fields:
                    message = "Gender information is not available in this dataset. The dataset only contains radiology imaging data without demographic information like gender, age, or personal details."
                elif 'age' in missing_fields:
                    message = "Age information is not available in this dataset. The dataset only contains radiology imaging data without demographic information like age, gender, or personal details."
                else:
                    message = f"The following information is not available in this dataset: {', '.join(missing_fields)}. This dataset only contains radiology imaging data."
                
                return {
                    'is_medical': True,
                    'domain_relevant': True,
                    'data_available': False,
                    'message': f"Your query '{query}' cannot be answered because {message}",
                    'missing_fields': missing_fields,
                    'unknown_variables': unknown_variables,
                    'available_fields': ['patient_id', 'date', 'stone_presence', 'stone_size', 'kidney_size', 'bladder_volume', 'history'],
                    'suggestion': f"Please ask about information available in the dataset, such as: stone presence, stone size, kidney size, bladder volume, and medical history.",
                    'user_query': None
                }
            else:
                # Medical query and data is available
                logger.info(f"Query '{query}' is relevant to medical radiology domain and data is available")
                return {
                    'is_medical': True,
                    'domain_relevant': True,
                    'data_available': True,
                    'message': "Query is relevant to medical radiology domain and data is available. Proceeding with medical data analysis.",
                    'user_query': None  # Will be set by parse_query if needed
                }
        
        # Step 6: Ambiguous cases - default to non-medical
        logger.warning("Query is ambiguous - no clear medical or non-medical keywords detected")
        return {
            'is_medical': False,
            'domain_relevant': False,
            'data_available': True,
            'message': f"The query '{query}' is ambiguous and cannot be determined without additional context.",
            'suggestion': "Please rephrase your query with more specific medical terms or ask about non-medical topics.",
            'user_query': None
        }
    
    def _get_general_llm_response(self, query: str) -> str:
        """Get a general LLM response for non-medical queries."""
        try:
            client = get_llm_client()
            if not client:
                return "I'm sorry, but I can only help with medical radiology queries. Please ask about patient records, kidney stones, bladder conditions, or other radiology-related topics."
            
            prompt = f"""
            The user asked: "{query}"
            
            This query is not related to medical radiology data. Please provide a helpful response using your general knowledge.
            Be informative and helpful, but also mention that this information is not from the medical dataset.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to get general LLM response: {e}")
            return "I'm sorry, but I can only help with medical radiology queries. Please ask about patient records, kidney stones, bladder conditions, or other radiology-related topics."
    
    
    # ==================== STAGE 2: CORE QUERY PARSING ====================
    
    def parse_query(self, query: str) -> UserQuery:
        """
        Parse a natural language query into structured components.
        
        Args:
            query: Natural language query string
            
        Returns:
            UserQuery object with parsed components
        """
        query_lower = query.lower().strip()
        
        # Extract goal
        goal = self._extract_goal(query_lower)
        
        # Extract input fields
        input_fields = self._extract_input_fields(query_lower)
        
        # Extract filters
        filters = self._extract_filters(query_lower)
        
        # Extract desired outputs
        outputs = self._extract_outputs(query_lower, goal)
        
        # Generate assumptions
        assumptions = self._generate_assumptions(filters, query_lower)
        
        return UserQuery(
            goal=goal,
            input_fields=input_fields,
            filters=filters,
            outputs=outputs,
            assumptions=assumptions
        )
    
    def _extract_goal(self, query: str) -> str:
        """Extract the main goal from the query using hybrid approach."""
        # First try regex patterns
        goal = self._extract_goal_regex(query)
        if goal != "Extract and filter data based on criteria":
            return goal
        
        # If regex failed, try LLM-based classification
        return self._extract_goal_llm(query)
    
    def _extract_goal_regex(self, query: str) -> str:
        """Extract goal using regex patterns (original method)."""
        if any(pattern in query for pattern in self.patterns['count_queries']):
            return "Count patients/records matching criteria"
        elif 'show' in query or 'display' in query or 'list' in query:
            return "Display matching records in table format"
        elif 'plot' in query or 'chart' in query or 'graph' in query:
            return "Create visualization of data"
        elif 'analyze' in query or 'compare' in query:
            return "Analyze and compare data"
        else:
            return "Extract and filter data based on criteria"
    
    def _extract_goal_llm(self, query: str) -> str:
        """Extract goal using LLM for semantic understanding."""
        try:
            client = get_llm_client()
            if not client:
                logger.warning("OpenAI client not available, falling back to default goal")
                return "Extract and filter data based on criteria"
            
            prompt = f"""
            Analyze this medical query and determine the primary intent. Return only the category name.
            
            Query: "{query}"
            
            Intent categories:
            - COUNT: Asking for numbers, quantities, totals, counts, how many, how much, tally, sum, amount, figure, headcount, census
            - DISPLAY: Asking to show, display, list, present, view records or data
            - VISUALIZE: Asking for plots, charts, graphs, visualizations, diagrams
            - ANALYZE: Asking for analysis, comparison, evaluation, assessment, study
            - EXTRACT: General data extraction or filtering without specific intent
            
            Return only the category name (COUNT, DISPLAY, VISUALIZE, ANALYZE, or EXTRACT).
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            intent = response.choices[0].message.content.strip().upper()
            logger.info(f"LLM classified query intent as: {intent}")
            
            # Map LLM intent to goal
            goal_mapping = {
                "COUNT": "Count patients/records matching criteria",
                "DISPLAY": "Display matching records in table format", 
                "VISUALIZE": "Create visualization of data",
                "ANALYZE": "Analyze and compare data",
                "EXTRACT": "Extract and filter data based on criteria"
            }
            
            return goal_mapping.get(intent, "Extract and filter data based on criteria")
            
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            return "Extract and filter data based on criteria"
    
    def _extract_input_fields(self, query: str) -> List[str]:
        """Extract which input fields are relevant to the query."""
        fields = []
        
        if any(re.search(pattern, query) for pattern in self.patterns['stone_queries']):
            fields.append('narrative')
        if any(re.search(pattern, query) for pattern in self.patterns['bladder_queries']):
            fields.append('narrative')
        if 'date' in query or 'year' in query or 'time' in query:
            fields.append('imaging_date')
        if 'surgery' in query or 'surg' in query:
            fields.append('surg_date')
        
        # Default to narrative if no specific fields detected
        if not fields:
            fields = ['narrative']
        
        return fields
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from the query."""
        filters = {}
        
        # Extract side filters
        side_filter = self._extract_side_filter(query)
        if side_filter:
            filters['side'] = side_filter
        
        # Extract size filters
        size_filter = self._extract_size_filter(query)
        if size_filter:
            filters.update(size_filter)
        
        # Extract date filters
        date_filter = self._extract_date_filter(query)
        if date_filter:
            filters.update(date_filter)
        
        # Extract stone presence filters
        stone_filter = self._extract_stone_filter(query)
        if stone_filter:
            filters.update(stone_filter)
        
        # Extract bladder filters
        bladder_filter = self._extract_bladder_filter(query)
        if bladder_filter:
            filters.update(bladder_filter)
        
        return filters
    
    def _extract_side_filter(self, query: str) -> Optional[str]:
        """Extract side filter (left, right, bilateral)."""
        if 'left' in query and 'right' not in query:
            return 'left'
        elif 'right' in query and 'left' not in query:
            return 'right'
        elif 'bilateral' in query or 'both' in query:
            return 'bilateral'
        return None
    
    def _extract_size_filter(self, query: str) -> Dict[str, Any]:
        """Extract size-related filters."""
        filters = {}
        
        for pattern in self.patterns['size_queries']:
            matches = re.findall(pattern, query)
            if matches:
                size_value = float(matches[0])
                
                # Convert mm to cm if needed
                if 'mm' in query:
                    size_value = size_value / 10
                
                if 'larger' in query or 'greater' in query or 'bigger' in query or '>' in query:
                    filters['min_size_cm'] = size_value
                elif 'smaller' in query or 'less' in query or '<' in query:
                    filters['max_size_cm'] = size_value
                else:
                    # Default to minimum size
                    filters['min_size_cm'] = size_value
                
                break
        
        return filters
    
    def _extract_date_filter(self, query: str) -> Dict[str, Any]:
        """Extract date range filters."""
        filters = {}
        
        for pattern in self.patterns['date_queries']:
            matches = re.findall(pattern, query)
            if matches:
                if len(matches[0]) == 2:  # Between X and Y
                    start_year, end_year = matches[0]
                    filters['start_year'] = int(start_year)
                    filters['end_year'] = int(end_year)
                else:  # Single year
                    year = int(matches[0])
                    if 'since' in query or 'after' in query:
                        filters['start_year'] = year
                    elif 'before' in query or 'prior' in query:
                        filters['end_year'] = year - 1
                    else:
                        filters['start_year'] = year
                        filters['end_year'] = year
                break
        
        return filters
    
    def _extract_stone_filter(self, query: str) -> Dict[str, Any]:
        """Extract stone-related filters."""
        filters = {}
        
        if any(re.search(pattern, query) for pattern in self.patterns['stone_queries']):
            if 'no' in query or 'without' in query or 'absent' in query:
                filters['stone_presence'] = 'absent'
            elif 'present' in query or 'with' in query:
                filters['stone_presence'] = 'present'
            else:
                filters['stone_presence'] = 'present'  # Default assumption
        
        return filters
    
    def _extract_bladder_filter(self, query: str) -> Dict[str, Any]:
        """Extract bladder-related filters."""
        filters = {}
        
        if any(re.search(pattern, query) for pattern in self.patterns['bladder_queries']):
            # Look for volume specifications - try multiple patterns
            volume_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter[s]?)',  # With units
                r'(\d+(?:\.\d+)?)\s*(?=\s|$)',  # Without units (standalone numbers)
                r'>\s*(\d+(?:\.\d+)?)',  # After > symbol
                r'<\s*(\d+(?:\.\d+)?)',  # After < symbol
                r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter[s]?)?'  # Optional units
            ]
            
            volume_matches = []
            for pattern in volume_patterns:
                matches = re.findall(pattern, query)
                if matches:
                    volume_matches.extend(matches)
                    break  # Use first pattern that matches
            
            if volume_matches:
                volume = float(volume_matches[0])
                if 'larger' in query or 'greater' in query or '>' in query:
                    filters['min_bladder_volume_ml'] = volume
                elif 'smaller' in query or 'less' in query or '<' in query:
                    filters['max_bladder_volume_ml'] = volume
                else:
                    filters['min_bladder_volume_ml'] = volume
        
        return filters
    
    def _extract_outputs(self, query: str, goal: str) -> List[str]:
        """Extract desired output columns."""
        outputs = ['recordid', 'imaging_date']  # Always include these
        
        # Don't return early for count queries - we still need to compute statistics
        
        # Add relevant columns based on query content
        if 'stone' in query:
            outputs.extend(['right_stone', 'left_stone'])
            if 'size' in query:
                outputs.extend(['right_stone_size_cm', 'left_stone_size_cm'])
        
        if 'kidney' in query and 'size' in query:
            outputs.extend(['right_kidney_size_cm', 'left_kidney_size_cm'])
        
        if 'bladder' in query or 'volume' in query:
            outputs.append('bladder_volume_ml')
        
        if 'history' in query:
            outputs.append('history_summary')
        
        return outputs
    
    def _generate_assumptions(self, filters: Dict[str, Any], query: str) -> List[str]:
        """Generate assumptions made during parsing."""
        assumptions = []
        
        # Negation handling
        if 'no' in query or 'without' in query:
            assumptions.append("Negation terms detected - will look for absence of findings")
        else:
            assumptions.append("Will look for presence of findings (default assumption)")
        
        # Size assumptions
        if 'min_size_cm' in filters:
            assumptions.append(f"Minimum stone size threshold: {filters['min_size_cm']} cm")
        if 'max_size_cm' in filters:
            assumptions.append(f"Maximum stone size threshold: {filters['max_size_cm']} cm")
        
        # Date assumptions
        if 'start_year' in filters or 'end_year' in filters:
            assumptions.append("Date filtering will be applied to imaging_date")
        
        # Side assumptions
        if 'side' in filters:
            if filters['side'] == 'bilateral':
                assumptions.append("Will include both left and right sides")
            else:
                assumptions.append(f"Will focus on {filters['side']} side only")
        
        # Missing data assumptions
        assumptions.append("Missing or unclear data will be included as 'unclear' or null")
        assumptions.append("Size measurements will be converted to cm for consistency")
        
        return assumptions
    
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
    
    def _load_dynamic_patterns(self) -> Dict[str, List[str]]:
        """Load dynamically generated patterns from file."""
        try:
            if os.path.exists(self.dynamic_patterns_file):
                with open(self.dynamic_patterns_file, 'r') as f:
                    patterns = json.load(f)
                    logger.info(f"Loaded {len(patterns)} dynamic pattern categories")
                    return patterns
        except Exception as e:
            logger.error(f"Failed to load dynamic patterns: {e}")
        return {}
    
    def _save_dynamic_patterns(self):
        """Save dynamic patterns to file."""
        try:
            os.makedirs(os.path.dirname(self.dynamic_patterns_file), exist_ok=True)
            with open(self.dynamic_patterns_file, 'w') as f:
                json.dump(self.dynamic_patterns, f, indent=2)
            logger.info(f"Saved {len(self.dynamic_patterns)} dynamic pattern categories")
        except Exception as e:
            logger.error(f"Failed to save dynamic patterns: {e}")
    
    def _detect_new_categories(self, query: str) -> List[str]:
        """Detect new medical categories in the query using LLM."""
        try:
            client = get_llm_client()
            if not client:
                return []
            
            prompt = f"""
            Analyze this medical query and identify any medical categories that are NOT in the existing list.
            
            Query: "{query}"
            
            Existing categories: count_queries, side_queries, size_queries, date_queries, stone_queries, bladder_queries
            
            Look for new medical concepts like:
            - Anatomical locations (kidney, ureter, prostate, etc.)
            - Medical conditions (hydronephrosis, obstruction, infection, etc.)
            - Symptoms (pain, nausea, fever, etc.)
            - Treatments (surgery, medication, therapy, etc.)
            - Severity levels (mild, moderate, severe, etc.)
            - Other medical concepts
            
            Return a JSON list of new category names (e.g., ["hydronephrosis_queries", "pain_queries"]).
            If no new categories are found, return an empty list [].
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"LLM detected new categories: {result}")
            
            # Parse JSON response
            try:
                new_categories = json.loads(result)
                return new_categories if isinstance(new_categories, list) else []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {result}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to detect new categories: {e}")
            return []
    
    def _generate_patterns_for_category(self, category: str, query: str) -> List[str]:
        """Generate regex patterns for a new category using LLM."""
        try:
            client = get_llm_client()
            if not client:
                return []
            
            prompt = f"""
            Generate regex patterns for the medical category: {category}
            
            Based on this query: "{query}"
            
            Create regex patterns that would match similar medical terms and phrases.
            Consider:
            - Medical terminology variations
            - Common abbreviations
            - Related terms
            - Word boundaries (use \\b)
            - Case variations
            
            Return a JSON list of regex patterns.
            IMPORTANT: Use single backslashes in the patterns, not double backslashes.
            Example: ["\\bhydronephrosis\\b", "\\bkidney\\s+swelling\\b"]
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Generated patterns for {category}: {result}")
            
            # Parse JSON response
            try:
                # Clean up the response to handle escaped backslashes
                cleaned_result = result.replace('\\', '\\\\')
                patterns = json.loads(cleaned_result)
                return patterns if isinstance(patterns, list) else []
            except json.JSONDecodeError:
                # Try parsing without cleaning first
                try:
                    patterns = json.loads(result)
                    return patterns if isinstance(patterns, list) else []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse generated patterns as JSON: {result}")
                    return []
                
        except Exception as e:
            logger.error(f"Failed to generate patterns for {category}: {e}")
            return []
    
    def _learn_from_query(self, query: str):
        """Learn new patterns from a query and update the parser."""
        # Detect new categories
        new_categories = self._detect_new_categories(query)
        
        if not new_categories:
            return
        
        logger.info(f"Learning new categories: {new_categories}")
        
        # Generate patterns for each new category
        for category in new_categories:
            if category not in self.dynamic_patterns:
                patterns = self._generate_patterns_for_category(category, query)
                if patterns:
                    self.dynamic_patterns[category] = patterns
                    self.patterns[category] = patterns
                    logger.info(f"Added new category '{category}' with {len(patterns)} patterns")
        
        # Save updated patterns
        if new_categories:
            self._save_dynamic_patterns()
    
    def parse_query_with_learning(self, query: str) -> UserQuery:
        """
        Parse a natural language query using learned patterns from file.
        
        Args:
            query: Natural language query string
            
        Returns:
            UserQuery object with parsed components
        """
        # Parse using learned patterns (no additional learning)
        return self.parse_query(query)
    
    # ==================== STAGE 3: DYNAMIC LEARNING ====================
    
