"""OpenAI LLM extractor with JSON schema validation and caching."""

import json
import time
import sqlite3
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

from .config import OPENAI_API_KEY, MODEL_NAME, MAX_RETRIES, TIMEOUT_SECONDS, MODEL_COSTS, CACHE_DIR
from .safety import check_text, create_cache_key, log_usage_stats
from .llm_schema import RadiologyExtraction, validate_extraction_json, create_empty_extraction

logger = logging.getLogger(__name__)

class LLMExtractor:
    """OpenAI LLM extractor with caching and error handling."""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = MODEL_NAME
        self.cache_db = os.path.join(CACHE_DIR, "extraction_cache.db")
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache database."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_cache (
                    cache_key TEXT PRIMARY KEY,
                    narrative_hash TEXT,
                    model_name TEXT,
                    extraction_json TEXT,
                    tokens_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _get_cache_key(self, narrative: str) -> str:
        """Get cache key for narrative."""
        return create_cache_key(narrative, self.model_name)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get extraction result from cache."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT extraction_json, tokens_used FROM extraction_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                result = cursor.fetchone()
                if result:
                    return {
                        'extraction_json': json.loads(result[0]),
                        'tokens_used': json.loads(result[1]) if result[1] else None
                    }
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, narrative: str, 
                      extraction_json: Dict[str, Any], tokens_used: Dict[str, int]):
        """Save extraction result to cache."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO extraction_cache (cache_key, narrative_hash, model_name, extraction_json, tokens_used) VALUES (?, ?, ?, ?, ?)",
                    (cache_key, hash(narrative), self.model_name, 
                     json.dumps(extraction_json), json.dumps(tokens_used))
                )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM."""
        return """You are a clinical NLP assistant specialized in extracting structured information from radiology narratives. 

Your task is to extract information about kidney stones, kidney sizes, and bladder volume from radiology reports.

IMPORTANT RULES:
1. Return ONLY valid JSON that matches the exact schema provided
2. Be conservative - if information is not clearly stated, use null or "unclear"
3. Handle negation carefully - "no stones" means "absent", not "present"
4. For stone sizes, extract only the largest dimension if multiple are given
5. For kidney sizes, preserve the exact format as "L x W x AP cm" or "L x W cm"
6. Do not make assumptions or guesses beyond what is explicitly stated
7. If a finding is not mentioned at all, use "unclear" for status and null for measurements

Focus on accuracy over completeness. It's better to return null than to guess."""
    
    def _create_user_prompt(self, narrative: str) -> str:
        """Create user prompt with narrative."""
        return f"""Extract the following information from this radiology narrative:

```text
{narrative}
```

Return a JSON object with this exact structure:
{{
  "right": {{
    "stone_status": "present"|"absent"|"unclear",
    "stone_size_cm": number|null,
    "kidney_size_cm": "L x W x AP cm"|null
  }},
  "left": {{
    "stone_status": "present"|"absent"|"unclear", 
    "stone_size_cm": number|null,
    "kidney_size_cm": "L x W x AP cm"|null
  }},
  "bladder": {{
    "volume_ml": number|null,
    "wall": "normal"|"abnormal"|null
  }},
  "history_summary": string|null,
  "key_sentences": [string]|null
}}"""
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def _call_openai_api(self, narrative: str) -> Dict[str, Any]:
        """Call OpenAI API with retry logic."""
        start_time = time.time()
        
        try:
            # Safety check
            safety_report = check_text(narrative)
            if not safety_report["safe"]:
                logger.warning(f"Potential PHI detected, using redacted text")
                narrative = safety_report["redacted_text"]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": self._create_user_prompt(narrative)}
                ],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "radiology_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "right": {
                                "type": "object",
                                "properties": {
                                    "stone_status": {"type": "string", "enum": ["present", "absent", "unclear"]},
                                    "stone_size_cm": {"type": ["number", "null"]},
                                    "kidney_size_cm": {"type": ["string", "null"]}
                                },
                                "required": ["stone_status", "stone_size_cm", "kidney_size_cm"]
                            },
                            "left": {
                                "type": "object", 
                                "properties": {
                                    "stone_status": {"type": "string", "enum": ["present", "absent", "unclear"]},
                                    "stone_size_cm": {"type": ["number", "null"]},
                                    "kidney_size_cm": {"type": ["string", "null"]}
                                },
                                "required": ["stone_status", "stone_size_cm", "kidney_size_cm"]
                            },
                            "bladder": {
                                "type": "object",
                                "properties": {
                                    "volume_ml": {"type": ["number", "null"]},
                                    "wall": {"type": ["string", "null"], "enum": ["normal", "abnormal", null]}
                                },
                                "required": ["volume_ml", "wall"]
                            },
                            "history_summary": {"type": ["string", "null"]},
                            "key_sentences": {"type": ["array", "null"], "items": {"type": "string"}}
                        },
                        "required": ["right", "left", "bladder", "history_summary", "key_sentences"]
                    }
                }},
                temperature=0.1,
                timeout=TIMEOUT_SECONDS
            )
            
            # Extract response
            content = response.choices[0].message.content
            extraction_json = json.loads(content)
            
            # Calculate tokens and cost
            tokens_used = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            cost = self._calculate_cost(tokens_used)
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"API call successful: {tokens_used['total_tokens']} tokens, ${cost:.4f}, {processing_time}ms")
            
            return {
                'extraction_json': extraction_json,
                'tokens_used': tokens_used,
                'cost': cost,
                'processing_time_ms': processing_time
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _calculate_cost(self, tokens_used: Dict[str, int]) -> float:
        """Calculate estimated cost for API call."""
        if self.model_name not in MODEL_COSTS:
            return 0.0
        
        costs = MODEL_COSTS[self.model_name]
        prompt_cost = (tokens_used['prompt_tokens'] / 1000) * costs['input']
        completion_cost = (tokens_used['completion_tokens'] / 1000) * costs['output']
        
        return prompt_cost + completion_cost
    
    def extract(self, narrative: str, record_id: str = "unknown") -> RadiologyExtraction:
        """
        Extract structured information from radiology narrative.
        
        Args:
            narrative: Radiology narrative text
            record_id: Patient record ID for logging
            
        Returns:
            RadiologyExtraction object
        """
        if not narrative or not narrative.strip():
            logger.warning(f"Empty narrative for record {record_id}")
            return create_empty_extraction()
        
        cache_key = self._get_cache_key(narrative)
        
        # Try cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for record {record_id}")
            try:
                extraction = validate_extraction_json(cached_result['extraction_json'])
                return extraction
            except Exception as e:
                logger.warning(f"Invalid cached data for record {record_id}: {e}")
                # Fall through to API call
        
        # API call
        try:
            result = self._call_openai_api(narrative)
            extraction_json = result['extraction_json']
            tokens_used = result['tokens_used']
            cost = result['cost']
            
            # Validate JSON
            extraction = validate_extraction_json(extraction_json)
            
            # Save to cache
            self._save_to_cache(cache_key, narrative, extraction_json, tokens_used)
            
            # Log usage
            log_usage_stats(record_id, self.model_name, 
                          tokens_used['prompt_tokens'], 
                          tokens_used['completion_tokens'], 
                          cost)
            
            return extraction
            
        except Exception as e:
            logger.error(f"Extraction failed for record {record_id}: {e}")
            return create_empty_extraction()
    
    def batch_extract(self, narratives: List[str], record_ids: List[str] = None) -> List[RadiologyExtraction]:
        """
        Extract information from multiple narratives.
        
        Args:
            narratives: List of narrative texts
            record_ids: Optional list of record IDs
            
        Returns:
            List of RadiologyExtraction objects
        """
        if record_ids is None:
            record_ids = [f"record_{i}" for i in range(len(narratives))]
        
        if len(narratives) != len(record_ids):
            raise ValueError("narratives and record_ids must have the same length")
        
        results = []
        for narrative, record_id in zip(narratives, record_ids):
            try:
                extraction = self.extract(narrative, record_id)
                results.append(extraction)
            except Exception as e:
                logger.error(f"Batch extraction failed for {record_id}: {e}")
                results.append(create_empty_extraction())
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM extraction_cache")
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT SUM(LENGTH(extraction_json)) FROM extraction_cache")
                total_size = cursor.fetchone()[0] or 0
                
                return {
                    'total_entries': total_entries,
                    'total_size_bytes': total_size,
                    'cache_file': self.cache_db
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}



