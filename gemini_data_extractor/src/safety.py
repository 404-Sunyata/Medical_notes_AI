"""Safety utilities for handling de-identified radiology data."""

import re
import hashlib
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Patterns for potential PHI that might have been missed
PHI_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'date_of_birth': r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b',
    'mrn': r'\b(?:MRN|Medical Record Number)[\s:]*\d+\b',
    'patient_id': r'\b(?:Patient ID|Pt ID)[\s:]*[A-Za-z0-9]+\b'
}

def check_text(text: str) -> Dict[str, Any]:
    """
    Check text for potential PHI and return safety report.
    
    Args:
        text: Input text to check
        
    Returns:
        Dict with safety report including redacted text and warnings
    """
    if not isinstance(text, str):
        return {"safe": True, "redacted_text": str(text), "warnings": []}
    
    redacted_text = text
    warnings = []
    
    for pattern_name, pattern in PHI_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            warnings.append(f"Potential {pattern_name} detected: {matches}")
            # Redact the matches
            redacted_text = re.sub(pattern, f"[REDACTED_{pattern_name.upper()}]", 
                                 redacted_text, flags=re.IGNORECASE)
    
    # Additional checks for common PHI indicators
    phi_indicators = [
        r'\b(?:patient|pt)\s+name\b',
        r'\b(?:doctor|dr|physician)\s+name\b',
        r'\baddress\b',
        r'\bphone\b',
        r'\bemail\b'
    ]
    
    for indicator in phi_indicators:
        if re.search(indicator, text, re.IGNORECASE):
            warnings.append(f"PHI indicator found: {indicator}")
    
    is_safe = len(warnings) == 0
    
    if not is_safe:
        logger.warning(f"Potential PHI detected in text. Warnings: {warnings}")
    
    return {
        "safe": is_safe,
        "redacted_text": redacted_text,
        "warnings": warnings,
        "original_length": len(text),
        "redacted_length": len(redacted_text)
    }

def validate_narrative(narrative: str) -> bool:
    """
    Validate that a narrative is safe for API processing.
    
    Args:
        narrative: Radiology narrative text
        
    Returns:
        True if safe, False otherwise
    """
    safety_report = check_text(narrative)
    return safety_report["safe"]

def create_cache_key(narrative: str, model_name: str) -> str:
    """
    Create a cache key for a narrative and model combination.
    
    Args:
        narrative: Input narrative text
        model_name: Model name used for processing
        
    Returns:
        SHA-256 hash as cache key
    """
    # Use redacted text for cache key to ensure consistency
    safety_report = check_text(narrative)
    cache_input = f"{safety_report['redacted_text']}:{model_name}"
    return hashlib.sha256(cache_input.encode()).hexdigest()

def sanitize_for_logging(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for safe logging.
    
    Args:
        text: Input text
        max_length: Maximum length for logged text
        
    Returns:
        Sanitized text safe for logging
    """
    if not isinstance(text, str):
        return str(text)
    
    # Check for PHI first
    safety_report = check_text(text)
    sanitized = safety_report["redacted_text"]
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized

def log_usage_stats(record_id: str, model_name: str, prompt_tokens: int, 
                   completion_tokens: int, cost: float):
    """
    Log usage statistics for monitoring and cost tracking.
    
    Args:
        record_id: Patient record ID
        model_name: Model used
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        cost: Estimated cost in USD
    """
    logger.info(f"Usage - Record: {record_id}, Model: {model_name}, "
               f"Tokens: {prompt_tokens}+{completion_tokens}, Cost: ${cost:.4f}")



