"""Rule-based extraction for radiology narratives as baseline."""

import re
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Regex patterns for extraction
PATTERNS = {
    'stone_presence': {
        'right': [
            r'\bright\s+kidney\s+stone[s]?\b',
            r'\bright\s+renal\s+stone[s]?\b',
            r'\bright\s+ureteral\s+stone[s]?\b',
            r'\bstone[s]?\s+in\s+right\s+kidney\b',
            r'\bstone[s]?\s+in\s+right\s+ureter\b',
            r'\bright\s+side\s+stone[s]?\b'
        ],
        'left': [
            r'\bleft\s+kidney\s+stone[s]?\b',
            r'\bleft\s+renal\s+stone[s]?\b',
            r'\bleft\s+ureteral\s+stone[s]?\b',
            r'\bstone[s]?\s+in\s+left\s+kidney\b',
            r'\bstone[s]?\s+in\s+left\s+ureter\b',
            r'\bleft\s+side\s+stone[s]?\b'
        ]
    },
    'stone_size': [
        r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
        r'(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
        r'(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)'
    ],
    'kidney_size': [
        r'(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?)',
        r'(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?)'
    ],
    'bladder_volume': [
        r'bladder\s+volume[:\s]*(\d+(?:\.\d+)?)\s*(?:ml|milliliter[s]?)',
        r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter[s]?)\s+bladder\s+volume'
    ],
    'negation': [
        r'\bno\s+',
        r'\bnot\s+',
        r'\bnegative\s+',
        r'\babsent\s+',
        r'\bwithout\s+',
        r'\bdenies\s+',
        r'\bunremarkable\s+',
        r'\bnormal\s+'
    ]
}

def extract_stone_presence(text: str, side: str) -> str:
    """
    Extract stone presence for a specific side using regex.
    
    Args:
        text: Radiology narrative text
        side: 'right' or 'left'
        
    Returns:
        'present', 'absent', or 'unclear'
    """
    if side not in ['right', 'left']:
        raise ValueError("Side must be 'right' or 'left'")
    
    text_lower = text.lower()
    
    # Check for negation patterns near stone mentions
    stone_patterns = PATTERNS['stone_presence'][side]
    
    for pattern in stone_patterns:
        matches = list(re.finditer(pattern, text_lower))
        for match in matches:
            # Check for negation in surrounding context (50 chars before/after)
            start = max(0, match.start() - 50)
            end = min(len(text_lower), match.end() + 50)
            context = text_lower[start:end]
            
            # Check for negation patterns
            for neg_pattern in PATTERNS['negation']:
                if re.search(neg_pattern, context):
                    return 'absent'
            
            return 'present'
    
    # If no stone patterns found, check for explicit absence
    absence_patterns = [
        f'no {side} kidney stone',
        f'no {side} renal stone',
        f'{side} kidney normal',
        f'{side} kidney unremarkable'
    ]
    
    for pattern in absence_patterns:
        if re.search(pattern, text_lower):
            return 'absent'
    
    return 'unclear'

def extract_stone_size(text: str, side: str) -> Optional[float]:
    """
    Extract stone size for a specific side.
    
    Args:
        text: Radiology narrative text
        side: 'right' or 'left'
        
    Returns:
        Size in cm, or None if not found
    """
    if side not in ['right', 'left']:
        raise ValueError("Side must be 'right' or 'left'")
    
    text_lower = text.lower()
    
    # Look for size patterns near side-specific stone mentions
    stone_patterns = PATTERNS['stone_presence'][side]
    
    for stone_pattern in stone_patterns:
        stone_matches = list(re.finditer(stone_pattern, text_lower))
        for stone_match in stone_matches:
            # Look for size patterns in surrounding context
            start = max(0, stone_match.start() - 100)
            end = min(len(text_lower), stone_match.end() + 100)
            context = text_lower[start:end]
            
            for size_pattern in PATTERNS['stone_size']:
                size_matches = re.findall(size_pattern, context)
                if size_matches:
                    # Take the first size found
                    size_str = size_matches[0]
                    if isinstance(size_str, tuple):
                        # Multi-dimensional size, take the largest dimension
                        sizes = [float(s) for s in size_str if s]
                        size = max(sizes)
                    else:
                        size = float(size_str)
                    
                    # Convert mm to cm if needed
                    if 'mm' in context[context.find(size_str):context.find(size_str)+20]:
                        size = size / 10
                    
                    return size
    
    return None

def extract_kidney_size(text: str, side: str) -> Optional[str]:
    """
    Extract kidney size for a specific side.
    
    Args:
        text: Radiology narrative text
        side: 'right' or 'left'
        
    Returns:
        Size string in format "L x W x AP cm", or None if not found
    """
    if side not in ['right', 'left']:
        raise ValueError("Side must be 'right' or 'left'")
    
    text_lower = text.lower()
    
    # Look for kidney size patterns near side-specific mentions
    side_patterns = [
        f'{side} kidney',
        f'{side} renal'
    ]
    
    for side_pattern in side_patterns:
        side_matches = list(re.finditer(side_pattern, text_lower))
        for side_match in side_matches:
            # Look for size patterns in surrounding context
            start = max(0, side_match.start() - 100)
            end = min(len(text_lower), side_match.end() + 100)
            context = text_lower[start:end]
            
            for size_pattern in PATTERNS['kidney_size']:
                size_matches = re.findall(size_pattern, context)
                if size_matches:
                    # Format the size string
                    size_tuple = size_matches[0]
                    if len(size_tuple) == 3:
                        return f"{size_tuple[0]} x {size_tuple[1]} x {size_tuple[2]} cm"
                    elif len(size_tuple) == 2:
                        return f"{size_tuple[0]} x {size_tuple[1]} cm"
    
    return None

def extract_bladder_volume(text: str) -> Optional[float]:
    """
    Extract bladder volume from text.
    
    Args:
        text: Radiology narrative text
        
    Returns:
        Volume in ml, or None if not found
    """
    text_lower = text.lower()
    
    for pattern in PATTERNS['bladder_volume']:
        matches = re.findall(pattern, text_lower)
        if matches:
            volume = float(matches[0])
            return volume
    
    return None

def extract_history_summary(text: str) -> Optional[str]:
    """
    Extract history-related information from text.
    
    Args:
        text: Radiology narrative text
        
    Returns:
        History summary string, or None if not found
    """
    text_lower = text.lower()
    
    # Look for history-related keywords
    history_keywords = [
        'history of',
        'previous',
        'prior',
        'past medical history',
        'pmh',
        'h/o'
    ]
    
    history_sentences = []
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        for keyword in history_keywords:
            if keyword in sentence_lower:
                history_sentences.append(sentence.strip())
                break
    
    if history_sentences:
        return '; '.join(history_sentences[:3])  # Limit to first 3 sentences
    
    return None

def extract_regex_baseline(narrative: str) -> Dict[str, Any]:
    """
    Extract all information using regex baseline.
    
    Args:
        narrative: Radiology narrative text
        
    Returns:
        Dictionary with extracted information
    """
    result = {
        'right': {
            'stone_status': extract_stone_presence(narrative, 'right'),
            'stone_size_cm': extract_stone_size(narrative, 'right'),
            'kidney_size_cm': extract_kidney_size(narrative, 'right')
        },
        'left': {
            'stone_status': extract_stone_presence(narrative, 'left'),
            'stone_size_cm': extract_stone_size(narrative, 'left'),
            'kidney_size_cm': extract_kidney_size(narrative, 'left')
        },
        'bladder': {
            'volume_ml': extract_bladder_volume(narrative),
            'wall': None  # Not easily extractable with regex
        },
        'history_summary': extract_history_summary(narrative),
        'key_sentences': None  # Not implemented in regex baseline
    }
    
    return result



