"""Configuration settings for the radiology AI agent."""

import os
import logging
from dotenv import load_dotenv
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "gemini"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")  # Default to 2.5, will auto-detect if not available

# Common Configuration
MODEL_NAME = os.getenv("MODEL_NAME", OPENAI_MODEL_NAME if LLM_PROVIDER == "openai" else GEMINI_MODEL_NAME)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

# File paths
OUTPUT_DIR = "out"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Validation
if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "dummy_key_for_testing":
        logger.warning("OPENAI_API_KEY not set or using dummy key - LLM features will be disabled")
elif LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "dummy_key_for_testing":
        logger.warning("GEMINI_API_KEY not set or using dummy key - LLM features will be disabled")
else:
    logger.warning(f"Unknown LLM provider: {LLM_PROVIDER}. Supported: 'openai', 'gemini'")

# Model configuration
SUPPORTED_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
# Gemini model names - use the exact names from the API
SUPPORTED_GEMINI_MODELS = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]

# Cost estimation (per 1K tokens, as of 2024)
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-1.5-pro": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-pro": {"input": 0.0, "output": 0.0}  # Free tier
}

def get_llm_client():
    """
    Get LLM client based on configured provider.
    Returns a unified client interface that works with both OpenAI and Gemini.
    """
    if LLM_PROVIDER == "openai":
        return get_openai_client()
    elif LLM_PROVIDER == "gemini":
        return get_gemini_client()
    else:
        logger.error(f"Unsupported LLM provider: {LLM_PROVIDER}")
        return None

def get_openai_client():
    """Get OpenAI client if API key is available."""
    try:
        from openai import OpenAI
        if not OPENAI_API_KEY or OPENAI_API_KEY == "dummy_key_for_testing":
            return None
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None

def get_gemini_client():
    """Get Gemini client if API key is available."""
    try:
        import google.generativeai as genai
        if not GEMINI_API_KEY or GEMINI_API_KEY == "dummy_key_for_testing":
            return None
        genai.configure(api_key=GEMINI_API_KEY)
        return genai
    except ImportError:
        logger.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
        return None
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None

# Backward compatibility - alias for get_llm_client
def get_openai_client():
    """Backward compatibility alias for get_llm_client."""
    return get_llm_client()
