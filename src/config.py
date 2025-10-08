"""Configuration settings for the radiology AI agent."""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
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
if not OPENAI_API_KEY or OPENAI_API_KEY == "dummy_key_for_testing":
    logger.warning("OPENAI_API_KEY not set or using dummy key - LLM features will be disabled")

# Model configuration
SUPPORTED_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
if MODEL_NAME not in SUPPORTED_MODELS:
    raise ValueError(f"Unsupported model: {MODEL_NAME}. Supported models: {SUPPORTED_MODELS}")

# Cost estimation (per 1K tokens, as of 2024)
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03}
}
