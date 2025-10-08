#!/usr/bin/env python3
"""
Radiology AI Agent - Main Entry Point

A Python AI agent that uses OpenAI (ChatGPT) API to transform de-identified 
radiology narratives into structured data, with a confirmation step before extraction.

Usage:
    python main.py data.xlsx --query "How many patients had left kidney stones > 1 cm?"
    python main.py data.xlsx --dry-run --limit 10
    python main.py data.xlsx  # Interactive mode
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator import main

if __name__ == "__main__":
    main()



