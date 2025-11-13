#!/usr/bin/env python3
"""
Quick script to list available Gemini models and test the API.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set in .env file")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    print("=" * 60)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 60)
    
    models = genai.list_models()
    available_models = []
    
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name
            display_name = model.display_name
            available_models.append(model_name)
            print(f"✅ {model_name}")
            print(f"   Display Name: {display_name}")
            print(f"   Supported Methods: {model.supported_generation_methods}")
            print()
    
    print("=" * 60)
    print("RECOMMENDED MODEL NAMES FOR .env FILE")
    print("=" * 60)
    
    # Extract short names
    short_names = []
    for model_name in available_models:
        # Extract the last part after the last '/'
        short_name = model_name.split('/')[-1]
        short_names.append(short_name)
        print(f"GEMINI_MODEL_NAME={short_name}")
    
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    # Try to use the first available model
    if available_models:
        test_model_name = available_models[0].split('/')[-1]
        print(f"Testing with model: {test_model_name}")
        
        try:
            model = genai.GenerativeModel(test_model_name)
            response = model.generate_content(
                "Say 'Hello, Gemini is working!' in JSON format: {\"message\": \"your message here\"}",
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
            )
            print(f"✅ Model test successful!")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
    else:
        print("❌ No available models found")
        
except ImportError:
    print("❌ google-generativeai package not installed")
    print("Install with: pip install google-generativeai")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


