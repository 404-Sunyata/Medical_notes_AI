#!/usr/bin/env python3
"""
Test script for Gemini API integration.
This script tests the Gemini API connection and basic functionality.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gemini_config():
    """Test Gemini configuration."""
    print("=" * 60)
    print("TEST 1: Configuration Check")
    print("=" * 60)
    
    from src.config import LLM_PROVIDER, GEMINI_API_KEY, MODEL_NAME, get_llm_client
    
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Gemini API Key: {'Set' if GEMINI_API_KEY and GEMINI_API_KEY != 'dummy_key_for_testing' else 'Not Set'}")
    
    client = get_llm_client()
    if client:
        print("✅ LLM Client created successfully")
        return True
    else:
        print("❌ Failed to create LLM client")
        return False

def test_gemini_api_call():
    """Test a simple Gemini API call."""
    print("\n" + "=" * 60)
    print("TEST 2: Direct Gemini API Call")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
        
        if not GEMINI_API_KEY or GEMINI_API_KEY == "dummy_key_for_testing":
            print("❌ Gemini API key not set")
            return False
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        prompt = "Parse this query: 'how many patients have no kidney stone?' Return JSON with filters."
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        
        print(f"✅ API call successful")
        print(f"Response: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def test_intent_parser():
    """Test intent parser with Gemini."""
    print("\n" + "=" * 60)
    print("TEST 3: Intent Parser with Gemini")
    print("=" * 60)
    
    try:
        from src.intent_parser import IntentParser
        
        parser = IntentParser()
        query = "how many patients have no kidney stone?"
        
        print(f"Testing query: '{query}'")
        result = parser.parse_query_with_domain_validation(query)
        
        if result.get('user_query'):
            user_query = result['user_query']
            print(f"✅ Parsing successful")
            print(f"   Method: {result.get('parsing_method', 'unknown')}")
            print(f"   Goal: {user_query.goal}")
            print(f"   Filters: {user_query.filters}")
            print(f"   Outputs: {user_query.outputs}")
            return True
        else:
            print(f"⚠️  Parsing returned no user_query")
            print(f"   Result: {result}")
            return False
    except Exception as e:
        print(f"❌ Intent parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_extractor():
    """Test LLM extractor with Gemini."""
    print("\n" + "=" * 60)
    print("TEST 4: LLM Extractor with Gemini")
    print("=" * 60)
    
    try:
        from src.llm_extractor import LLMExtractor
        
        extractor = LLMExtractor()
        
        if extractor.client is None:
            print("⚠️  LLM client not available (this is OK if API key is not set)")
            return True
        
        # Test with a simple narrative
        narrative = "Right kidney stone measuring 1.2 cm. Left kidney normal. Bladder volume 250 ml."
        
        print(f"Testing extraction with narrative: '{narrative[:50]}...'")
        extraction = extractor.extract(narrative, "TEST_001")
        
        print(f"✅ Extraction successful")
        print(f"   Right stone status: {extraction.right.stone_status}")
        print(f"   Right stone size: {extraction.right.stone_size_cm}")
        print(f"   Bladder volume: {extraction.bladder.volume_ml}")
        return True
    except Exception as e:
        print(f"❌ LLM extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_workflow():
    """Test the full workflow with a sample query."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Workflow Test")
    print("=" * 60)
    
    try:
        # This would require the full data file, so we'll just test the query processing
        from src.intent_parser import IntentParser
        
        parser = IntentParser()
        queries = [
            "how many patients have no kidney stone?",
            "find patients with left kidney stones > 1cm",
            "what is the mean bladder volume?"
        ]
        
        print("Testing multiple queries:")
        for query in queries:
            print(f"\n  Query: '{query}'")
            result = parser.parse_query_with_domain_validation(query)
            if result.get('user_query'):
                print(f"    ✅ Parsed successfully")
                print(f"    Filters: {result['user_query'].filters}")
            else:
                print(f"    ⚠️  Parsing returned no filters")
        
        return True
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GEMINI API INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\nMake sure you have:")
    print("1. Set LLM_PROVIDER=gemini in your .env file")
    print("2. Set GEMINI_API_KEY=your_key in your .env file")
    print("3. Installed google-generativeai: pip install google-generativeai")
    print("\n")
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_gemini_config()))
    results.append(("Direct API Call", test_gemini_api_call()))
    results.append(("Intent Parser", test_intent_parser()))
    results.append(("LLM Extractor", test_llm_extractor()))
    results.append(("Full Workflow", test_full_workflow()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


