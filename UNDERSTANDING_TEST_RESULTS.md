# Understanding Test Results - Guide

## 📋 Which Test Did You Run?

There are two test files:
1. **`test_gemini_models.py`** - Lists available Gemini models
2. **`test_gemini.py`** - Tests the full integration

## 🔍 Understanding `test_gemini_models.py` Results

### What This Test Does:
- Lists all Gemini models available with your API key
- Shows which model names you can use
- Tests if the API connection works

### Expected Output Format:

```
============================================================
AVAILABLE GEMINI MODELS
============================================================
✅ models/gemini-1.5-flash-latest
   Display Name: Gemini 1.5 Flash
   Supported Methods: ['generateContent', ...]

✅ models/gemini-1.5-pro-latest
   Display Name: Gemini 1.5 Pro
   Supported Methods: ['generateContent', ...]

============================================================
RECOMMENDED MODEL NAMES FOR .env FILE
============================================================
GEMINI_MODEL_NAME=gemini-1.5-flash-latest
GEMINI_MODEL_NAME=gemini-1.5-pro-latest

============================================================
TESTING MODEL
============================================================
Testing with model: gemini-1.5-flash-latest
✅ Model test successful!
Response: {"message": "Hello, Gemini is working!"}
```

### What Each Section Means:

#### 1. **AVAILABLE GEMINI MODELS**
- Shows all models your API key can access
- **Model Name**: The full API path (e.g., `models/gemini-1.5-flash-latest`)
- **Display Name**: Human-readable name
- **Supported Methods**: What the model can do

#### 2. **RECOMMENDED MODEL NAMES**
- Shows the **short name** to use in your `.env` file
- Copy one of these values to your `.env` file
- Example: `GEMINI_MODEL_NAME=gemini-1.5-flash-latest`

#### 3. **TESTING MODEL**
- Tests if the API actually works
- If you see ✅, the API is working!
- If you see ❌, there's an error (see troubleshooting below)

---

## 🔍 Understanding `test_gemini.py` Results

### What This Test Does:
- Tests configuration
- Tests direct API calls
- Tests intent parser
- Tests LLM extractor
- Tests full workflow

### Expected Output Format:

```
============================================================
GEMINI API INTEGRATION TEST SUITE
============================================================

============================================================
TEST 1: Configuration Check
============================================================
LLM Provider: gemini
Model Name: gemini-1.5-flash-latest
Gemini API Key: Set
✅ LLM Client created successfully

============================================================
TEST 2: Direct Gemini API Call
============================================================
✅ API call successful
Response: {"goal": "Count patients", "filters": {"stone_presence": "absent"}, ...}

============================================================
TEST 3: Intent Parser with Gemini
============================================================
Testing query: 'how many patients have no kidney stone?'
✅ Parsing successful
   Method: llm_primary
   Goal: Count patients
   Filters: {'stone_presence': 'absent'}
   Outputs: ['recordid']

============================================================
TEST 4: LLM Extractor with Gemini
============================================================
Testing extraction with narrative: 'Right kidney stone measuring 1.2 cm...'
✅ Extraction successful
   Right stone status: present
   Right stone size: 1.2
   Bladder volume: 250

============================================================
TEST 5: Full Workflow Test
============================================================
Testing multiple queries:
  Query: 'how many patients have no kidney stone?'
    ✅ Parsed successfully
    Filters: {'stone_presence': 'absent'}
  Query: 'find patients with left kidney stones > 1cm'
    ✅ Parsed successfully
    Filters: {'side': 'left', 'min_size_cm': 1.0}
  Query: 'what is the mean bladder volume?'
    ✅ Parsed successfully
    Filters: {}

============================================================
TEST SUMMARY
============================================================
✅ PASS: Configuration
✅ PASS: Direct API Call
✅ PASS: Intent Parser
✅ PASS: LLM Extractor
✅ PASS: Full Workflow

Total: 5/5 tests passed

🎉 All tests passed! Gemini integration is working correctly.
```

### What Each Test Means:

#### **TEST 1: Configuration Check**
- ✅ **PASS**: Your `.env` file is set up correctly
- ❌ **FAIL**: Check your `.env` file has `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`

#### **TEST 2: Direct API Call**
- ✅ **PASS**: Gemini API is working, you can make API calls
- ❌ **FAIL**: Check your API key is valid, or model name is correct

#### **TEST 3: Intent Parser**
- ✅ **PASS**: Query parsing with Gemini works
- ❌ **FAIL**: Check model name or API key

#### **TEST 4: LLM Extractor**
- ✅ **PASS**: Data extraction with Gemini works
- ❌ **FAIL**: Check model name or API key

#### **TEST 5: Full Workflow**
- ✅ **PASS**: Everything works end-to-end
- ❌ **FAIL**: One of the components has an issue

---

## 🐛 Common Error Messages Explained

### ❌ "google-generativeai package not installed"
**Meaning**: The Gemini package isn't installed  
**Fix**: 
```bash
pip install google-generativeai
```

### ❌ "GEMINI_API_KEY not set in .env file"
**Meaning**: Your API key isn't configured  
**Fix**: Add to `.env` file:
```bash
GEMINI_API_KEY=your_actual_key_here
```

### ❌ "404 models/gemini-1.5-flash is not found"
**Meaning**: The model name is wrong  
**Fix**: 
1. Run `python test_gemini_models.py` to see available models
2. Update `.env` with the correct model name from the output

### ❌ "Invalid API key"
**Meaning**: Your API key is wrong or expired  
**Fix**: 
1. Get a new key from https://makersuite.google.com/app/apikey
2. Update your `.env` file

### ❌ "429 you exceeded your current quota" or "Rate limit exceeded"
**Meaning**: You've exceeded your Gemini API quota/rate limit  
**This is NOT a code problem - it's a quota/billing issue**

**Solutions**:

1. **Wait and Retry** (for rate limits):
   - Wait 1-2 hours and try again
   - Rate limits usually reset hourly or daily

2. **Check Your Quota** (for quota limits):
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Check your usage and quota limits
   - Free tier has limited requests per day

3. **Upgrade Your Plan**:
   - Free tier: ~15 requests per minute, limited daily quota
   - Paid tier: Higher limits
   - Visit [Google Cloud Console](https://console.cloud.google.com/) to upgrade

4. **Switch to OpenAI Temporarily**:
   - If you have OpenAI credits, switch back:
   ```bash
   # In your .env file, change:
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_key
   ```

5. **Use Pattern Matching Only** (no API calls):
   - The agent can work without LLM using pattern matching
   - Set `--dry-run` flag to avoid API calls
   - Or wait for quota to reset

**Important**: The 429 error means your API key is valid, but you've used up your quota. The code is working correctly!

---

## ✅ What "PASS" vs "FAIL" Means

### ✅ PASS
- **Configuration**: Your settings are correct
- **API Call**: Gemini API is responding
- **Parsing**: Query parsing works
- **Extraction**: Data extraction works
- **Workflow**: Everything works together

### ❌ FAIL
- Something is wrong - check the error message
- Usually it's: missing package, wrong API key, or wrong model name

---

## 📝 Next Steps Based on Results

### If All Tests Pass ✅
1. Your Gemini integration is working!
2. You can run the main agent:
   ```bash
   python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
   ```

### If Tests Fail ❌
1. **Check which test failed** - look at the test name
2. **Read the error message** - it tells you what's wrong
3. **Fix the issue** using the troubleshooting guide above
4. **Run the test again** to verify the fix

---

## 🎯 Quick Diagnostic

**Paste your test output here and I can help interpret it!**

Or answer these questions:
1. Which test file did you run? (`test_gemini.py` or `test_gemini_models.py`)
2. What does the output show? (Copy/paste the last 20-30 lines)
3. Do you see any ❌ (red X) or ✅ (green checkmark)?

---

## 💡 Example: Good vs Bad Results

### ✅ Good Result:
```
✅ PASS: Configuration
✅ PASS: Direct API Call
Total: 2/2 tests passed
🎉 All tests passed!
```

### ❌ Bad Result:
```
✅ PASS: Configuration
❌ FAIL: Direct API Call
Error: 404 models/gemini-1.5-flash is not found
Total: 1/2 tests passed
```

**For the bad result**: The model name is wrong. Run `test_gemini_models.py` to find the correct name.

---

**Share your test output and I'll help you understand what it means!** 🚀

