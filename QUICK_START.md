# Quick Start Guide - Testing Gemini Integration

## ✅ Your Command is Correct!

```bash
python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
```

This is the right command to run the agent with Gemini.

## 📋 Before Running - Setup Checklist

### 1. Install Gemini Package
```bash
pip install google-generativeai
```

### 2. Get Your Gemini API Key
- Go to: https://makersuite.google.com/app/apikey
- Sign in and create an API key
- Copy the key

### 3. Configure Environment Variables

**Option A: Create/Update `.env` file** (Recommended)
```bash
# Create or edit .env file in the project root
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

**Option B: Export in Terminal**
```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run the Agent
```bash
python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
```

## 🧪 Test Commands

### Basic Query Test
```bash
python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
```

### Test with Different Queries
```bash
# Count query
python main.py sample_radiology_data.xlsx --query "how many patients have left kidney stones > 1cm?" --confirm off

# Statistical query
python main.py sample_radiology_data.xlsx --query "what is the mean bladder volume?" --confirm off

# Patient search
python main.py sample_radiology_data.xlsx --query "which patient has the biggest stone?" --confirm off
```

### Run Test Suite
```bash
python test_gemini.py
```

## 🔍 Verify Gemini is Being Used

When you run the command, you should see in the logs:
```
INFO - LLM Provider: gemini
INFO - Using LLM to parse query: ...
INFO - Gemini API call successful: ...
```

If you see "OpenAI client not available" or "Pattern matching" instead, check:
1. `LLM_PROVIDER=gemini` is set correctly
2. `GEMINI_API_KEY` is set correctly
3. The `.env` file is in the project root directory

## 🐛 Troubleshooting

### Error: "Google Generative AI package not installed"
```bash
pip install google-generativeai
```

### Error: "GEMINI_API_KEY not set"
- Check your `.env` file exists and has the key
- Or export it: `export GEMINI_API_KEY=your_key`

### Error: "Invalid API key"
- Verify your API key at https://makersuite.google.com/app/apikey
- Make sure there are no extra spaces in `.env` file
- Restart your terminal after updating `.env`

### Still Using OpenAI?
- Check: `echo $LLM_PROVIDER` (should show "gemini")
- Check your `.env` file has `LLM_PROVIDER=gemini` (not "openai")
- Restart your terminal/IDE

## 📝 Example .env File

Create a file named `.env` in the project root:

```bash
# LLM Provider (choose: "openai" or "gemini")
LLM_PROVIDER=gemini

# Gemini Configuration
GEMINI_API_KEY=AIzaSy...your_actual_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash

# Optional: Override model name
# MODEL_NAME=gemini-1.5-flash
```

## ✅ Expected Output

When everything is working, you should see:
```
INFO - LLM Provider: gemini
INFO - Using LLM to parse query: how many patients have no kidney stone?
INFO - Gemini API call successful: ... tokens (estimated), $0.0000, XXXms
INFO - Successfully parsed with LLM
...
============================================================
QUERY RESULTS
============================================================
Total records: 12
Unique patients: 12
...
```

---

**Your command is correct!** Just make sure you've set up the Gemini API key first. 🚀


