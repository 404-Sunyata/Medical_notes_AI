# Gemini API Setup Guide

## Overview
The Medical AI Agent now supports both OpenAI and Google Gemini APIs. You can easily switch between providers by setting the `LLM_PROVIDER` environment variable.

## Setup Instructions

### 1. Install Required Package
```bash
pip install google-generativeai>=0.3.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 3. Configure Environment Variables

Create or update your `.env` file:

```bash
# Choose your LLM provider: "openai" or "gemini"
LLM_PROVIDER=gemini

# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash

# Optional: Override with specific model name
MODEL_NAME=gemini-1.5-flash
```

### 4. Available Gemini Models
- `gemini-1.5-flash` (Recommended - Fast and free tier)
- `gemini-1.5-pro` (More capable, free tier)
- `gemini-pro` (Legacy, free tier)

### 5. Test the Setup

Run a test query:
```bash
python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
```

## Switching Between Providers

### Use Gemini:
```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key_here
```

### Use OpenAI:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
```

## Differences Between Providers

### OpenAI
- ✅ Structured JSON schema support
- ✅ Token usage tracking
- ✅ Cost tracking
- ⚠️ Requires paid API key

### Gemini
- ✅ Free tier available
- ✅ JSON response format support
- ✅ Fast response times
- ⚠️ Token usage is estimated (not exact)
- ⚠️ Different API structure

## Troubleshooting

### Issue: "Google Generative AI package not installed"
**Solution**: Run `pip install google-generativeai`

### Issue: "GEMINI_API_KEY not set"
**Solution**: Make sure your `.env` file contains `GEMINI_API_KEY=your_key_here`

### Issue: "Invalid API key"
**Solution**: 
1. Verify your API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Make sure there are no extra spaces in your `.env` file
3. Restart your terminal/IDE after updating `.env`

### Issue: "Rate limit exceeded"
**Solution**: 
- Gemini free tier has rate limits
- Wait a few minutes and try again
- Consider using `gemini-1.5-flash` for faster responses

## Cost Comparison

### OpenAI
- `gpt-4o-mini`: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- `gpt-4o`: ~$5 per 1M input tokens, $15 per 1M output tokens

### Gemini
- Free tier: No cost for most use cases
- Paid tier: Check [Google AI pricing](https://ai.google.dev/pricing)

## Performance Notes

- **Gemini 1.5 Flash**: Fastest, good for most tasks
- **Gemini 1.5 Pro**: More capable, slightly slower
- **OpenAI GPT-4o-mini**: Fast and cost-effective
- **OpenAI GPT-4o**: Most capable, higher cost

## Example Usage

### Test with Gemini:
```bash
# Set environment variables
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key_here

# Run the agent
python main.py sample_radiology_data.xlsx --query "how many patients have left kidney stones > 1cm?" --confirm off
```

### Test with OpenAI:
```bash
# Set environment variables
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here

# Run the agent
python main.py sample_radiology_data.xlsx --query "how many patients have left kidney stones > 1cm?" --confirm off
```

## Next Steps

1. Test both providers with your queries
2. Compare performance and accuracy
3. Choose the provider that works best for your use case
4. Update your `.env` file with your preferred settings

---

**Note**: The system will automatically fall back to pattern matching if the LLM provider is unavailable, so your workflow will continue to function even without an API key.


