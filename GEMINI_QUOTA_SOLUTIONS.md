# 🔴 Gemini API 429 Quota Error - Solutions

## What Does This Error Mean?

**Error**: `429 you exceeded your current quota`

This means:
- ✅ Your API key is **valid** and working
- ✅ Your code is **correct**
- ❌ You've **exceeded your quota/rate limit**

This is a **billing/quota issue**, not a code problem.

---

## 🎯 Quick Solutions

### Option 1: Wait for Quota Reset (Recommended for Free Tier)
- **Free tier quotas** reset daily (usually at midnight UTC)
- Wait 1-24 hours and try again
- Check your quota status at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Option 2: Switch Back to OpenAI (Temporary)
If you have OpenAI credits available:

1. **Update your `.env` file**:
   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_key_here
   MODEL_NAME=gpt-4o-mini
   ```

2. **Test it works**:
   ```bash
   python test_gemini.py
   ```

3. **Run your agent**:
   ```bash
   python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
   ```

### Option 3: Use Pattern Matching Only (No API Calls)
The agent can work without LLM using regex patterns:

```bash
# Use --dry-run to avoid API calls
python main.py sample_radiology_data.xlsx --dry-run --query "how many patients have no kidney stone?" --confirm off
```

**Note**: Pattern matching is less accurate than LLM, but works for simple queries.

### Option 4: Upgrade Your Gemini Plan
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "APIs & Services" > "Quotas"
3. Request quota increase or upgrade to paid tier
4. Free tier limits:
   - ~15 requests per minute
   - Limited daily quota (varies by region)

---

## 📊 Understanding Gemini Quotas

### Free Tier Limits:
- **Requests per minute**: ~15
- **Daily quota**: Varies (check your dashboard)
- **Models**: Limited to certain models

### Paid Tier:
- Higher rate limits
- More daily quota
- Access to all models

---

## 🔍 How to Check Your Quota Status

1. **Visit Google AI Studio**:
   - https://makersuite.google.com/app/apikey
   - Log in with your Google account
   - Check usage dashboard

2. **Check Google Cloud Console**:
   - https://console.cloud.google.com/
   - Go to "APIs & Services" > "Dashboard"
   - Look for "Generative Language API"

---

## ⏰ When Will Quota Reset?

- **Rate limits**: Usually reset every hour
- **Daily quota**: Resets at midnight UTC (check your timezone)
- **Monthly quota**: Resets on the 1st of each month

---

## 🛠️ Temporary Workaround: Use OpenAI

Since your code already supports both providers, you can easily switch:

### Step 1: Get OpenAI API Key (if you don't have one)
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add credits to your account

### Step 2: Update `.env` File
```bash
# Switch to OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini

# Comment out Gemini (or leave it for later)
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...
```

### Step 3: Test
```bash
python test_gemini.py  # This will now test OpenAI
```

### Step 4: Run Your Agent
```bash
python main.py sample_radiology_data.xlsx --query "how many patients have no kidney stone?" --confirm off
```

---

## 📝 Summary

| Solution | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Wait for reset** | Free tier, not urgent | No cost | Must wait |
| **Switch to OpenAI** | Have OpenAI credits | Immediate | Costs money |
| **Use --dry-run** | Testing only | No API calls | Less accurate |
| **Upgrade plan** | Need more quota | Higher limits | Costs money |

---

## ✅ Recommended Action

**For now**: Switch to OpenAI temporarily if you have credits, or wait for quota reset.

**For later**: Consider upgrading Gemini plan if you need higher limits.

---

## 🆘 Still Having Issues?

1. **Check your quota status** at Google AI Studio
2. **Verify your API key** is correct
3. **Try a different API key** (create a new one)
4. **Contact Google Support** if quota doesn't reset

---

**The good news**: Your code is working! The 429 error just means you need more quota. 🎉

