# Comparison: test_parse_intent_standalone.py vs parse_intent Method

## Overview

**test_parse_intent_standalone.py** is a **TEST SCRIPT** that calls and tests the `parse_intent` method.
**parse_intent** is the **ACTUAL METHOD** in the `GeminiDataExtractor` class that performs the intent parsing.

## Key Differences

### 1. **Purpose**

**test_parse_intent_standalone.py:**
- Test/utility script
- Wrapper around the actual method
- Used for debugging and testing
- Displays results in a human-readable format

**parse_intent method:**
- Core functionality
- Part of the `GeminiDataExtractor` class
- Performs the actual LLM API call and JSON parsing
- Returns structured data

### 2. **What They Do**

**test_parse_intent_standalone.py:**
```python
# 1. Loads data from Excel file
df = pd.read_excel("sample_radiology_data.xlsx")

# 2. Creates an instance of GeminiDataExtractor
extractor = GeminiDataExtractor()

# 3. CALLS the parse_intent method
intent_result = extractor.parse_intent(query, df)

# 4. Displays the results in a formatted way
print("INTENT:", intent_result.get('intent'))
print("VARIABLES:", intent_result.get('variables_of_interest'))
# etc.
```

**parse_intent method:**
```python
def parse_intent(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
    # 1. Creates a prompt based on query and dataset schema
    prompt = self._create_extraction_prompt(query, df)
    
    # 2. Calls Gemini LLM API
    response = self.model.generate_content(prompt, ...)
    
    # 3. Parses JSON response
    result = json.loads(response.text)
    
    # 4. Returns the parsed result
    return result
```

### 3. **Logic Flow**

**test_parse_intent_standalone.py:**
```
Load Data → Initialize Extractor → Call parse_intent() → Display Results
```

**parse_intent method:**
```
Create Prompt → Call Gemini API → Parse JSON → Return Result
```

### 4. **Dependencies**

**test_parse_intent_standalone.py:**
- Depends on `parse_intent` method
- Needs the `GeminiDataExtractor` class
- Requires data file (sample_radiology_data.xlsx)

**parse_intent method:**
- Part of `GeminiDataExtractor` class
- Uses `_create_extraction_prompt` helper method
- Calls Gemini API directly
- No external data file needed (receives DataFrame as parameter)

### 5. **Input/Output**

**test_parse_intent_standalone.py:**
- **Input:** Query string (optional, has default)
- **Output:** Prints formatted results to console, returns intent_result dictionary

**parse_intent method:**
- **Input:** 
  - `query`: str - User's query
  - `df`: pd.DataFrame - Dataset
- **Output:** 
  - Dictionary with:
    - `intent`: Description
    - `variables_of_interest`: List of column names
    - `extraction_plan`: Detailed plan including narrative_extraction
    - `output_schema`: Schema description

### 6. **Error Handling**

**test_parse_intent_standalone.py:**
- Catches exceptions and prints error messages
- Shows traceback for debugging
- Returns None on error

**parse_intent method:**
- Has retry logic (using @retry decorator)
- Logs errors
- Raises exceptions to be handled by caller

## Relationship Diagram

```
┌─────────────────────────────────────┐
│  test_parse_intent_standalone.py    │
│  (Test Script)                      │
│                                     │
│  1. Load data                       │
│  2. Create extractor                │
│  3. CALLS ──────────────────┐      │
│  4. Display results          │      │
└──────────────────────────────│──────┘
                               │
                               ▼
┌─────────────────────────────────────┐
│  GeminiDataExtractor class          │
│                                     │
│  def parse_intent(...):            │
│    1. Create prompt                │
│    2. Call Gemini API              │
│    3. Parse JSON                   │
│    4. RETURN result ◄──────────────┘
└─────────────────────────────────────┘
```

## Summary

- **test_parse_intent_standalone.py** is a **test/utility script** that tests the `parse_intent` method
- **parse_intent** is the **actual implementation** of the intent parsing logic
- The test script **depends on** and **calls** the parse_intent method
- They serve different purposes: testing vs. core functionality

