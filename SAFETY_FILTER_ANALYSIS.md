# Safety Filter Trigger Analysis

## 🔴 Parts of Code That Trigger Safety Filters

### 1. **`_detect_new_medical_concepts` method** (Line 1268-1290 in `src/query_tools.py`)

**Location**: `src/query_tools.py:1268-1290`

**Triggering Prompt**:
```python
prompt = f"""
Analyze this medical query and identify any medical concepts that might require filtering:
Query: "{query_text}"

Available columns: {', '.join(self.available_columns)}

Identify medical concepts that could be used for filtering (e.g., hydronephrosis, infection, pain, etc.).
Return only the concept names, one per line.
If no new medical concepts are found, return "NONE".
"""
```

**Why it triggers**: 
- Contains medical terminology: "hydronephrosis", "infection", "pain"
- Mentions "medical concepts" and "filtering" which might be interpreted as dangerous content
- The word "infection" might trigger health-related safety filters

**Error Log**:
```
2025-11-13 11:32:30,603 - src.query_tools - WARNING - Gemini API blocked content (finish_reason=SAFETY)
2025-11-13 11:32:30,603 - src.query_tools - ERROR - Failed to detect medical concepts: Gemini API blocked the content due to safety filters
```

---

### 2. **`_detect_new_statistics` method** (Line 1328-1350 in `src/query_tools.py`)

**Location**: `src/query_tools.py:1328-1350`

**Triggering Prompt**:
```python
prompt = f"""
Analyze this query and identify statistical operations that are not in the standard set:
Query: "{query_text}"

Standard statistics: mean, median, max, min, sum, count, std

Identify any non-standard statistical operations (e.g., variance, mode, range, percentile, etc.).
Return only the operation names, one per line.
If no new statistical operations are found, return "NONE".
"""
```

**Why it triggers**:
- Less likely to trigger, but the combination of "analyze", "identify", and medical query context might still trigger filters
- If the `query_text` contains medical terms, the entire prompt gets flagged

**Error Log**:
```
2025-11-13 11:32:34,098 - src.query_tools - WARNING - Gemini API blocked content (finish_reason=SAFETY)
2025-11-13 11:32:34,098 - src.query_tools - ERROR - Failed to detect new statistics: Gemini API blocked the content due to safety filters
```

---

## 🐛 NoneType Error Locations

### 1. **`estimate_matching_rows` → `apply_filters` → `_validate_and_suggest_filters`** (Line 424)

**Location**: `src/query_tools.py:424`

**Problem Code**:
```python
if query_context:
    query_text = query_context.get('query_text', '').lower()  # Line 424
```

**Issue**: If `query_context.get('query_text')` returns `None` (key exists but value is None), calling `.lower()` on `None` fails.

**Fix Needed**:
```python
if query_context:
    query_text = (query_context.get('query_text') or '').lower()
```

---

### 2. **`_get_specific_statistics` method** (Line 904)

**Location**: `src/query_tools.py:904`

**Problem Code**:
```python
def _get_specific_statistics(self, filtered_df: pd.DataFrame, query_text: str) -> Dict[str, Any]:
    specific_stats = {}
    query_lower = query_text.lower()  # Line 904 - fails if query_text is None
```

**Issue**: Even though the type hint says `str`, `query_text` could be `None` if called incorrectly.

**Fix Needed**:
```python
def _get_specific_statistics(self, filtered_df: pd.DataFrame, query_text: Optional[str]) -> Dict[str, Any]:
    specific_stats = {}
    if not query_text:
        return specific_stats
    query_lower = query_text.lower()
```

---

## 📋 Summary

### Safety Filter Triggers:
1. ✅ **`_detect_new_medical_concepts`** - Prompt contains medical terms
2. ✅ **`_detect_new_statistics`** - Prompt analyzed with medical query context

### NoneType Errors:
1. ✅ **Line 424** - `query_context.get('query_text', '').lower()` when value is None
2. ✅ **Line 904** - `query_text.lower()` when query_text is None

---

## 🔧 Recommended Fixes

### Fix 1: Make prompts less likely to trigger safety filters
- Remove specific medical examples from prompts
- Use more generic language
- Add context that this is for medical research/analysis

### Fix 2: Add None checks before calling `.lower()`
- Check if value is None before string operations
- Use `or ''` to provide default empty string

### Fix 3: Consider disabling dynamic learning for Gemini
- These features are optional and can be disabled
- Pattern matching works without LLM calls

