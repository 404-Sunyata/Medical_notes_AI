# Gemini Data Extractor - Processing Pipeline

## 🔄 Complete Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     START: User Query + Dataset                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 0: INITIALIZATION                                      │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GeminiDataExtractor.__init__()                                     │   │
│  │                                                                      │   │
│  │ 1. Model Setup                                                      │   │
│  │    • Load GEMINI_API_KEY from config                                │   │
│  │    • Configure Google Generative AI                                 │   │
│  │    • Try multiple model variants with fallback                      │   │
│  │    • Auto-detect available models if primary fails                  │   │
│  │                                                                      │   │
│  │ Returns: Initialized extractor with Gemini model                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 1: SCHEMA ANALYSIS & NARRATIVE DETECTION                   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Schema Analysis (Code-Based)                                        │   │
│  │                                                                      │   │
│  │ 1. _create_schema_analysis_prompt()                                │   │
│  │    • Analyze DataFrame structure                                    │   │
│  │    • Identify column types (numeric, datetime, text/categorical)    │   │
│  │    • Extract statistics (range, unique values, null counts)         │   │
│  │    • Sample values from text columns                                │   │
│  │                                                                      │   │
│  │ 2. _detect_narrative_columns()                                      │   │
│  │    • Check for common narrative column names                        │   │
│  │      (narrative, notes, report, text, history_summary, etc.)        │   │
│  │    • Detect long text columns (>100 chars avg length)               │   │
│  │                                                                      │   │
│  │ Returns: Schema info string and list of narrative columns           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 2: LLM-ONLY INTENT PARSING                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ parse_intent() - LLM-Only                                           │   │
│  │                                                                      │   │
│  │ 1. Prompt Generation                                                │   │
│  │    • _create_extraction_prompt()                                    │   │
│  │    • Include schema information from Phase 1                        │   │
│  │    • Add narrative column details                                   │   │
│  │    • Detect negation and bilateral conditions in query              │   │
│  │    • Add special instructions for narrative extraction              │   │
│  │                                                                      │   │
│  │ 2. LLM Call (Gemini API)                                            │   │
│  │    • Call Gemini with JSON response format                          │   │
│  │    • Temperature: 0.1 (low for consistency)                         │   │
│  │    • Retry logic with exponential backoff                           │   │
│  │                                                                      │   │
│  │ 3. JSON Parsing & Validation                                        │   │
│  │    • Extract JSON from markdown code blocks if present              │   │
│  │    • Fix common JSON issues (trailing commas)                       │   │
│  │    • Parse into structured dictionary                               │   │
│  │                                                                      │   │
│  │ Returns: Intent result with:                                        │   │
│  │    • intent: Description of user's intent                           │   │
│  │    • variables_of_interest: List of relevant columns                │   │
│  │    • extraction_plan: Detailed extraction strategy                  │   │
│  │      - primary_columns: Main columns to extract                     │   │
│  │      - narrative_extraction: Narrative parsing config               │   │
│  │      - filters: Pre-filtering conditions                            │   │
│  │      - grouping: Grouping columns                                   │   │
│  │      - aggregations: Aggregation functions                          │   │
│  │    • output_schema: Description of output columns                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: VARIABLE EXTRACTION                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ extract_variables()                                                 │   │
│  │                                                                      │   │
│  │ 1. Apply Pre-Filters (Pandas-based)                                │   │
│  │    • Apply filters from extraction_plan                             │   │
│  │    • Handle range conditions (min/max)                              │   │
│  │    • Handle equality and contains filters                           │   │
│  │                                                                      │   │
│  │ 2. Select Variables of Interest                                    │   │
│  │    • Extract columns from variables_of_interest                     │   │
│  │    • Fallback to all columns if none found                         │   │
│  │                                                                      │   │
│  │ 3. Grouping & Aggregation (if needed)                              │   │
│  │    • Apply grouping on specified columns                            │   │
│  │    • Apply aggregation functions (count, sum, mean, min, max)       │   │
│  │                                                                      │   │
│  │ 4. Narrative Extraction (if needed)                                │   │
│  │    • Check if narrative_extraction is requested                     │   │
│  │    • If yes → Proceed to Phase 4                                    │   │
│  │                                                                      │   │
│  │ 5. Apply Narrative Filters (Rule-based)                            │   │
│  │    • Detect if query asks for filtering                             │   │
│  │    • Apply filters based on extracted narrative values              │   │
│  │    • Handle negation and bilateral conditions                       │   │
│  │                                                                      │   │
│  │ Returns: DataFrame with extracted variables                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          ┌─────────▼─────────┐          ┌─────────▼──────────┐
          │ No Narrative      │          │ Narrative          │
          │ Extraction        │          │ Extraction Needed  │
          │ Needed            │          │                    │
          └─────────┬─────────┘          └─────────┬──────────┘
                    │                               │
                    │                               ▼
                    │          ┌──────────────────────────────────────────┐
                    │          │     PHASE 4: NARRATIVE EXTRACTION        │
                    │          │                                          │
                    │          │  ┌────────────────────────────────────┐  │
                    │          │  │ _extract_from_narrative()          │  │
                    │          │  │                                    │  │
                    │          │  │ 1. Merge Narrative Data            │  │
                    │          │  │    • Merge result_df with          │  │
                    │          │  │      source_df narrative columns   │  │
                    │          │  │    • Align by recordid/index       │  │
                    │          │  │                                    │  │
                    │          │  │ 2. Iterate Through Rows            │  │
                    │          │  │    • For each row in DataFrame     │  │
                    │          │  │    • Extract narrative text        │  │
                    │          │  │    • Skip empty/null narratives    │  │
                    │          │  │                                    │  │
                    │          │  │ 3. LLM-Based Extraction            │  │
                    │          │  │    • Call _parse_single_narrative()│  │
                    │          │  │      for each row (LLM per row)    │  │
                    │          │  │    • Extract specified fields      │  │
                    │          │  │    • Handle presence/absence       │  │
                    │          │  │    • Progress logging              │  │
                    │          │  │                                    │  │
                    │          │  │ 4. Add Extracted Columns           │  │
                    │          │  │    • Add extracted fields as       │  │
                    │          │  │      new columns                   │  │
                    │          │  │    • Align with result_df order    │  │
                    │          │  │                                    │  │
                    │          │  │ Returns: DataFrame with extracted  │  │
                    │          │  │           narrative fields added   │  │
                    │          │  └────────────────────────────────────┘  │
                    │          │                                          │
                    │          │  ┌────────────────────────────────────┐  │
                    │          │  │ _parse_single_narrative() - LLM    │  │
                    │          │  │                                    │  │
                    │          │  │ 1. Build Extraction Prompt         │  │
                    │          │  │    • Include narrative text        │  │
                    │          │  │    • List fields to extract        │  │
                    │          │  │    • Add field descriptions        │  │
                    │          │  │    • Special rules for medical     │  │
                    │          │  │      fields (left/right kidney)    │  │
                    │          │  │    • Handle presence/absence       │  │
                    │          │  │      detection                     │  │
                    │          │  │                                    │  │
                    │          │  │ 2. LLM Call (Gemini API)           │  │
                    │          │  │    • Single row extraction         │  │
                    │          │  │    • JSON response format          │  │
                    │          │  │    • Retry logic                   │  │
                    │          │  │                                    │  │
                    │          │  │ 3. Parse & Normalize               │  │
                    │          │  │    • Extract JSON from response    │  │
                    │          │  │    • Normalize values              │  │
                    │          │  │      (null, true/false → present/  │  │
                    │          │  │       absent)                      │  │
                    │          │  │                                    │  │
                    │          │  │ Returns: Dict with extracted       │  │
                    │          │  │           field values             │  │
                    │          │  └────────────────────────────────────┘  │
                    │          └──────────────────────────────────────────┘
                    │                               │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              PHASE 5: NARRATIVE-BASED FILTERING                               │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _apply_narrative_filters() - Rule-Based                             │   │
│  │                                                                      │   │
│  │ 1. Detect Filtering Intent                                          │   │
│  │    • Check if query asks for filtering                              │   │
│  │      (e.g., "which", "who has", "patients with")                   │   │
│  │                                                                      │   │
│  │ 2. Detect Query Features                                            │   │
│  │    • Negation detection: "no", "not", "don't", "without"           │   │
│  │    • Bilateral detection: "both sides", "bilateral", "either side" │   │
│  │                                                                      │   │
│  │ 3. Field Matching                                                   │   │
│  │    • Match query terms to extracted field names                     │   │
│  │    • Use medical term variations                                    │   │
│  │    • Score matches and select best field                            │   │
│  │                                                                      │   │
│  │ 4. Apply Filters                                                    │   │
│  │    • Bilateral queries:                                             │   │
│  │      - Positive: (left_present OR right_present)                    │   │
│  │      - Negation: (left_absent AND right_absent)                     │   │
│  │    • Single-side queries:                                           │   │
│  │      - Positive: field = "present"                                  │   │
│  │      - Negation: field = "absent"                                   │   │
│  │    • Fallback: Filter by any "present" status field                 │   │
│  │                                                                      │   │
│  │ Returns: Filtered DataFrame                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 6: RESULT ORGANIZATION                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ extract_and_organize() - Orchestrator                               │   │
│  │                                                                      │   │
│  │ 1. Execute Pipeline                                                 │   │
│  │    • Call parse_intent() → Phase 2                                  │   │
│  │    • Call extract_variables() → Phase 3-5                           │   │
│  │                                                                      │   │
│  │ 2. Create Summary Statistics                                        │   │
│  │    • Original vs extracted row counts                               │   │
│  │    • Original vs extracted column counts                            │   │
│  │    • List of extracted variables                                    │   │
│  │    • Intent summary                                                 │   │
│  │                                                                      │   │
│  │ Returns: Dictionary with:                                           │   │
│  │    • intent: Parsed intent result                                   │   │
│  │    • extracted_data: DataFrame with extracted variables             │   │
│  │    • summary: Summary statistics                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           END: Return Results                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```
┌──────────────┐
│ User Query   │
│ + Dataset    │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              INITIALIZATION                              │
│  ┌──────────────┐                                       │
│  │ Load Config  │                                       │
│  │ Setup Gemini │                                       │
│  └──────┬───────┘                                       │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│          SCHEMA ANALYSIS (Code-Based)                   │
│  ┌──────────────┐      ┌────────────────────┐          │
│  │ Analyze      │      │ Detect Narrative   │          │
│  │ Columns      │─────▶│ Columns            │          │
│  └──────┬───────┘      └─────────┬──────────┘          │
└─────────┼─────────────────────────┼─────────────────────┘
          │                         │
          └──────────────┬──────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   LLM INTENT PARSING         │
          │   (Gemini API Call)          │
          │                              │
          │  • Understand query intent   │
          │  • Identify variables        │
          │  • Plan extraction           │
          │  • Specify narrative needs   │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   VARIABLE EXTRACTION        │
          │                              │
          │  ┌──────────────────────┐   │
          │  │ Apply Filters        │   │
          │  │ (Pandas-based)       │   │
          │  └──────────┬───────────┘   │
          │             │                │
          │  ┌──────────▼───────────┐   │
          │  │ Select Variables     │   │
          │  └──────────┬───────────┘   │
          │             │                │
          │  ┌──────────▼───────────┐   │
          │  │ Group & Aggregate    │   │
          │  └──────────┬───────────┘   │
          └─────────────┼───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        │    ┌──────────────────────┐   │
        │    │ Narrative Needed?    │   │
        │    └────┬──────────────┬──┘   │
        │         │              │      │
        │    ┌────▼─────┐  ┌────▼─────┐│
        │    │    No    │  │   Yes    ││
        │    └────┬─────┘  └────┬─────┘│
        │         │              │      │
        │         │      ┌───────▼───────┐
        │         │      │ NARRATIVE     │
        │         │      │ EXTRACTION    │
        │         │      │               │
        │         │      │ For each row: │
        │         │      │ ┌──────────┐  │
        │         │      │ │ LLM Call │  │
        │         │      │ │ (Gemini) │  │
        │         │      │ └────┬─────┘  │
        │         │      │      │        │
        │         │      │ Add extracted │
        │         │      │ columns       │
        │         │      └───────┬───────┘
        │         │              │
        │         └──────┬───────┘
        │                │
        │                ▼
        │    ┌───────────────────────┐
        │    │ NARRATIVE FILTERING   │
        │    │ (Rule-Based)          │
        │    │                       │
        │    │ • Detect negation     │
        │    │ • Detect bilateral    │
        │    │ • Apply filters       │
        │    └───────────┬───────────┘
        │                │
        └────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   ORGANIZE RESULTS           │
          │                              │
          │  • Create summary            │
          │  • Return structured data    │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   Output:                    │
          │   • intent                   │
          │   • extracted_data (DF)      │
          │   • summary                  │
          └──────────────────────────────┘
```

---

## 🔑 Key Decision Points

### 1. Model Initialization
- **API Key Available?** → Initialize Gemini model
- **API Key Missing?** → Raise ValueError
- **Primary Model Fails?** → Try fallback variants
- **All Variants Fail?** → Auto-detect available models

### 2. Narrative Detection
- **Common Names Found?** → Mark as narrative column
- **Long Text (>100 chars avg)?** → Mark as narrative column
- **No Narrative Columns?** → Skip narrative extraction

### 3. Intent Parsing
- **LLM Available?** → Use Gemini for intent parsing
- **LLM Fails?** → Retry with exponential backoff (up to MAX_RETRIES)
- **JSON Parse Error?** → Attempt to fix (remove markdown, trailing commas)
- **All Retries Failed?** → Raise exception

### 4. Narrative Extraction
- **Narrative Extraction Requested?** → Process each row with LLM
- **Empty Narrative?** → Skip extraction, set fields to None
- **Extraction Fails?** → Set fields to None, continue with next row

### 5. Filtering
- **Query Asks for Filtering?** → Apply narrative filters
- **Negation Detected?** → Filter for "absent" values
- **Bilateral Query?** → Apply both-sides logic
  - **Positive Bilateral:** (left_present OR right_present)
  - **Negation Bilateral:** (left_absent AND right_absent)
- **Single Side Query?** → Filter by single field

---

## 🎯 LLM Usage Phases

### Phase 2: Intent Parsing (LLM)
- **Purpose**: Understand user intent and identify extraction requirements
- **Model**: Gemini (JSON mode)
- **Input**: User query + dataset schema + narrative column info
- **Output**: Structured extraction plan
- **Frequency**: Once per query

### Phase 4: Narrative Extraction (LLM per Row)
- **Purpose**: Extract structured fields from unstructured narrative text
- **Model**: Gemini (JSON mode)
- **Input**: Single narrative text + fields to extract + field descriptions
- **Output**: Extracted field values (present/absent/null)
- **Frequency**: Once per row that contains narrative text

---

## 📈 Performance Metrics

### Processing Times (Estimated)
- **Initialization**: ~100-500ms (model setup)
- **Schema Analysis**: ~10-50ms (code-based)
- **Intent Parsing (LLM)**: ~500-2000ms (single API call)
- **Variable Extraction**: ~10-100ms (Pandas operations)
- **Narrative Extraction**: ~500-2000ms per row (LLM calls)
  - **Total for N rows**: N × 500-2000ms
- **Narrative Filtering**: ~10-50ms (rule-based)

### Accuracy Rates (Estimated)
- **Schema Analysis**: ~100% (deterministic)
- **Intent Parsing**: ~85-95% (depends on query clarity)
- **Narrative Extraction**: ~80-90% (depends on text quality)
- **Filtering**: ~95-98% (rule-based, deterministic)

---

## 🔄 Error Handling & Retries

### LLM API Calls
- **Retry Strategy**: Exponential backoff
- **Max Retries**: MAX_RETRIES (default: 3)
- **Wait Times**: 2-10 seconds (exponential)
- **Retry On**: Any exception during API call

### JSON Parsing
- **Markdown Extraction**: Remove ```json and ``` blocks
- **Trailing Comma Fix**: Remove trailing commas before closing braces/brackets
- **Fallback**: Return None for failed fields

### Narrative Extraction
- **Empty Narrative**: Skip, set fields to None
- **Extraction Failure**: Log warning, set fields to None, continue
- **Length Mismatch**: Pad or truncate to match DataFrame length

---

## 💡 Design Decisions

### 1. Hybrid LLM + Rule-Based Approach
- **LLM for**: Intent understanding, narrative parsing (unstructured → structured)
- **Rule-Based for**: Filtering, aggregation, schema analysis (deterministic operations)

### 2. Per-Row Narrative Extraction
- **Trade-off**: Accuracy vs. Performance
- **Choice**: Extract per row for maximum accuracy (can be slow for large datasets)
- **Future**: Could batch multiple narratives per LLM call

### 3. Separate Filtering Phase
- **Reason**: Allows extracted narrative values to be filtered using deterministic rules
- **Benefit**: Combines LLM extraction accuracy with rule-based filtering reliability

### 4. Model Fallback Strategy
- **Reason**: Gemini model names may vary (1.5 vs 2.5, latest suffixes)
- **Approach**: Try multiple variants, then auto-detect from available models

---

**Last Updated**: 2024-12-19
**Version**: 1.0
