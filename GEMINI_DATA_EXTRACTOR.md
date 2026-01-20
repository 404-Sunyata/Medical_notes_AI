# Gemini Data Extractor

A general-purpose data extraction module that uses Google Gemini to parse user intent and extract variables of interest from datasets.

## Overview

The `GeminiDataExtractor` class provides a flexible way to:
1. **Parse user intent** from natural language queries
2. **Identify variables of interest** from a dataset
3. **Extract and organize** those variables into a structured table

This module is designed to work with any dataset, not just medical/radiology data.

## Installation

Ensure you have the required dependencies:

```bash
pip install google-generativeai pandas
```

Set your Gemini API key in your environment or `.env` file:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Basic Usage

```python
import pandas as pd
from src.gemini_data_extractor import GeminiDataExtractor

# Create or load your dataset
df = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003'],
    'age': [45, 32, 67],
    'diagnosis': ['Hypertension', 'Diabetes', 'Hypertension'],
    'blood_pressure': [140, 120, 150]
})

# Initialize the extractor
extractor = GeminiDataExtractor()

# Extract variables based on a query
query = "Extract patient ID, age, and diagnosis for all patients with hypertension"
result = extractor.extract_and_organize(query, df)

# Access the results
print(result['intent'])          # Parsed intent information
print(result['extracted_data'])  # DataFrame with extracted variables
print(result['summary'])         # Summary statistics
```

## API Reference

### `GeminiDataExtractor(model_name=None)`

Initialize the Gemini data extractor.

**Parameters:**
- `model_name` (str, optional): Gemini model name. Defaults to `GEMINI_MODEL_NAME` from config.

**Raises:**
- `ValueError`: If `GEMINI_API_KEY` is not set.

### `parse_intent(query: str, df: pd.DataFrame) -> Dict[str, Any]`

Parse user intent and identify variables of interest.

**Parameters:**
- `query` (str): User's query/question
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- Dictionary with:
  - `intent`: Description of user's intent
  - `variables_of_interest`: List of relevant column names
  - `extraction_plan`: Plan for extraction (filters, grouping, aggregations)
  - `output_schema`: Description of output columns

### `extract_variables(query: str, df: pd.DataFrame) -> pd.DataFrame`

Extract variables of interest from the dataset.

**Parameters:**
- `query` (str): User's query/question
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- New DataFrame with extracted variables organized into columns

### `extract_and_organize(query: str, df: pd.DataFrame) -> Dict[str, Any]`

Complete extraction pipeline: parse intent, extract variables, and organize into table.

**Parameters:**
- `query` (str): User's query/question
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- Dictionary with:
  - `intent`: Parsed intent information
  - `extracted_data`: DataFrame with extracted variables
  - `summary`: Summary statistics

## Examples

### Example 1: Simple Variable Extraction

```python
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'NYC']
})

extractor = GeminiDataExtractor()
result = extractor.extract_and_organize(
    "Show me names and ages of people in NYC",
    df
)

print(result['extracted_data'])
# Output:
#    name  age
# 0  Alice   25
# 2  Charlie 35
```

### Example 2: Filtering and Aggregation

```python
df = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B', 'A'],
    'sales': [100, 150, 200, 120, 180],
    'region': ['North', 'South', 'North', 'South', 'North']
})

extractor = GeminiDataExtractor()
result = extractor.extract_and_organize(
    "Count total sales by product",
    df
)

print(result['extracted_data'])
# Output:
#   product  sales
# 0       A    480
# 1       B    270
```

### Example 3: Medical Data Extraction

```python
df = pd.DataFrame({
    'recordid': ['R001', 'R002', 'R003'],
    'right_hydronephrosis': ['present', 'absent', 'present'],
    'left_hydronephrosis': ['absent', 'present', 'present'],
    'imaging_date': ['2024-01-15', '2024-02-20', '2024-01-10']
})

extractor = GeminiDataExtractor()
result = extractor.extract_and_organize(
    "Extract all patients with hydronephrosis on either side",
    df
)

print(result['extracted_data'])
# Output: Filtered DataFrame with patients having hydronephrosis
```

## How It Works

1. **Schema Analysis**: The module first analyzes the dataset schema (column names, types, value ranges, etc.)

2. **Intent Parsing**: Gemini parses the user query to understand:
   - What variables are needed
   - What filters should be applied
   - Whether grouping or aggregation is needed

3. **Variable Extraction**: Based on the parsed intent, the module:
   - Filters the data if needed
   - Selects relevant columns
   - Applies grouping and aggregation if specified
   - Returns a new organized DataFrame

## Error Handling

The module includes retry logic for API calls and handles common errors:
- Invalid JSON responses (attempts to fix common JSON issues)
- Model availability (tries multiple model name variants)
- Missing columns (falls back to available columns)

## Limitations

- Derived columns (computed columns) are identified but not automatically computed
- Complex aggregations may require manual post-processing
- Large datasets may need chunking for better performance

## Integration

This module can be used independently or integrated with the main radiology agent:

```python
from src.gemini_data_extractor import GeminiDataExtractor
from src.orchestrator import RadiologyAgent

# Use with structured data from the main agent
agent = RadiologyAgent()
structured_df = agent.extract_structured_data(raw_df)

# Extract specific variables
extractor = GeminiDataExtractor()
result = extractor.extract_and_organize(
    "Show patients with stones larger than 1cm",
    structured_df
)
```

