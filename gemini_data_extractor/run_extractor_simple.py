"""Simple script to run Gemini Data Extractor with sample_radiology_data.xlsx"""

import pandas as pd
import os
from datetime import datetime
from src.gemini_data_extractor import GeminiDataExtractor

# Load the Excel file
print("Loading sample_radiology_data.xlsx...")
df = pd.read_excel("sample_radiology_data.xlsx")
LIMIT_ROWS = 20
if LIMIT_ROWS:
    df = df.head(LIMIT_ROWS)
    print(f"Loaded {len(df)} rows (limited to first {LIMIT_ROWS} rows)\n")
else:
    print(f"Loaded {len(df)} rows\n")

# Initialize extractor
print("Initializing Gemini Data Extractor...")
extractor = GeminiDataExtractor()
print("Ready!\n")

# Example query
query = "Which patients have the left kidney stone? Extract this information from the narrative text column."

print(f"Query: {query}\n")
print("Processing...\n")

# Extract data
result = extractor.extract_and_organize(query, df)

# Display results
print("="*70)
print("RESULTS")
print("="*70)
print(f"\nIntent: {result['intent']['intent']}")
print(f"\nExtracted Data:")
print(result['extracted_data'])
print(f"\nSummary:")
print(f"  Rows extracted: {result['summary']['extracted_rows']}")
print(f"  Columns extracted: {result['summary']['extracted_columns']}")

# Export to CSV
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"extracted_data_{timestamp}.csv"
filepath = os.path.join(output_dir, filename)

# Save to CSV
result['extracted_data'].to_csv(filepath, index=False)
print(f"\n✓ Extracted data saved to: {filepath}")

