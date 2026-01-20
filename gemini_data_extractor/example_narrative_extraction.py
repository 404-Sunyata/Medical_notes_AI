"""Example of narrative text extraction with Gemini Data Extractor"""

import pandas as pd
import os
from datetime import datetime
from src.gemini_data_extractor import GeminiDataExtractor

def main():
    """Demonstrate narrative text extraction."""
    
    # Load the Excel file
    print("Loading sample_radiology_data.xlsx...")
    df = pd.read_excel("sample_radiology_data.xlsx")
    print(f"Loaded {len(df)} rows\n")
    
    # Show sample narrative
    if 'narrative' in df.columns:
        print("Sample narrative text:")
        print(df['narrative'].iloc[0][:200] + "...")
        print("\n" + "="*70 + "\n")
    
    # Initialize extractor
    print("Initializing Gemini Data Extractor...")
    extractor = GeminiDataExtractor()
    print("Ready!\n")
    
    # Example queries that require narrative extraction
    queries = [
        {
            "name": "Extract Hydronephrosis from Narrative",
            "query": "Extract patient IDs, imaging dates, and whether they have hydronephrosis. Extract hydronephrosis information from the narrative text column."
        },
        {
            "name": "Extract Stone Information from Narrative",
            "query": "Extract patient IDs and kidney stone information (presence and size) from the narrative text"
        },
        {
            "name": "Extract Multiple Findings",
            "query": "Extract patient IDs, imaging dates, stone presence, stone size, and hydronephrosis status from narrative text"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {query_info['name']}")
        print(f"{'='*70}")
        print(f"Query: {query_info['query']}\n")
        
        try:
            result = extractor.extract_and_organize(query_info['query'], df)
            
            print(f"Intent: {result['intent']['intent']}")
            print(f"\nExtracted Data ({len(result['extracted_data'])} rows):")
            print(result['extracted_data'].head(10))  # Show first 10 rows
            print(f"\nColumns: {list(result['extracted_data'].columns)}")
            print(f"\nSummary:")
            print(f"  - Original rows: {result['summary']['original_rows']}")
            print(f"  - Extracted rows: {result['summary']['extracted_rows']}")
            print(f"  - Variables extracted: {result['summary']['variables_extracted']}")
            
            # Export to CSV
            output_dir = "out"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = query_info['name'].lower().replace(' ', '_')
            filename = f"narrative_extracted_{safe_name}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            result['extracted_data'].to_csv(filepath, index=False)
            print(f"\n✓ Saved to: {filepath}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*70)
        
        # Limit to first example for demo (remove this to run all)
        if i == 1:
            print("\n(Stopping after first example. Remove the break to run all examples)")
            break


if __name__ == "__main__":
    main()

