"""Run Gemini Data Extractor with sample_radiology_data.xlsx"""

import pandas as pd
import sys
import os
from datetime import datetime
from src.gemini_data_extractor import GeminiDataExtractor

def main():
    """Main function to run the Gemini data extractor."""
    
    # Check if file exists
    excel_file = "sample_radiology_data.xlsx"
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found in current directory.")
        print("Please make sure the file exists or provide the correct path.")
        sys.exit(1)
    
    print("="*70)
    print("Gemini Data Extractor - Sample Radiology Data")
    print("="*70)
    print(f"\nLoading data from: {excel_file}\n")
    
    # Load the Excel file
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        print("\n" + "="*70 + "\n")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit(1)
    
    # Initialize the extractor
    try:
        print("Initializing Gemini Data Extractor...")
        extractor = GeminiDataExtractor()
        print("✓ Extractor initialized successfully\n")
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        print("Make sure GEMINI_API_KEY is set in your environment or .env file")
        sys.exit(1)
    
    # Example queries
    queries = [
        {
            "name": "Extract Patient IDs and Imaging Dates",
            "query": "Extract patient record IDs and their imaging dates"
        },
        {
            "name": "Find Patients with Stones",
            "query": "Extract all patients who have kidney stones, showing their record ID, imaging date, and stone information"
        },
        {
            "name": "Extract Hydronephrosis Patients",
            "query": "Show me all patients with hydronephrosis, including their record ID, imaging date, and which side has hydronephrosis"
        },
        {
            "name": "Extract Stone Size Information",
            "query": "Extract patient IDs, imaging dates, and stone sizes for patients with stones larger than 1 cm"
        }
    ]
    
    # Run queries
    for i, query_info in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query_info['name']}")
        print(f"{'='*70}")
        print(f"Query: {query_info['query']}\n")
        
        try:
            result = extractor.extract_and_organize(query_info['query'], df)
            
            print(f"Intent: {result['intent']['intent']}")
            print(f"\nVariables of Interest: {result['intent'].get('variables_of_interest', [])}")
            print(f"\nExtracted Data ({len(result['extracted_data'])} rows):")
            print(result['extracted_data'])
            print(f"\nSummary:")
            print(f"  - Original rows: {result['summary']['original_rows']}")
            print(f"  - Extracted rows: {result['summary']['extracted_rows']}")
            print(f"  - Original columns: {result['summary']['original_columns']}")
            print(f"  - Extracted columns: {result['summary']['extracted_columns']}")
            print(f"  - Variables extracted: {result['summary']['variables_extracted']}")
            
            # Export to CSV
            output_dir = "out"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = query_info['name'].lower().replace(' ', '_')
            filename = f"extracted_{safe_name}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            result['extracted_data'].to_csv(filepath, index=False)
            print(f"\n✓ Saved to: {filepath}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*70)
    
    # Interactive mode
    print("\n" + "="*70)
    print("Interactive Mode")
    print("="*70)
    print("Enter your own queries (type 'quit' or 'exit' to stop)\n")
    
    while True:
        try:
            user_query = input("Enter query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not user_query:
                continue
            
            print(f"\nProcessing: {user_query}\n")
            result = extractor.extract_and_organize(user_query, df)
            
            print(f"Intent: {result['intent']['intent']}")
            print(f"\nExtracted Data:")
            print(result['extracted_data'])
            print(f"\nSummary: {result['summary']}")
            
            # Export to CSV
            output_dir = "out"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_interactive_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            result['extracted_data'].to_csv(filepath, index=False)
            print(f"\n✓ Saved to: {filepath}")
            print("\n" + "-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            continue


if __name__ == "__main__":
    main()

