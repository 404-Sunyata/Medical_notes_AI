"""Example usage of GeminiDataExtractor for data extraction."""

import pandas as pd
from src.gemini_data_extractor import GeminiDataExtractor

def example_basic_extraction():
    """Basic example of extracting variables from a dataset."""
    
    # Create sample dataset
    data = {
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'age': [45, 32, 67, 28, 55],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'diagnosis': ['Hypertension', 'Diabetes', 'Hypertension', 'Asthma', 'Diabetes'],
        'blood_pressure_systolic': [140, 120, 150, 110, 135],
        'blood_pressure_diastolic': [90, 80, 95, 70, 85],
        'medication': ['Lisinopril', 'Metformin', 'Amlodipine', 'Albuterol', 'Lisinopril']
    }
    df = pd.DataFrame(data)
    
    print("Original Dataset:")
    print(df)
    print("\n" + "="*70 + "\n")
    
    # Initialize extractor
    extractor = GeminiDataExtractor()
    
    # Example query 1: Extract specific variables
    query1 = "Extract patient ID, age, and diagnosis for all patients"
    print(f"Query 1: {query1}")
    result1 = extractor.extract_and_organize(query1, df)
    print(f"\nIntent: {result1['intent']['intent']}")
    print(f"\nExtracted Data:")
    print(result1['extracted_data'])
    print(f"\nSummary: {result1['summary']}")
    print("\n" + "="*70 + "\n")
    
    # Example query 2: Filter and extract
    query2 = "Show me all patients with hypertension, including their age and blood pressure"
    print(f"Query 2: {query2}")
    result2 = extractor.extract_and_organize(query2, df)
    print(f"\nIntent: {result2['intent']['intent']}")
    print(f"\nExtracted Data:")
    print(result2['extracted_data'])
    print(f"\nSummary: {result2['summary']}")
    print("\n" + "="*70 + "\n")
    
    # Example query 3: Aggregation
    query3 = "Count patients by diagnosis"
    print(f"Query 3: {query3}")
    result3 = extractor.extract_and_organize(query3, df)
    print(f"\nIntent: {result3['intent']['intent']}")
    print(f"\nExtracted Data:")
    print(result3['extracted_data'])
    print(f"\nSummary: {result3['summary']}")


def example_medical_data():
    """Example with medical/radiology data."""
    
    # Sample radiology data
    data = {
        'recordid': ['R001', 'R002', 'R003', 'R004', 'R005'],
        'imaging_date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-03-05', '2024-02-28'],
        'patient_age': [45, 32, 67, 28, 55],
        'right_stone': ['present', 'absent', 'present', 'absent', 'present'],
        'left_stone': ['absent', 'present', 'present', 'absent', 'absent'],
        'right_stone_size_cm': [1.2, None, 0.8, None, 2.1],
        'left_stone_size_cm': [None, 1.5, 1.3, None, None],
        'right_hydronephrosis': ['present', 'absent', 'present', 'absent', 'absent'],
        'left_hydronephrosis': ['absent', 'present', 'present', 'absent', 'absent']
    }
    df = pd.DataFrame(data)
    df['imaging_date'] = pd.to_datetime(df['imaging_date'])
    
    print("Medical Dataset:")
    print(df)
    print("\n" + "="*70 + "\n")
    
    extractor = GeminiDataExtractor()
    
    # Query: Extract hydronephrosis patients
    query = "Extract all patients with hydronephrosis, showing their record ID, imaging date, and which side has hydronephrosis"
    print(f"Query: {query}")
    result = extractor.extract_and_organize(query, df)
    print(f"\nIntent: {result['intent']['intent']}")
    print(f"\nExtracted Data:")
    print(result['extracted_data'])
    print(f"\nSummary: {result['summary']}")


if __name__ == "__main__":
    print("="*70)
    print("Gemini Data Extractor - Example Usage")
    print("="*70)
    print("\n")
    
    try:
        # Run basic example
        example_basic_extraction()
        
        print("\n\n" + "="*70)
        print("Medical Data Example")
        print("="*70 + "\n")
        
        # Run medical example
        example_medical_data()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

