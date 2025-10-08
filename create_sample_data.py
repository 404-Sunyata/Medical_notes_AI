#!/usr/bin/env python3
"""Create sample data for testing the radiology AI agent."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data(num_records: int = 50) -> pd.DataFrame:
    """Create sample radiology data for testing."""
    
    # Sample narratives with various scenarios
    sample_narratives = [
        "CT abdomen and pelvis shows bilateral kidney stones. Right kidney stone measures 1.2 cm. Left kidney stone measures 0.8 cm. Bladder volume is 250 ml. No hydronephrosis.",
        "Ultrasound reveals left kidney stone measuring 0.5 cm. Right kidney is normal. Bladder appears normal with volume of 180 ml.",
        "CT scan demonstrates right renal stone 2.1 cm in size. Left kidney is unremarkable. Bladder volume 300 ml. Patient has history of recurrent stones.",
        "No kidney stones identified on imaging. Both kidneys appear normal in size (right: 11 x 5.5 x 4 cm, left: 10.5 x 5 x 4 cm). Bladder volume 200 ml.",
        "Bilateral kidney stones present. Right stone 1.8 cm, left stone 1.5 cm. Moderate hydronephrosis on the right. Bladder volume 280 ml.",
        "Left kidney stone measuring 0.3 cm. Right kidney normal. Bladder volume 150 ml. No evidence of obstruction.",
        "CT shows multiple small stones in right kidney, largest measuring 0.7 cm. Left kidney clear. Bladder volume 220 ml.",
        "No stones identified. Kidneys normal size. Bladder volume 190 ml. Patient denies history of kidney stones.",
        "Right kidney stone 1.0 cm with mild hydronephrosis. Left kidney normal. Bladder volume 240 ml.",
        "Bilateral stones: right 1.4 cm, left 0.9 cm. Bladder volume 260 ml. Patient has family history of kidney stones.",
        "Left kidney stone 0.6 cm. Right kidney appears normal. Bladder volume 170 ml. No hydronephrosis.",
        "CT abdomen shows right kidney stone 1.7 cm. Left kidney normal. Bladder volume 290 ml. History of previous stone episodes.",
        "No kidney stones seen. Both kidneys normal size. Bladder volume 210 ml. Patient asymptomatic.",
        "Bilateral kidney stones: right 1.1 cm, left 1.3 cm. Bladder volume 270 ml. Mild bilateral hydronephrosis.",
        "Left kidney stone 0.4 cm. Right kidney normal. Bladder volume 160 ml. No obstruction noted.",
        "Right kidney stone 1.9 cm with moderate hydronephrosis. Left kidney clear. Bladder volume 310 ml.",
        "No stones identified. Kidneys normal. Bladder volume 185 ml. Patient has diabetes mellitus.",
        "Bilateral stones: right 0.8 cm, left 1.2 cm. Bladder volume 230 ml. Patient on stone prevention diet.",
        "Left kidney stone 1.6 cm. Right kidney normal. Bladder volume 250 ml. History of recurrent UTIs.",
        "CT shows right kidney stone 0.9 cm. Left kidney normal. Bladder volume 200 ml. No hydronephrosis."
    ]
    
    # Generate sample data
    data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(num_records):
        record_id = f"PAT_{i+1:04d}"
        
        # Random imaging date between 2020-2023
        days_offset = random.randint(0, 1460)  # 4 years
        imaging_date = base_date + timedelta(days=days_offset)
        
        # Random surgery date (some patients may not have surgery)
        if random.random() < 0.3:  # 30% have surgery
            surg_days_offset = random.randint(-30, 365)  # Surgery within 1 year of imaging
            surg_date = imaging_date + timedelta(days=surg_days_offset)
        else:
            surg_date = None
        
        # Random narrative
        narrative = random.choice(sample_narratives)
        
        # Add some variation to narratives
        if random.random() < 0.2:  # 20% chance of additional details
            additional_details = [
                " Patient reports flank pain.",
                " No acute findings.",
                " Follow-up recommended.",
                " Patient asymptomatic.",
                " Mild perinephric stranding noted."
            ]
            narrative += random.choice(additional_details)
        
        data.append({
            'recordid': record_id,
            'surg_date': surg_date,
            'imaging_date': imaging_date,
            'narrative': narrative
        })
    
    return pd.DataFrame(data)

def main():
    """Create and save sample data."""
    print("Creating sample radiology data...")
    
    # Create sample data
    df = create_sample_data(100)  # Create 100 records
    
    # Save to Excel
    output_file = "sample_radiology_data.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"Sample data created: {output_file}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['imaging_date'].min().date()} to {df['imaging_date'].max().date()}")
    print(f"Unique patients: {df['recordid'].nunique()}")
    
    # Show sample
    print("\nSample records:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()



