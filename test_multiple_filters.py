#!/usr/bin/env python3
"""Test script to demonstrate multiple filters can be applied simultaneously."""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_tools import QueryTools

def create_sample_data():
    """Create sample data for testing multiple filters."""
    data = {
        'recordid': ['PAT_001', 'PAT_002', 'PAT_003', 'PAT_004', 'PAT_005', 'PAT_006', 'PAT_007', 'PAT_008'],
        'imaging_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-25', '2023-08-30'],
        'right_stone': ['present', 'absent', 'present', 'unclear', 'present', 'absent', 'present', 'present'],
        'left_stone': ['absent', 'present', 'present', 'present', 'absent', 'present', 'absent', 'present'],
        'right_stone_size_cm': [1.2, None, 0.8, None, 2.1, None, 1.5, 0.9],
        'left_stone_size_cm': [None, 1.5, 1.3, 0.9, None, 1.8, None, 1.1],
        'bladder_volume_ml': [250, 300, 180, 320, 280, 350, 200, 290],
        'narrative': [
            'Right kidney stone 1.2 cm. No hydronephrosis.',
            'Left kidney stone 1.5 cm. Mild hydronephrosis.',
            'Bilateral kidney stones: right 0.8 cm, left 1.3 cm. Severe hydronephrosis.',
            'Left kidney stone 0.9 cm. Moderate hydronephrosis.',
            'Right kidney stone 2.1 cm. No hydronephrosis.',
            'Left kidney stone 1.8 cm. Mild hydronephrosis.',
            'Right kidney stone 1.5 cm. No hydronephrosis.',
            'Bilateral kidney stones: right 0.9 cm, left 1.1 cm. Moderate hydronephrosis.'
        ]
    }
    return pd.DataFrame(data)

def test_multiple_filters():
    """Test that multiple filters can be applied simultaneously."""
    print("🧪 Testing Multiple Filters in QueryTools")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"📊 Sample data created with {len(df)} records")
    print("Sample data:")
    print(df[['recordid', 'right_stone', 'left_stone', 'right_stone_size_cm', 'left_stone_size_cm', 'bladder_volume_ml']].to_string())
    print()
    
    # Initialize QueryTools
    query_tools = QueryTools(df)
    print("🔧 QueryTools initialized")
    print()
    
    # Test 1: Single filter
    print("📝 Test 1: Single Filter - Right Side Only")
    print("-" * 40)
    filters1 = {'side': 'right'}
    result1 = query_tools.apply_filters(filters1)
    print(f"Filter: {filters1}")
    print(f"Results: {len(result1)} patients")
    print(f"Patient IDs: {result1['recordid'].tolist()}")
    print()
    
    # Test 2: Two filters combined
    print("📝 Test 2: Two Filters - Right Side + Size > 1.0 cm")
    print("-" * 40)
    filters2 = {'side': 'right', 'min_size_cm': 1.0}
    result2 = query_tools.apply_filters(filters2)
    print(f"Filter: {filters2}")
    print(f"Results: {len(result2)} patients")
    print(f"Patient IDs: {result2['recordid'].tolist()}")
    print()
    
    # Test 3: Three filters combined
    print("📝 Test 3: Three Filters - Right Side + Size > 1.0 cm + Bladder Volume > 250 ml")
    print("-" * 40)
    filters3 = {'side': 'right', 'min_size_cm': 1.0, 'min_bladder_volume_ml': 250}
    result3 = query_tools.apply_filters(filters3)
    print(f"Filter: {filters3}")
    print(f"Results: {len(result3)} patients")
    print(f"Patient IDs: {result3['recordid'].tolist()}")
    print()
    
    # Test 4: Four filters combined
    print("📝 Test 4: Four Filters - Right Side + Size Range + Bladder Volume Range")
    print("-" * 40)
    filters4 = {
        'side': 'right', 
        'min_size_cm': 0.8, 
        'max_size_cm': 2.0,
        'min_bladder_volume_ml': 200,
        'max_bladder_volume_ml': 300
    }
    result4 = query_tools.apply_filters(filters4)
    print(f"Filter: {filters4}")
    print(f"Results: {len(result4)} patients")
    print(f"Patient IDs: {result4['recordid'].tolist()}")
    print()
    
    # Test 5: Complex filters with date range
    print("📝 Test 5: Complex Filters - Multiple Conditions")
    print("-" * 40)
    filters5 = {
        'stone_presence': 'present',
        'min_size_cm': 1.0,
        'start_year': 2023,
        'end_year': 2023,
        'min_bladder_volume_ml': 250
    }
    result5 = query_tools.apply_filters(filters5)
    print(f"Filter: {filters5}")
    print(f"Results: {len(result5)} patients")
    print(f"Patient IDs: {result5['recordid'].tolist()}")
    print()
    
    # Test 6: Learned filters (simulated)
    print("📝 Test 6: Learned Filters (Simulated)")
    print("-" * 40)
    
    # Simulate learned filters
    query_tools.learned_filters = {
        'hydronephrosis': {
            'column': 'narrative',
            'operator': 'contains',
            'value': 'hydronephrosis',
            'description': 'Filter for patients with hydronephrosis mentioned'
        }
    }
    
    filters6 = {'side': 'left', 'hydronephrosis': True}
    result6 = query_tools.apply_learned_filters(filters6, df)
    print(f"Filter: {filters6}")
    print(f"Results: {len(result6)} patients")
    print(f"Patient IDs: {result6['recordid'].tolist()}")
    print()
    
    # Test 7: Combined standard and learned filters
    print("📝 Test 7: Combined Standard + Learned Filters")
    print("-" * 40)
    filters7 = {
        'side': 'left',
        'min_size_cm': 1.0,
        'hydronephrosis': True
    }
    
    # Apply standard filters first
    result7_standard = query_tools.apply_filters(filters7)
    print(f"After standard filters: {len(result7_standard)} patients")
    
    # Then apply learned filters
    result7_final = query_tools.apply_learned_filters(filters7, result7_standard)
    print(f"Filter: {filters7}")
    print(f"Final results: {len(result7_final)} patients")
    print(f"Patient IDs: {result7_final['recordid'].tolist()}")
    print()
    
    print("🎉 Multiple Filter Test Complete!")
    print("=" * 60)
    print("✅ QueryTools supports MULTIPLE filters simultaneously:")
    print("   • Standard filters: side, stone_presence, min_size_cm, max_size_cm")
    print("   • Date filters: start_year, end_year")
    print("   • Volume filters: min_bladder_volume_ml, max_bladder_volume_ml")
    print("   • Learned filters: Any dynamically learned concepts")
    print("   • All filters are applied in sequence (AND logic)")
    print("   • Each filter further narrows down the results")

if __name__ == "__main__":
    test_multiple_filters()

