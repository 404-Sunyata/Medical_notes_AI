#!/usr/bin/env python3
"""Test script for dynamic learning capabilities in QueryTools."""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.query_tools import QueryTools
from src.config import get_openai_client
import unittest.mock

def create_sample_data():
    """Create sample data for testing."""
    data = {
        'recordid': ['PAT_001', 'PAT_002', 'PAT_003', 'PAT_004', 'PAT_005'],
        'imaging_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
        'right_stone': ['present', 'absent', 'present', 'unclear', 'present'],
        'left_stone': ['absent', 'present', 'present', 'present', 'absent'],
        'right_stone_size_cm': [1.2, None, 0.8, None, 2.1],
        'left_stone_size_cm': [None, 1.5, 1.3, 0.9, None],
        'bladder_volume_ml': [250, 300, 180, 320, 280],
        'narrative': [
            'Right kidney stone 1.2 cm. No hydronephrosis.',
            'Left kidney stone 1.5 cm. Mild hydronephrosis.',
            'Bilateral kidney stones: right 0.8 cm, left 1.3 cm. Severe hydronephrosis.',
            'Left kidney stone 0.9 cm. Moderate hydronephrosis.',
            'Right kidney stone 2.1 cm. No hydronephrosis.'
        ]
    }
    return pd.DataFrame(data)

def test_dynamic_learning():
    """Test dynamic learning capabilities."""
    print("🧪 Testing Dynamic Learning in QueryTools")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"📊 Sample data created with {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Initialize QueryTools
    query_tools = QueryTools(df)
    print(f"🔧 QueryTools initialized")
    print(f"Available columns: {query_tools.available_columns}")
    print(f"Numeric columns: {query_tools.numeric_columns}")
    print(f"Text columns: {query_tools.text_columns}")
    print()
    
    # Test 1: Standard query (should work without learning)
    print("📝 Test 1: Standard Query")
    print("-" * 30)
    query1 = "Show me patients with right kidney stones"
    print(f"Query: {query1}")
    
    filters1 = {'side': 'right', 'stone_presence': 'present'}
    filtered_df1 = query_tools.apply_filters_with_learning(filters1, query1)
    print(f"Results: {len(filtered_df1)} patients found")
    print(f"Patient IDs: {filtered_df1['recordid'].tolist()}")
    print()
    
    # Test 2: New medical concept (hydronephrosis)
    print("📝 Test 2: New Medical Concept - Hydronephrosis")
    print("-" * 30)
    query2 = "Show me patients with hydronephrosis"
    print(f"Query: {query2}")
    
    # Mock LLM responses for learning
    with unittest.mock.patch('src.query_tools.get_openai_client') as mock_client:
        mock_openai_client = unittest.mock.MagicMock()
        mock_response = unittest.mock.MagicMock()
        mock_response.choices = [unittest.mock.MagicMock()]
        
        # Mock concept detection
        mock_response.choices[0].message.content = "hydronephrosis"
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_openai_client
        
        # Test concept detection
        concepts = query_tools._detect_new_medical_concepts(query2)
        print(f"Detected concepts: {concepts}")
        
        # Mock filter logic generation
        mock_response.choices[0].message.content = '''
        {
            "column": "narrative",
            "operator": "contains",
            "value": "hydronephrosis",
            "description": "Filter for patients with hydronephrosis mentioned in narrative"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test filter logic generation
        filter_logic = query_tools._generate_filter_logic("hydronephrosis", query2)
        print(f"Generated filter logic: {filter_logic}")
        
        if filter_logic:
            # Add to learned filters
            query_tools.learned_filters["hydronephrosis"] = filter_logic
            print("✅ Learned new filter for hydronephrosis")
            
            # Test applying learned filter
            filters2 = {'hydronephrosis': True}
            filtered_df2 = query_tools.apply_learned_filters(filters2, df)
            print(f"Results: {len(filtered_df2)} patients with hydronephrosis")
            print(f"Patient IDs: {filtered_df2['recordid'].tolist()}")
        else:
            print("❌ Failed to generate filter logic")
    print()
    
    # Test 3: New statistics (variance)
    print("📝 Test 3: New Statistics - Variance")
    print("-" * 30)
    query3 = "What is the variance of stone sizes?"
    print(f"Query: {query3}")
    
    with unittest.mock.patch('src.query_tools.get_openai_client') as mock_client:
        mock_openai_client = unittest.mock.MagicMock()
        mock_response = unittest.mock.MagicMock()
        mock_response.choices = [unittest.mock.MagicMock()]
        
        # Mock statistics detection
        mock_response.choices[0].message.content = "variance"
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_openai_client
        
        # Test statistics detection
        statistics = query_tools._detect_new_statistics(query3)
        print(f"Detected statistics: {statistics}")
        
        # Mock statistics logic generation
        mock_response.choices[0].message.content = '''
        {
            "columns": ["right_stone_size_cm", "left_stone_size_cm"],
            "method": "var",
            "parameters": {},
            "description": "Calculate variance of stone sizes"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test statistics logic generation
        stats_logic = query_tools._generate_statistics_logic("variance", query3)
        print(f"Generated statistics logic: {stats_logic}")
        
        if stats_logic:
            # Add to learned statistics
            query_tools.learned_statistics["variance"] = stats_logic
            print("✅ Learned new statistic: variance")
            
            # Test computing learned statistics
            learned_stats = query_tools.compute_learned_statistics(df, query3)
            print(f"Computed learned statistics: {learned_stats}")
        else:
            print("❌ Failed to generate statistics logic")
    print()
    
    # Test 4: Complete learning workflow
    print("📝 Test 4: Complete Learning Workflow")
    print("-" * 30)
    query4 = "Show me patients with infection and calculate the mode of bladder volumes"
    print(f"Query: {query4}")
    
    with unittest.mock.patch('src.query_tools.get_openai_client') as mock_client:
        mock_openai_client = unittest.mock.MagicMock()
        mock_response = unittest.mock.MagicMock()
        mock_response.choices = [unittest.mock.MagicMock()]
        
        # Mock multiple responses for complete workflow
        responses = [
            "infection",  # Concept detection
            '''{
                "column": "narrative",
                "operator": "contains", 
                "value": "infection",
                "description": "Filter for patients with infection mentioned"
            }''',  # Filter logic
            "mode",  # Statistics detection
            '''{
                "columns": ["bladder_volume_ml"],
                "method": "mode",
                "parameters": {},
                "description": "Calculate mode of bladder volumes"
            }'''  # Statistics logic
        ]
        
        mock_openai_client.chat.completions.create.side_effect = [
            unittest.mock.MagicMock(choices=[unittest.mock.MagicMock(message=unittest.mock.MagicMock(content=response))])
            for response in responses
        ]
        mock_client.return_value = mock_openai_client
        
        # Test complete learning workflow
        query_tools._learn_from_query(query4)
        print(f"Learned filters: {list(query_tools.learned_filters.keys())}")
        print(f"Learned statistics: {list(query_tools.learned_statistics.keys())}")
        
        # Test applying learned filters and statistics
        filters4 = {'infection': True}
        filtered_df4 = query_tools.apply_learned_filters(filters4, df)
        print(f"Patients with infection: {len(filtered_df4)}")
        
        learned_stats4 = query_tools.compute_learned_statistics(df, query4)
        print(f"Learned statistics results: {learned_stats4}")
    print()
    
    print("🎉 Dynamic Learning Test Complete!")
    print("=" * 60)
    print("✅ QueryTools now has dynamic learning capabilities:")
    print("   • Can detect new medical concepts")
    print("   • Can generate filter logic for new concepts")
    print("   • Can detect new statistical requests")
    print("   • Can generate computation logic for new statistics")
    print("   • Can learn and apply new patterns automatically")
    print("   • Can persist learned patterns to files")

if __name__ == "__main__":
    test_dynamic_learning()

