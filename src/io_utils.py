"""I/O utilities for loading and processing Excel data."""

import pandas as pd
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_excel_data(file_path: str) -> pd.DataFrame:
    """
    Load Excel data with required columns validation.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        DataFrame with validated columns
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    # Validate required columns
    required_columns = ['recordid', 'surg_date', 'imaging_date', 'narrative']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean and validate data
    df = clean_dataframe(df)
    
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert date columns
    for date_col in ['surg_date', 'imaging_date']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Ensure recordid is string
    df['recordid'] = df['recordid'].astype(str)
    
    # Clean narrative text
    if 'narrative' in df.columns:
        df['narrative'] = df['narrative'].fillna('').astype(str)
        # Remove extra whitespace
        df['narrative'] = df['narrative'].str.strip()
    
    logger.info(f"Cleaned DataFrame: {len(df)} rows remaining")
    return df

def concat_narratives_by_imaging(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate multiple narratives for the same (recordid, imaging_date) combination.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with concatenated narratives
    """
    # Group by recordid and imaging_date, concatenate narratives
    grouped = df.groupby(['recordid', 'imaging_date']).agg({
        'narrative': lambda x: ' '.join(x.astype(str)),
        'surg_date': 'first'  # Keep first surgery date
    }).reset_index()
    
    # Clean concatenated narratives
    grouped['narrative'] = grouped['narrative'].str.replace(r'\s+', ' ', regex=True)
    grouped['narrative'] = grouped['narrative'].str.strip()
    
    logger.info(f"Concatenated narratives: {len(grouped)} unique (recordid, imaging_date) combinations")
    return grouped

def save_structured_data(df: pd.DataFrame, filename: str, format: str = 'parquet') -> str:
    """
    Save structured data to output directory.
    
    Args:
        df: DataFrame to save
        filename: Base filename (without extension)
        format: Output format ('parquet', 'csv', 'excel')
        
    Returns:
        Full path to saved file
    """
    from .config import OUTPUT_DIR
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if format == 'parquet':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.parquet")
        df.to_parquet(file_path, index=False)
    elif format == 'csv':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
        df.to_csv(file_path, index=False)
    elif format == 'excel':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {len(df)} rows to {file_path}")
    return file_path

def load_structured_data(filename: str, format: str = 'parquet') -> pd.DataFrame:
    """
    Load previously saved structured data.
    
    Args:
        filename: Base filename (without extension)
        format: File format ('parquet', 'csv', 'excel')
        
    Returns:
        Loaded DataFrame
    """
    from .config import OUTPUT_DIR
    
    if format == 'parquet':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.parquet")
        df = pd.read_parquet(file_path)
    elif format == 'csv':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
        df = pd.read_csv(file_path)
    elif format == 'excel':
        file_path = os.path.join(OUTPUT_DIR, f"{filename}.xlsx")
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_rows': len(df),
        'unique_patients': df['recordid'].nunique() if 'recordid' in df.columns else 0,
        'date_range': None,
        'narrative_stats': {}
    }
    
    # Date range
    date_cols = ['surg_date', 'imaging_date']
    for col in date_cols:
        if col in df.columns and df[col].notna().any():
            summary['date_range'] = {
                'min': df[col].min(),
                'max': df[col].max()
            }
            break
    
    # Narrative statistics
    if 'narrative' in df.columns:
        narrative_lengths = df['narrative'].str.len()
        summary['narrative_stats'] = {
            'mean_length': narrative_lengths.mean(),
            'median_length': narrative_lengths.median(),
            'min_length': narrative_lengths.min(),
            'max_length': narrative_lengths.max(),
            'empty_narratives': (df['narrative'] == '').sum()
        }
    
    return summary



