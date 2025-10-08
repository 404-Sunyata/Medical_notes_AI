"""Query tools for filtering and analyzing structured data."""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging

from .llm_schema import StructuredOutput, PlanSummary

logger = logging.getLogger(__name__)

class QueryTools:
    """Tools for querying and filtering structured radiology data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with structured DataFrame.
        
        Args:
            df: DataFrame with structured radiology data
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for querying."""
        # Ensure date columns are datetime
        if 'imaging_date' in self.df.columns:
            self.df['imaging_date'] = pd.to_datetime(self.df['imaging_date'], errors='coerce')
        
        # Convert size columns to numeric
        size_columns = ['right_stone_size_cm', 'left_stone_size_cm', 'bladder_volume_ml']
        for col in size_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create helper columns for easier querying
        self.df['has_right_stone'] = self.df['right_stone'] == 'present'
        self.df['has_left_stone'] = self.df['left_stone'] == 'present'
        self.df['has_any_stone'] = (self.df['has_right_stone'] | self.df['has_left_stone'])
        self.df['has_bilateral_stones'] = (self.df['has_right_stone'] & self.df['has_left_stone'])
        
        # Extract year from imaging_date
        if 'imaging_date' in self.df.columns:
            self.df['imaging_year'] = self.df['imaging_date'].dt.year
    
    def apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.
        
        Args:
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        # Apply side filter
        if 'side' in filters:
            side = filters['side'].lower()
            if side == 'left':
                filtered_df = filtered_df[filtered_df['has_left_stone']]
            elif side == 'right':
                filtered_df = filtered_df[filtered_df['has_right_stone']]
            elif side == 'bilateral':
                filtered_df = filtered_df[filtered_df['has_bilateral_stones']]
        
        # Apply stone presence filter
        if 'stone_presence' in filters:
            presence = filters['stone_presence'].lower()
            if presence == 'present':
                filtered_df = filtered_df[filtered_df['has_any_stone']]
            elif presence == 'absent':
                filtered_df = filtered_df[~filtered_df['has_any_stone']]
            elif presence == 'unclear':
                filtered_df = filtered_df[
                    (filtered_df['right_stone'] == 'unclear') | 
                    (filtered_df['left_stone'] == 'unclear')
                ]
        
        # Apply size filters
        if 'min_size_cm' in filters:
            min_size = filters['min_size_cm']
            size_condition = (
                (filtered_df['right_stone_size_cm'] >= min_size) |
                (filtered_df['left_stone_size_cm'] >= min_size)
            )
            filtered_df = filtered_df[size_condition]
        
        if 'max_size_cm' in filters:
            max_size = filters['max_size_cm']
            size_condition = (
                (filtered_df['right_stone_size_cm'] <= max_size) |
                (filtered_df['left_stone_size_cm'] <= max_size)
            )
            filtered_df = filtered_df[size_condition]
        
        # Apply date filters
        if 'start_year' in filters:
            start_year = filters['start_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] >= start_year]
        
        if 'end_year' in filters:
            end_year = filters['end_year']
            filtered_df = filtered_df[filtered_df['imaging_year'] <= end_year]
        
        # Apply bladder volume filters
        if 'min_bladder_volume_ml' in filters:
            min_volume = filters['min_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] >= min_volume]
        
        if 'max_bladder_volume_ml' in filters:
            max_volume = filters['max_bladder_volume_ml']
            filtered_df = filtered_df[filtered_df['bladder_volume_ml'] <= max_volume]
        
        # Add matched_reason column
        filtered_df = self._add_matched_reason(filtered_df, filters)
        
        return filtered_df
    
    def _add_matched_reason(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Add matched_reason column explaining why rows matched filters."""
        reasons = []
        
        for _, row in df.iterrows():
            reason_parts = []
            
            # Side reason
            if 'side' in filters:
                side = filters['side']
                if side == 'left' and row['has_left_stone']:
                    reason_parts.append(f"left side stone present")
                elif side == 'right' and row['has_right_stone']:
                    reason_parts.append(f"right side stone present")
                elif side == 'bilateral' and row['has_bilateral_stones']:
                    reason_parts.append(f"bilateral stones present")
            
            # Size reason
            if 'min_size_cm' in filters:
                min_size = filters['min_size_cm']
                if row['right_stone_size_cm'] and row['right_stone_size_cm'] >= min_size:
                    reason_parts.append(f"right stone ≥{min_size}cm")
                if row['left_stone_size_cm'] and row['left_stone_size_cm'] >= min_size:
                    reason_parts.append(f"left stone ≥{min_size}cm")
            
            if 'max_size_cm' in filters:
                max_size = filters['max_size_cm']
                if row['right_stone_size_cm'] and row['right_stone_size_cm'] <= max_size:
                    reason_parts.append(f"right stone ≤{max_size}cm")
                if row['left_stone_size_cm'] and row['left_stone_size_cm'] <= max_size:
                    reason_parts.append(f"left stone ≤{max_size}cm")
            
            # Date reason
            if 'start_year' in filters or 'end_year' in filters:
                year = row['imaging_year']
                if pd.notna(year):
                    reason_parts.append(f"imaging year {int(year)}")
            
            # Bladder reason
            if 'min_bladder_volume_ml' in filters:
                volume = row['bladder_volume_ml']
                if pd.notna(volume):
                    reason_parts.append(f"bladder volume ≥{filters['min_bladder_volume_ml']}ml")
            
            if 'max_bladder_volume_ml' in filters:
                volume = row['bladder_volume_ml']
                if pd.notna(volume):
                    reason_parts.append(f"bladder volume ≤{filters['max_bladder_volume_ml']}ml")
            
            if reason_parts:
                reasons.append("; ".join(reason_parts))
            else:
                reasons.append("matches general criteria")
        
        df['matched_reason'] = reasons
        return df
    
    def count_patients_with_stone(self, side: Optional[str] = None, 
                                min_size_cm: Optional[float] = None) -> int:
        """
        Count patients with stones matching criteria.
        
        Args:
            side: 'left', 'right', or None for any side
            min_size_cm: Minimum stone size in cm
            
        Returns:
            Number of patients matching criteria
        """
        if side == 'left':
            condition = self.df['has_left_stone']
            if min_size_cm:
                condition = condition & (self.df['left_stone_size_cm'] >= min_size_cm)
        elif side == 'right':
            condition = self.df['has_right_stone']
            if min_size_cm:
                condition = condition & (self.df['right_stone_size_cm'] >= min_size_cm)
        else:
            condition = self.df['has_any_stone']
            if min_size_cm:
                condition = condition & (
                    (self.df['right_stone_size_cm'] >= min_size_cm) |
                    (self.df['left_stone_size_cm'] >= min_size_cm)
                )
        
        return condition.sum()
    
    def get_stone_size_distribution(self, side: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stone size distribution statistics.
        
        Args:
            side: 'left', 'right', or None for both sides
            
        Returns:
            Dictionary with size statistics
        """
        if side == 'left':
            sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
        elif side == 'right':
            sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
        else:
            # Combine both sides
            left_sizes = self.df[self.df['has_left_stone']]['left_stone_size_cm'].dropna()
            right_sizes = self.df[self.df['has_right_stone']]['right_stone_size_cm'].dropna()
            sizes = pd.concat([left_sizes, right_sizes])
        
        if len(sizes) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }
        
        return {
            'count': len(sizes),
            'mean': float(sizes.mean()),
            'median': float(sizes.median()),
            'min': float(sizes.min()),
            'max': float(sizes.max()),
            'std': float(sizes.std())
        }
    
    def get_side_distribution(self) -> Dict[str, int]:
        """
        Get distribution of stones by side.
        
        Returns:
            Dictionary with counts by side
        """
        return {
            'left_only': (self.df['has_left_stone'] & ~self.df['has_right_stone']).sum(),
            'right_only': (self.df['has_right_stone'] & ~self.df['has_left_stone']).sum(),
            'bilateral': self.df['has_bilateral_stones'].sum(),
            'no_stones': (~self.df['has_any_stone']).sum()
        }
    
    def get_bladder_volume_stats(self) -> Dict[str, Any]:
        """
        Get bladder volume statistics.
        
        Returns:
            Dictionary with bladder volume statistics
        """
        volumes = self.df['bladder_volume_ml'].dropna()
        
        if len(volumes) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None
            }
        
        return {
            'count': len(volumes),
            'mean': float(volumes.mean()),
            'median': float(volumes.median()),
            'min': float(volumes.min()),
            'max': float(volumes.max())
        }
    
    def get_temporal_distribution(self) -> Dict[str, int]:
        """
        Get distribution of imaging by year.
        
        Returns:
            Dictionary with counts by year
        """
        if 'imaging_year' not in self.df.columns:
            return {}
        
        year_counts = self.df['imaging_year'].value_counts().sort_index()
        return {str(year): count for year, count in year_counts.items() if pd.notna(year)}
    
    def estimate_matching_rows(self, filters: Dict[str, Any]) -> int:
        """
        Estimate number of rows that would match given filters.
        
        Args:
            filters: Dictionary of filters
            
        Returns:
            Estimated number of matching rows
        """
        try:
            filtered_df = self.apply_filters(filters)
            return len(filtered_df)
        except Exception as e:
            logger.warning(f"Error estimating matching rows: {e}")
            return 0
    
    def get_summary_stats(self, filtered_df: Optional[pd.DataFrame] = None, 
                         relevant_fields: Optional[List[str]] = None,
                         query_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset, focusing only on relevant fields.
        
        Args:
            filtered_df: Optional filtered DataFrame, uses full dataset if None
            relevant_fields: List of fields that are relevant to the query
            query_text: Original query text to determine specific statistics needed
            
        Returns:
            Dictionary with summary statistics
        """
        if filtered_df is None:
            filtered_df = self.df
        
        stats = {
            'total_records': len(filtered_df),
            'unique_patients': filtered_df['recordid'].nunique() if 'recordid' in filtered_df.columns else 0
        }
        
        # Only compute stats for relevant fields
        if relevant_fields is None:
            relevant_fields = []
        
        if any('stone' in field.lower() for field in relevant_fields):
            stats['stone_distribution'] = self.get_side_distribution()
            stats['stone_size_stats'] = self.get_stone_size_distribution()
        
        if any('bladder' in field.lower() or 'volume' in field.lower() for field in relevant_fields):
            stats['bladder_volume_stats'] = self.get_bladder_volume_stats()
        
        if any('date' in field.lower() or 'year' in field.lower() or 'time' in field.lower() for field in relevant_fields):
            stats['temporal_distribution'] = self.get_temporal_distribution()
        
        # Add specific statistics based on query text
        if query_text:
            stats.update(self._get_specific_statistics(filtered_df, query_text))
        
        return stats
    
    def _get_specific_statistics(self, filtered_df: pd.DataFrame, query_text: str) -> Dict[str, Any]:
        """
        Get specific statistics based on query text.
        
        Args:
            filtered_df: Filtered DataFrame
            query_text: Original query text
            
        Returns:
            Dictionary with specific statistics
        """
        specific_stats = {}
        query_lower = query_text.lower()
        
        # Get stone size data once for reuse
        left_sizes = filtered_df[filtered_df['has_left_stone']]['left_stone_size_cm'].dropna()
        right_sizes = filtered_df[filtered_df['has_right_stone']]['right_stone_size_cm'].dropna()
        all_sizes = pd.concat([left_sizes, right_sizes]) if len(left_sizes) > 0 or len(right_sizes) > 0 else pd.Series(dtype=float)
        
        # Get bladder volume data once for reuse
        volumes = filtered_df['bladder_volume_ml'].dropna()
        
        # Check for multiple statistics in the same query (use if statements instead of elif)
        
        # Mean bladder volume
        if 'mean' in query_lower and 'bladder' in query_lower and 'volume' in query_lower:
            if len(volumes) > 0:
                specific_stats['mean_bladder_volume_ml'] = float(volumes.mean())
                specific_stats['bladder_volume_count'] = len(volumes)
        
        # Mean stone size
        if 'mean' in query_lower and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['mean_stone_size_cm'] = float(all_sizes.mean())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Maximum stone size (biggest, largest, max)
        # Handle both "biggest stone size" and "biggest stone" queries
        if any(word in query_lower for word in ['biggest', 'largest', 'maximum', 'max']) and 'stone' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['max_stone_size_cm'] = float(all_sizes.max())
                specific_stats['stone_size_count'] = len(all_sizes)
                
                # If asking "which patient", include patient details
                if any(word in query_lower for word in ['which', 'who', 'what patient', 'which patient']):
                    max_size = all_sizes.max()
                    # Find the patient(s) with the max stone size
                    max_patients = filtered_df[
                        (filtered_df['right_stone_size_cm'] == max_size) | 
                        (filtered_df['left_stone_size_cm'] == max_size)
                    ]
                    specific_stats['patients_with_max_stone'] = max_patients[['recordid', 'imaging_date', 'right_stone_size_cm', 'left_stone_size_cm']].to_dict('records')
        
        # Minimum stone size (smallest, min)
        # Handle both "smallest stone size" and "smallest stone" queries
        if any(word in query_lower for word in ['smallest', 'minimum', 'min']) and 'stone' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['min_stone_size_cm'] = float(all_sizes.min())
                specific_stats['stone_size_count'] = len(all_sizes)
                
                # If asking "which patient", include patient details
                if any(word in query_lower for word in ['which', 'who', 'what patient', 'which patient']):
                    min_size = all_sizes.min()
                    # Find the patient(s) with the min stone size
                    min_patients = filtered_df[
                        (filtered_df['right_stone_size_cm'] == min_size) | 
                        (filtered_df['left_stone_size_cm'] == min_size)
                    ]
                    specific_stats['patients_with_min_stone'] = min_patients[['recordid', 'imaging_date', 'right_stone_size_cm', 'left_stone_size_cm']].to_dict('records')
        
        # Sum stone size (total, sum)
        if any(word in query_lower for word in ['sum', 'total']) and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['sum_stone_size_cm'] = float(all_sizes.sum())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Median stone size
        if 'median' in query_lower and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['median_stone_size_cm'] = float(all_sizes.median())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Standard deviation stone size
        if any(word in query_lower for word in ['standard deviation', 'std', 'deviation']) and 'stone' in query_lower and 'size' in query_lower:
            if len(all_sizes) > 0:
                specific_stats['std_stone_size_cm'] = float(all_sizes.std())
                specific_stats['stone_size_count'] = len(all_sizes)
        
        # Count queries
        if 'how many' in query_lower or 'count' in query_lower:
            # This is handled by total_records, but we can add more specific counts
            if 'stone' in query_lower:
                stone_count = filtered_df['has_any_stone'].sum()
                specific_stats['patients_with_stones'] = int(stone_count)
            if 'bladder' in query_lower:
                bladder_count = filtered_df['bladder_volume_ml'].notna().sum()
                specific_stats['patients_with_bladder_volume'] = int(bladder_count)
        
        return specific_stats
    
    def format_summary(self, stats: Dict[str, Any]) -> str:
        """
        Format summary statistics for display, showing only relevant information.
        
        Args:
            stats: Summary statistics dictionary
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            "QUERY RESULTS",
            "=" * 60,
            f"Total records: {stats['total_records']}",
            f"Unique patients: {stats['unique_patients']}",
        ]
        
        # Show specific statistics first (these are the direct answers to queries)
        if 'mean_bladder_volume_ml' in stats:
            lines.append("")
            lines.append("Mean bladder volume:")
            lines.append(f"  • {stats['mean_bladder_volume_ml']:.1f} ml (based on {stats.get('bladder_volume_count', 0)} records)")
        
        if 'mean_stone_size_cm' in stats:
            lines.append("")
            lines.append("Mean stone size:")
            lines.append(f"  • {stats['mean_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'max_stone_size_cm' in stats:
            lines.append("")
            lines.append("Biggest stone size:")
            lines.append(f"  • {stats['max_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
            
            # Show patient details if available
            if 'patients_with_max_stone' in stats:
                lines.append("")
                lines.append("Patient(s) with biggest stone:")
                for patient in stats['patients_with_max_stone']:
                    right_size = patient.get('right_stone_size_cm')
                    left_size = patient.get('left_stone_size_cm')
                    imaging_date = patient.get('imaging_date')
                    
                    # Format date
                    if hasattr(imaging_date, 'strftime'):
                        date_str = imaging_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(imaging_date)
                    
                    # Show which side has the max stone
                    if pd.notna(right_size) and right_size == stats['max_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Right: {right_size:.2f} cm)")
                    if pd.notna(left_size) and left_size == stats['max_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Left: {left_size:.2f} cm)")
        
        if 'min_stone_size_cm' in stats:
            lines.append("")
            lines.append("Smallest stone size:")
            lines.append(f"  • {stats['min_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
            
            # Show patient details if available
            if 'patients_with_min_stone' in stats:
                lines.append("")
                lines.append("Patient(s) with smallest stone:")
                for patient in stats['patients_with_min_stone']:
                    right_size = patient.get('right_stone_size_cm')
                    left_size = patient.get('left_stone_size_cm')
                    imaging_date = patient.get('imaging_date')
                    
                    # Format date
                    if hasattr(imaging_date, 'strftime'):
                        date_str = imaging_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(imaging_date)
                    
                    # Show which side has the min stone
                    if pd.notna(right_size) and right_size == stats['min_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Right: {right_size:.2f} cm)")
                    if pd.notna(left_size) and left_size == stats['min_stone_size_cm']:
                        lines.append(f"  • {patient['recordid']} (Date: {date_str}, Left: {left_size:.2f} cm)")
        
        if 'sum_stone_size_cm' in stats:
            lines.append("")
            lines.append("Sum of stone sizes:")
            lines.append(f"  • {stats['sum_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'median_stone_size_cm' in stats:
            lines.append("")
            lines.append("Median stone size:")
            lines.append(f"  • {stats['median_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'std_stone_size_cm' in stats:
            lines.append("")
            lines.append("Standard deviation of stone sizes:")
            lines.append(f"  • {stats['std_stone_size_cm']:.2f} cm (based on {stats.get('stone_size_count', 0)} stones)")
        
        if 'patients_with_stones' in stats:
            lines.append("")
            lines.append("Patients with stones:")
            lines.append(f"  • {stats['patients_with_stones']} patients")
        
        if 'patients_with_bladder_volume' in stats:
            lines.append("")
            lines.append("Patients with bladder volume data:")
            lines.append(f"  • {stats['patients_with_bladder_volume']} patients")
        
        # Only show detailed distributions if no specific statistics were requested
        if not any(key in stats for key in ['mean_bladder_volume_ml', 'mean_stone_size_cm', 'max_stone_size_cm', 'min_stone_size_cm', 'sum_stone_size_cm', 'median_stone_size_cm', 'std_stone_size_cm', 'patients_with_stones', 'patients_with_bladder_volume']):
            # Only show stone distribution if it's relevant
            if 'stone_distribution' in stats:
                lines.append("")
                lines.append("Stone distribution:")
                stone_dist = stats['stone_distribution']
                lines.append(f"  • Left only: {stone_dist['left_only']}")
                lines.append(f"  • Right only: {stone_dist['right_only']}")
                lines.append(f"  • Bilateral: {stone_dist['bilateral']}")
                lines.append(f"  • No stones: {stone_dist['no_stones']}")
            
            # Only show stone size stats if relevant
            if 'stone_size_stats' in stats:
                size_stats = stats['stone_size_stats']
                if size_stats['count'] > 0:
                    lines.append("")
                    lines.append("Stone size statistics:")
                    lines.append(f"  • Count: {size_stats['count']}")
                    lines.append(f"  • Mean: {size_stats['mean']:.2f} cm")
                    lines.append(f"  • Median: {size_stats['median']:.2f} cm")
                    lines.append(f"  • Range: {size_stats['min']:.2f} - {size_stats['max']:.2f} cm")
            
            # Only show bladder volume stats if relevant
            if 'bladder_volume_stats' in stats:
                bladder_stats = stats['bladder_volume_stats']
                if bladder_stats['count'] > 0:
                    lines.append("")
                    lines.append("Bladder volume statistics:")
                    lines.append(f"  • Count: {bladder_stats['count']}")
                    lines.append(f"  • Mean: {bladder_stats['mean']:.1f} ml")
                    lines.append(f"  • Median: {bladder_stats['median']:.1f} ml")
                    lines.append(f"  • Range: {bladder_stats['min']:.1f} - {bladder_stats['max']:.1f} ml")
            
            # Only show temporal distribution if relevant
            if 'temporal_distribution' in stats:
                temporal_dist = stats['temporal_distribution']
                if temporal_dist:
                    lines.append("")
                    lines.append("Imaging by year:")
                    for year, count in sorted(temporal_dist.items()):
                        lines.append(f"  • {year}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
