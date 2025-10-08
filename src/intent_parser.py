"""Intent parser for natural language queries."""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging

from .llm_schema import UserQuery, PlanSummary

logger = logging.getLogger(__name__)

class IntentParser:
    """Parse natural language queries into structured filters and goals."""
    
    def __init__(self):
        # Common patterns for different types of queries
        self.patterns = {
            'count_queries': [
                r'how many',
                r'count',
                r'number of',
                r'total'
            ],
            'side_queries': [
                r'\b(left|right|bilateral|both)\b',
                r'\b(left|right)\s+(kidney|renal|side)\b'
            ],
            'size_queries': [
                r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)\s*(?:or\s+)?(?:larger|greater|bigger|>)',
                r'(?:larger|greater|bigger|>)\s*than\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)',
                r'(?:size|diameter)\s*(?:of\s+)?(?:at\s+least|>=?)\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeter[s]?|mm|millimeter[s]?)'
            ],
            'date_queries': [
                r'(?:between|from)\s+(\d{4})\s*(?:and|to|-)\s*(\d{4})',
                r'(?:in|during)\s+(\d{4})',
                r'(?:since|after)\s+(\d{4})',
                r'(?:before|prior\s+to)\s+(\d{4})'
            ],
            'stone_queries': [
                r'\bstone[s]?\b',
                r'\bcalculi\b',
                r'\bnephrolithiasis\b',
                r'\bkidney\s+stone[s]?\b',
                r'\brenal\s+stone[s]?\b'
            ],
            'bladder_queries': [
                r'\bbladder\b',
                r'\bvolume\b',
                r'\bcapacity\b'
            ]
        }
    
    def parse_query(self, query: str) -> UserQuery:
        """
        Parse a natural language query into structured components.
        
        Args:
            query: Natural language query string
            
        Returns:
            UserQuery object with parsed components
        """
        query_lower = query.lower().strip()
        
        # Extract goal
        goal = self._extract_goal(query_lower)
        
        # Extract input fields
        input_fields = self._extract_input_fields(query_lower)
        
        # Extract filters
        filters = self._extract_filters(query_lower)
        
        # Extract desired outputs
        outputs = self._extract_outputs(query_lower, goal)
        
        # Generate assumptions
        assumptions = self._generate_assumptions(filters, query_lower)
        
        return UserQuery(
            goal=goal,
            input_fields=input_fields,
            filters=filters,
            outputs=outputs,
            assumptions=assumptions
        )
    
    def _extract_goal(self, query: str) -> str:
        """Extract the main goal from the query."""
        if any(pattern in query for pattern in self.patterns['count_queries']):
            return "Count patients/records matching criteria"
        elif 'show' in query or 'display' in query or 'list' in query:
            return "Display matching records in table format"
        elif 'plot' in query or 'chart' in query or 'graph' in query:
            return "Create visualization of data"
        elif 'analyze' in query or 'compare' in query:
            return "Analyze and compare data"
        else:
            return "Extract and filter data based on criteria"
    
    def _extract_input_fields(self, query: str) -> List[str]:
        """Extract which input fields are relevant to the query."""
        fields = []
        
        if any(pattern in query for pattern in self.patterns['stone_queries']):
            fields.append('narrative')
        if any(pattern in query for pattern in self.patterns['bladder_queries']):
            fields.append('narrative')
        if 'date' in query or 'year' in query or 'time' in query:
            fields.append('imaging_date')
        if 'surgery' in query or 'surg' in query:
            fields.append('surg_date')
        
        # Default to narrative if no specific fields detected
        if not fields:
            fields = ['narrative']
        
        return fields
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from the query."""
        filters = {}
        
        # Extract side filters
        side_filter = self._extract_side_filter(query)
        if side_filter:
            filters['side'] = side_filter
        
        # Extract size filters
        size_filter = self._extract_size_filter(query)
        if size_filter:
            filters.update(size_filter)
        
        # Extract date filters
        date_filter = self._extract_date_filter(query)
        if date_filter:
            filters.update(date_filter)
        
        # Extract stone presence filters
        stone_filter = self._extract_stone_filter(query)
        if stone_filter:
            filters.update(stone_filter)
        
        # Extract bladder filters
        bladder_filter = self._extract_bladder_filter(query)
        if bladder_filter:
            filters.update(bladder_filter)
        
        return filters
    
    def _extract_side_filter(self, query: str) -> Optional[str]:
        """Extract side filter (left, right, bilateral)."""
        if 'left' in query and 'right' not in query:
            return 'left'
        elif 'right' in query and 'left' not in query:
            return 'right'
        elif 'bilateral' in query or 'both' in query:
            return 'bilateral'
        return None
    
    def _extract_size_filter(self, query: str) -> Dict[str, Any]:
        """Extract size-related filters."""
        filters = {}
        
        for pattern in self.patterns['size_queries']:
            matches = re.findall(pattern, query)
            if matches:
                size_value = float(matches[0])
                
                # Convert mm to cm if needed
                if 'mm' in query:
                    size_value = size_value / 10
                
                if 'larger' in query or 'greater' in query or 'bigger' in query or '>' in query:
                    filters['min_size_cm'] = size_value
                elif 'smaller' in query or 'less' in query or '<' in query:
                    filters['max_size_cm'] = size_value
                else:
                    # Default to minimum size
                    filters['min_size_cm'] = size_value
                
                break
        
        return filters
    
    def _extract_date_filter(self, query: str) -> Dict[str, Any]:
        """Extract date range filters."""
        filters = {}
        
        for pattern in self.patterns['date_queries']:
            matches = re.findall(pattern, query)
            if matches:
                if len(matches[0]) == 2:  # Between X and Y
                    start_year, end_year = matches[0]
                    filters['start_year'] = int(start_year)
                    filters['end_year'] = int(end_year)
                else:  # Single year
                    year = int(matches[0])
                    if 'since' in query or 'after' in query:
                        filters['start_year'] = year
                    elif 'before' in query or 'prior' in query:
                        filters['end_year'] = year - 1
                    else:
                        filters['start_year'] = year
                        filters['end_year'] = year
                break
        
        return filters
    
    def _extract_stone_filter(self, query: str) -> Dict[str, Any]:
        """Extract stone-related filters."""
        filters = {}
        
        if any(pattern in query for pattern in self.patterns['stone_queries']):
            if 'no' in query or 'without' in query or 'absent' in query:
                filters['stone_presence'] = 'absent'
            elif 'present' in query or 'with' in query:
                filters['stone_presence'] = 'present'
            else:
                filters['stone_presence'] = 'present'  # Default assumption
        
        return filters
    
    def _extract_bladder_filter(self, query: str) -> Dict[str, Any]:
        """Extract bladder-related filters."""
        filters = {}
        
        if any(pattern in query for pattern in self.patterns['bladder_queries']):
            # Look for volume specifications
            volume_pattern = r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter[s]?)'
            volume_matches = re.findall(volume_pattern, query)
            if volume_matches:
                volume = float(volume_matches[0])
                if 'larger' in query or 'greater' in query or '>' in query:
                    filters['min_bladder_volume_ml'] = volume
                elif 'smaller' in query or 'less' in query or '<' in query:
                    filters['max_bladder_volume_ml'] = volume
                else:
                    filters['min_bladder_volume_ml'] = volume
        
        return filters
    
    def _extract_outputs(self, query: str, goal: str) -> List[str]:
        """Extract desired output columns."""
        outputs = ['recordid', 'imaging_date']  # Always include these
        
        # Don't return early for count queries - we still need to compute statistics
        
        # Add relevant columns based on query content
        if 'stone' in query:
            outputs.extend(['right_stone', 'left_stone'])
            if 'size' in query:
                outputs.extend(['right_stone_size_cm', 'left_stone_size_cm'])
        
        if 'kidney' in query and 'size' in query:
            outputs.extend(['right_kidney_size_cm', 'left_kidney_size_cm'])
        
        if 'bladder' in query or 'volume' in query:
            outputs.append('bladder_volume_ml')
        
        if 'history' in query:
            outputs.append('history_summary')
        
        return outputs
    
    def _generate_assumptions(self, filters: Dict[str, Any], query: str) -> List[str]:
        """Generate assumptions made during parsing."""
        assumptions = []
        
        # Negation handling
        if 'no' in query or 'without' in query:
            assumptions.append("Negation terms detected - will look for absence of findings")
        else:
            assumptions.append("Will look for presence of findings (default assumption)")
        
        # Size assumptions
        if 'min_size_cm' in filters:
            assumptions.append(f"Minimum stone size threshold: {filters['min_size_cm']} cm")
        if 'max_size_cm' in filters:
            assumptions.append(f"Maximum stone size threshold: {filters['max_size_cm']} cm")
        
        # Date assumptions
        if 'start_year' in filters or 'end_year' in filters:
            assumptions.append("Date filtering will be applied to imaging_date")
        
        # Side assumptions
        if 'side' in filters:
            if filters['side'] == 'bilateral':
                assumptions.append("Will include both left and right sides")
            else:
                assumptions.append(f"Will focus on {filters['side']} side only")
        
        # Missing data assumptions
        assumptions.append("Missing or unclear data will be included as 'unclear' or null")
        assumptions.append("Size measurements will be converted to cm for consistency")
        
        return assumptions
    
    def create_plan_summary(self, user_query: UserQuery, 
                          estimated_rows: Optional[int] = None) -> PlanSummary:
        """
        Create a formatted plan summary for user confirmation.
        
        Args:
            user_query: Parsed user query
            estimated_rows: Estimated number of matching rows
            
        Returns:
            PlanSummary object
        """
        processing_time_estimate = "1-2 minutes" if estimated_rows and estimated_rows > 100 else "30-60 seconds"
        
        return PlanSummary(
            goal=user_query.goal,
            input_fields=user_query.input_fields,
            filters=user_query.filters,
            outputs=user_query.outputs,
            assumptions=user_query.assumptions,
            estimated_rows=estimated_rows,
            processing_time_estimate=processing_time_estimate
        )
    
    def format_plan_summary(self, plan: PlanSummary) -> str:
        """
        Format plan summary for display to user.
        
        Args:
            plan: PlanSummary object
            
        Returns:
            Formatted string for display
        """
        lines = [
            "=" * 60,
            "PLAN SUMMARY",
            "=" * 60,
            f"Goal: {plan.goal}",
            "",
            "Input fields detected:",
        ]
        
        for field in plan.input_fields:
            lines.append(f"  • {field}")
        
        lines.append("")
        lines.append("Filters:")
        if plan.filters:
            for key, value in plan.filters.items():
                lines.append(f"  • {key}: {value}")
        else:
            lines.append("  • No specific filters applied")
        
        lines.append("")
        lines.append("Outputs:")
        for output in plan.outputs:
            lines.append(f"  • {output}")
        
        lines.append("")
        lines.append("Assumptions:")
        for assumption in plan.assumptions:
            lines.append(f"  • {assumption}")
        
        if plan.estimated_rows:
            lines.append("")
            lines.append(f"Estimated matching rows: {plan.estimated_rows}")
        
        lines.append("")
        lines.append(f"Estimated processing time: {plan.processing_time_estimate}")
        lines.append("=" * 60)
        
        return "\n".join(lines)

