"""
Reflection architecture for self-assessment and improvement of query results.

This module implements a reflection system that allows the agent to:
1. Assess the quality of its initial answers
2. Identify potential issues or improvements
3. Generate refined answers based on reflection
4. Learn from its mistakes and successes
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd

from .config import get_llm_client, LLM_PROVIDER
from .llm_schema import UserQuery, PlanSummary

logger = logging.getLogger(__name__)

class ReflectionResult:
    """Container for reflection analysis results."""
    
    def __init__(self, 
                 quality_score: float,
                 issues: List[str],
                 improvements: List[str],
                 confidence: float,
                 reasoning: str,
                 suggested_actions: List[str]):
        self.quality_score = quality_score  # 0.0 to 1.0
        self.issues = issues
        self.improvements = improvements
        self.confidence = confidence  # 0.0 to 1.0
        self.reasoning = reasoning
        self.suggested_actions = suggested_actions
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'quality_score': self.quality_score,
            'issues': self.issues,
            'improvements': self.improvements,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'suggested_actions': self.suggested_actions,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionResult':
        """Create from dictionary."""
        return cls(
            quality_score=data['quality_score'],
            issues=data['issues'],
            improvements=data['improvements'],
            confidence=data['confidence'],
            reasoning=data['reasoning'],
            suggested_actions=data['suggested_actions']
        )

class ReflectionArchitecture:
    """
    Reflection architecture for self-assessment and improvement.
    
    This class implements a multi-stage reflection process:
    1. Initial Answer Assessment
    2. Issue Identification
    3. Improvement Generation
    4. Self-Correction
    5. Learning and Adaptation
    """
    
    def __init__(self):
        self.client = get_llm_client()
        self.provider = LLM_PROVIDER
        self.reflection_history_file = "out/reflection_history.json"
        self.learning_patterns_file = "out/learning_patterns.json"
        self.reflection_history = self._load_reflection_history()
        self.learning_patterns = self._load_learning_patterns()
        
        # Reflection criteria weights
        self.criteria_weights = {
            'accuracy': 0.3,
            'completeness': 0.25,
            'relevance': 0.2,
            'clarity': 0.15,
            'efficiency': 0.1
        }
    
    def reflect_on_answer(self, 
                         query: str,
                         user_query: UserQuery,
                         initial_answer: Dict[str, Any],
                         filtered_df: pd.DataFrame,
                         processing_time: float) -> ReflectionResult:
        """
        Perform comprehensive reflection on the initial answer.
        
        Args:
            query: Original user query
            user_query: Parsed user query object
            initial_answer: Initial answer dictionary
            filtered_df: Filtered DataFrame used for the answer
            processing_time: Time taken to process the query
            
        Returns:
            ReflectionResult with assessment and suggestions
        """
        logger.info("Starting reflection on initial answer")
        
        # Stage 1: Assess answer quality
        quality_assessment = self._assess_answer_quality(
            query, user_query, initial_answer, filtered_df
        )
        
        # Stage 2: Identify issues
        issues = self._identify_issues(
            query, user_query, initial_answer, filtered_df, quality_assessment
        )
        
        # Stage 3: Generate improvements
        improvements = self._generate_improvements(
            query, user_query, initial_answer, filtered_df, issues
        )
        
        # Stage 4: Calculate confidence
        confidence = self._calculate_confidence(
            quality_assessment, issues, improvements, processing_time
        )
        
        # Stage 5: Generate reasoning
        reasoning = self._generate_reasoning(
            quality_assessment, issues, improvements, confidence
        )
        
        # Stage 6: Suggest actions
        suggested_actions = self._suggest_actions(
            issues, improvements, confidence
        )
        
        # Create reflection result
        reflection_result = ReflectionResult(
            quality_score=quality_assessment['overall_score'],
            issues=issues,
            improvements=improvements,
            confidence=confidence,
            reasoning=reasoning,
            suggested_actions=suggested_actions
        )
        
        # Store reflection for learning
        self._store_reflection(query, reflection_result)
        
        logger.info(f"Reflection completed - Quality: {quality_assessment['overall_score']:.2f}, "
                   f"Confidence: {confidence:.2f}")
        
        return reflection_result
    
    def _assess_answer_quality(self, 
                              query: str,
                              user_query: UserQuery,
                              initial_answer: Dict[str, Any],
                              filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the initial answer across multiple criteria."""
        
        assessment = {
            'accuracy': 0.0,
            'completeness': 0.0,
            'relevance': 0.0,
            'clarity': 0.0,
            'efficiency': 0.0,
            'overall_score': 0.0
        }
        
        # Accuracy assessment
        assessment['accuracy'] = self._assess_accuracy(
            query, user_query, initial_answer, filtered_df
        )
        
        # Completeness assessment
        assessment['completeness'] = self._assess_completeness(
            query, user_query, initial_answer, filtered_df
        )
        
        # Relevance assessment
        assessment['relevance'] = self._assess_relevance(
            query, user_query, initial_answer, filtered_df
        )
        
        # Clarity assessment
        assessment['clarity'] = self._assess_clarity(
            query, user_query, initial_answer
        )
        
        # Efficiency assessment
        assessment['efficiency'] = self._assess_efficiency(
            query, user_query, initial_answer, filtered_df
        )
        
        # Calculate overall score
        assessment['overall_score'] = sum(
            assessment[criterion] * weight 
            for criterion, weight in self.criteria_weights.items()
        )
        
        return assessment
    
    def _assess_accuracy(self, 
                        query: str,
                        user_query: UserQuery,
                        initial_answer: Dict[str, Any],
                        filtered_df: pd.DataFrame) -> float:
        """Assess the accuracy of the answer."""
        
        # Check if filters were applied correctly
        filter_accuracy = 1.0
        
        # Verify side filter accuracy
        if 'side' in user_query.filters:
            side = user_query.filters['side']
            if side == 'left':
                expected_stones = filtered_df['has_left_stone'].sum()
                actual_stones = initial_answer.get('total_records', 0)
                if expected_stones != actual_stones:
                    filter_accuracy *= 0.8
        
        # Verify size filter accuracy
        if 'min_size_cm' in user_query.filters:
            min_size = user_query.filters['min_size_cm']
            if 'side' in user_query.filters:
                side = user_query.filters['side']
                if side == 'left':
                    expected_count = (filtered_df['left_stone_size_cm'] >= min_size).sum()
                elif side == 'right':
                    expected_count = (filtered_df['right_stone_size_cm'] >= min_size).sum()
                else:
                    expected_count = (
                        (filtered_df['left_stone_size_cm'] >= min_size) |
                        (filtered_df['right_stone_size_cm'] >= min_size)
                    ).sum()
                
                actual_count = initial_answer.get('total_records', 0)
                if expected_count != actual_count:
                    filter_accuracy *= 0.7
        
        # Verify bladder volume filter accuracy
        if 'min_bladder_volume_ml' in user_query.filters:
            min_volume = user_query.filters['min_bladder_volume_ml']
            expected_count = (filtered_df['bladder_volume_ml'] >= min_volume).sum()
            actual_count = initial_answer.get('total_records', 0)
            if expected_count != actual_count:
                filter_accuracy *= 0.6
        
        # Verify stone presence filter accuracy
        if 'stone_presence' in user_query.filters:
            presence = user_query.filters['stone_presence']
            if presence == 'absent':
                # For absent stones, should return patients with no stones
                expected_count = (~filtered_df['has_any_stone']).sum()
            elif presence == 'present':
                # For present stones, should return patients with stones
                expected_count = filtered_df['has_any_stone'].sum()
            else:
                expected_count = len(filtered_df)
            
            actual_count = initial_answer.get('total_records', 0)
            if expected_count != actual_count:
                filter_accuracy *= 0.5  # Stone presence is critical, so heavy penalty
        
        # Verify date filter accuracy
        if 'start_year' in user_query.filters or 'end_year' in user_query.filters:
            start_year = user_query.filters.get('start_year')
            end_year = user_query.filters.get('end_year')
            
            if start_year and end_year:
                expected_count = ((filtered_df['imaging_date'].dt.year >= start_year) & 
                                (filtered_df['imaging_date'].dt.year <= end_year)).sum()
            elif start_year:
                expected_count = (filtered_df['imaging_date'].dt.year >= start_year).sum()
            elif end_year:
                expected_count = (filtered_df['imaging_date'].dt.year <= end_year).sum()
            else:
                expected_count = len(filtered_df)
            
            actual_count = initial_answer.get('total_records', 0)
            if expected_count != actual_count:
                filter_accuracy *= 0.8
        
        # Verify max size filter accuracy
        if 'max_size_cm' in user_query.filters:
            max_size = user_query.filters['max_size_cm']
            if 'side' in user_query.filters:
                side = user_query.filters['side']
                if side == 'left':
                    expected_count = (filtered_df['left_stone_size_cm'] <= max_size).sum()
                elif side == 'right':
                    expected_count = (filtered_df['right_stone_size_cm'] <= max_size).sum()
                else:
                    expected_count = (
                        (filtered_df['left_stone_size_cm'] <= max_size) |
                        (filtered_df['right_stone_size_cm'] <= max_size)
                    ).sum()
            else:
                expected_count = (
                    (filtered_df['left_stone_size_cm'] <= max_size) |
                    (filtered_df['right_stone_size_cm'] <= max_size)
                ).sum()
            
            actual_count = initial_answer.get('total_records', 0)
            if expected_count != actual_count:
                filter_accuracy *= 0.7
        
        return filter_accuracy
    
    def _assess_completeness(self, 
                           query: str,
                           user_query: UserQuery,
                           initial_answer: Dict[str, Any],
                           filtered_df: pd.DataFrame) -> float:
        """Assess the completeness of the answer."""
        
        completeness = 1.0
        
        # Check if all requested information is provided
        if 'how many' in query.lower():
            if 'total_records' not in initial_answer:
                completeness *= 0.5
        
        if 'mean' in query.lower():
            if 'mean_bladder_volume_ml' not in initial_answer and 'mean_stone_size_cm' not in initial_answer:
                completeness *= 0.6
        
        if 'biggest' in query.lower() or 'largest' in query.lower():
            if 'max_stone_size_cm' not in initial_answer:
                completeness *= 0.7
        
        if 'which patient' in query.lower() or 'who' in query.lower():
            if 'matching_patients' not in initial_answer and 'patients_with_max_stone' not in initial_answer:
                completeness *= 0.8
        
        # Check if relevant statistics are missing
        if any(word in query.lower() for word in ['statistics', 'distribution', 'analysis']):
            if 'stone_distribution' not in initial_answer and 'stone_size_stats' not in initial_answer:
                completeness *= 0.7
        
        return completeness
    
    def _assess_relevance(self, 
                         query: str,
                         user_query: UserQuery,
                         initial_answer: Dict[str, Any],
                         filtered_df: pd.DataFrame) -> float:
        """Assess the relevance of the answer to the query."""
        
        relevance = 1.0
        
        # Check if the answer addresses the specific query intent
        if 'left' in query.lower() and 'right' not in query.lower():
            # Query is about left side specifically
            if 'matching_patients' in initial_answer:
                # Check if left side information is included
                left_info_present = any(
                    'left' in str(patient).lower() 
                    for patient in initial_answer['matching_patients'][:3]
                )
                if not left_info_present:
                    relevance *= 0.8
        
        if 'bladder' in query.lower():
            # Query is about bladder
            if 'bladder_volume_stats' not in initial_answer and 'mean_bladder_volume_ml' not in initial_answer:
                relevance *= 0.6
        
        if 'stone' in query.lower():
            # Query is about stones
            if 'stone_distribution' not in initial_answer and 'stone_size_stats' not in initial_answer:
                relevance *= 0.7
        
        return relevance
    
    def _assess_clarity(self, 
                       query: str,
                       user_query: UserQuery,
                       initial_answer: Dict[str, Any]) -> float:
        """Assess the clarity of the answer."""
        
        clarity = 1.0
        
        # Check if the answer is well-structured
        if 'total_records' in initial_answer:
            if initial_answer['total_records'] == 0:
                # Empty results should be clearly explained
                if 'matching_patients' not in initial_answer or len(initial_answer.get('matching_patients', [])) == 0:
                    clarity *= 0.9  # Good - clearly shows no results
                else:
                    clarity *= 0.5  # Bad - contradictory information
        
        # Check if specific statistics are clearly presented
        if 'mean_bladder_volume_ml' in initial_answer:
            if 'bladder_volume_count' not in initial_answer:
                clarity *= 0.8  # Missing context for the mean
        
        if 'max_stone_size_cm' in initial_answer:
            if 'stone_size_count' not in initial_answer:
                clarity *= 0.8  # Missing context for the max
        
        return clarity
    
    def _assess_efficiency(self, 
                          query: str,
                          user_query: UserQuery,
                          initial_answer: Dict[str, Any],
                          filtered_df: pd.DataFrame) -> float:
        """Assess the efficiency of the answer."""
        
        efficiency = 1.0
        
        # Check if the answer is concise and focused
        total_records = initial_answer.get('total_records', 0)
        
        # For count queries, the answer should be concise
        if 'how many' in query.lower():
            if total_records > 0 and 'matching_patients' in initial_answer:
                # If showing patient details for count queries, limit appropriately
                patient_count = len(initial_answer['matching_patients'])
                if patient_count > 20:
                    efficiency *= 0.8  # Too many details for a count query
        
        # Check if unnecessary information is included
        if 'bladder' not in query.lower():
            if 'bladder_volume_stats' in initial_answer:
                efficiency *= 0.9  # Unnecessary bladder information
        
        if 'stone' not in query.lower():
            if 'stone_distribution' in initial_answer:
                efficiency *= 0.9  # Unnecessary stone information
        
        return efficiency
    
    def _identify_issues(self, 
                        query: str,
                        user_query: UserQuery,
                        initial_answer: Dict[str, Any],
                        filtered_df: pd.DataFrame,
                        quality_assessment: Dict[str, Any]) -> List[str]:
        """Identify specific issues with the initial answer."""
        
        issues = []
        
        # Accuracy issues
        if quality_assessment['accuracy'] < 0.8:
            issues.append("Potential accuracy issues detected in filtering logic")
        
        # Completeness issues
        if quality_assessment['completeness'] < 0.8:
            issues.append("Answer may be missing requested information")
        
        # Relevance issues
        if quality_assessment['relevance'] < 0.8:
            issues.append("Answer may not fully address the query intent")
        
        # Clarity issues
        if quality_assessment['clarity'] < 0.8:
            issues.append("Answer structure could be clearer")
        
        # Efficiency issues
        if quality_assessment['efficiency'] < 0.8:
            issues.append("Answer may include unnecessary information")
        
        # Specific issue detection
        if 'how many' in query.lower() and 'total_records' in initial_answer:
            if initial_answer['total_records'] == 0:
                # Check if this is expected
                if len(filtered_df) > 0:
                    issues.append("No results returned despite matching data")
        
        # Check for contradictory information
        if 'matching_patients' in initial_answer and 'total_records' in initial_answer:
            if len(initial_answer['matching_patients']) != initial_answer['total_records']:
                issues.append("Inconsistent patient count information")
        
        # Comprehensive heuristic checks
        issues.extend(self._check_common_issues(query, user_query, initial_answer, filtered_df))
        
        return issues
    
    def _check_common_issues(self, 
                            query: str,
                            user_query: UserQuery,
                            initial_answer: Dict[str, Any],
                            filtered_df: pd.DataFrame) -> List[str]:
        """Check for common issues using heuristics."""
        issues = []
        
        # Check 1: Count queries should have total_records
        if 'count' in query.lower() or 'how many' in query.lower():
            if 'total_records' not in initial_answer:
                issues.append("Count query missing 'total_records' field")
        
        # Check 2: Stone presence queries should have appropriate counts
        if 'stone_presence' in user_query.filters:
            presence = user_query.filters['stone_presence']
            actual_count = initial_answer.get('total_records', 0)
            
            if presence == 'absent':
                # Should return patients with no stones
                expected_count = (~filtered_df['has_any_stone']).sum()
                if actual_count != expected_count:
                    issues.append(f"Stone absence filter returned {actual_count} patients, expected {expected_count}")
            elif presence == 'present':
                # Should return patients with stones
                expected_count = filtered_df['has_any_stone'].sum()
                if actual_count != expected_count:
                    issues.append(f"Stone presence filter returned {actual_count} patients, expected {expected_count}")
        
        # Check 3: Size filter queries should respect size constraints
        if 'min_size_cm' in user_query.filters or 'max_size_cm' in user_query.filters:
            min_size = user_query.filters.get('min_size_cm')
            max_size = user_query.filters.get('max_size_cm')
            side = user_query.filters.get('side')
            
            # Check if any returned patients violate size constraints
            if side == 'left' and 'left_stone_size_cm' in filtered_df.columns:
                if min_size and (filtered_df['left_stone_size_cm'] < min_size).any():
                    issues.append("Some patients have left stones smaller than minimum size")
                if max_size and (filtered_df['left_stone_size_cm'] > max_size).any():
                    issues.append("Some patients have left stones larger than maximum size")
            elif side == 'right' and 'right_stone_size_cm' in filtered_df.columns:
                if min_size and (filtered_df['right_stone_size_cm'] < min_size).any():
                    issues.append("Some patients have right stones smaller than minimum size")
                if max_size and (filtered_df['right_stone_size_cm'] > max_size).any():
                    issues.append("Some patients have right stones larger than maximum size")
        
        # Check 4: Statistical queries should have appropriate statistics
        if 'mean' in query.lower() or 'average' in query.lower():
            if 'mean_bladder_volume_ml' not in initial_answer and 'mean_stone_size_cm' not in initial_answer:
                issues.append("Mean/average query missing statistical information")
        
        if 'biggest' in query.lower() or 'largest' in query.lower():
            if 'max_stone_size_cm' not in initial_answer:
                issues.append("Max size query missing maximum stone size information")
        
        if 'smallest' in query.lower():
            if 'min_stone_size_cm' not in initial_answer:
                issues.append("Min size query missing minimum stone size information")
        
        # Check 5: "Which patient" queries should have patient details
        if 'which patient' in query.lower() or 'who has' in query.lower():
            if 'matching_patients' not in initial_answer:
                issues.append("'Which patient' query missing patient details")
        
        # Check 6: Date range queries should respect date constraints
        if 'start_year' in user_query.filters or 'end_year' in user_query.filters:
            start_year = user_query.filters.get('start_year')
            end_year = user_query.filters.get('end_year')
            
            if start_year and (filtered_df['imaging_date'].dt.year < start_year).any():
                issues.append("Some patients have dates before the start year")
            if end_year and (filtered_df['imaging_date'].dt.year > end_year).any():
                issues.append("Some patients have dates after the end year")
        
        # Check 7: Bladder volume queries should respect volume constraints
        if 'min_bladder_volume_ml' in user_query.filters:
            min_volume = user_query.filters['min_bladder_volume_ml']
            if (filtered_df['bladder_volume_ml'] < min_volume).any():
                issues.append("Some patients have bladder volumes below minimum threshold")
        
        # Check 8: Side-specific queries should only return patients with stones on that side
        if 'side' in user_query.filters:
            side = user_query.filters['side']
            if side == 'left':
                if (filtered_df['has_left_stone'] == False).any():
                    issues.append("Left side query returned patients without left stones")
            elif side == 'right':
                if (filtered_df['has_right_stone'] == False).any():
                    issues.append("Right side query returned patients without right stones")
        
        return issues
    
    def _generate_improvements(self, 
                              query: str,
                              user_query: UserQuery,
                              initial_answer: Dict[str, Any],
                              filtered_df: pd.DataFrame,
                              issues: List[str]) -> List[str]:
        """Generate specific improvements for the answer."""
        
        improvements = []
        
        # Generate improvements based on issues
        for issue in issues:
            if "accuracy" in issue.lower():
                improvements.append("Verify filter logic and data consistency")
            elif "completeness" in issue.lower():
                improvements.append("Include all requested statistics and details")
            elif "relevance" in issue.lower():
                improvements.append("Focus answer on specific query requirements")
            elif "clarity" in issue.lower():
                improvements.append("Improve answer structure and organization")
            elif "efficiency" in issue.lower():
                improvements.append("Remove unnecessary information and focus on essentials")
        
        # Generate improvements based on query type
        if 'how many' in query.lower():
            if 'matching_patients' not in initial_answer:
                improvements.append("Consider showing sample matching patients for context")
        
        if 'mean' in query.lower():
            if 'mean_bladder_volume_ml' not in initial_answer and 'mean_stone_size_cm' not in initial_answer:
                improvements.append("Calculate and display requested mean values")
        
        if 'which patient' in query.lower() or 'who' in query.lower():
            if 'matching_patients' not in initial_answer:
                improvements.append("List specific patients matching the criteria")
        
        # Generate improvements based on data availability
        if len(filtered_df) > 0:
            if 'stone' in query.lower() and 'stone_size_stats' not in initial_answer:
                improvements.append("Include stone size distribution statistics")
            
            if 'bladder' in query.lower() and 'bladder_volume_stats' not in initial_answer:
                improvements.append("Include bladder volume distribution statistics")
        
        # Generate specific improvements based on detected issues
        for issue in issues:
            if "Stone absence filter returned" in issue:
                improvements.append("Fix stone presence filtering logic to correctly identify patients without stones")
            elif "Stone presence filter returned" in issue:
                improvements.append("Fix stone presence filtering logic to correctly identify patients with stones")
            elif "smaller than minimum size" in issue:
                improvements.append("Review size filtering logic to ensure minimum size constraints are properly applied")
            elif "larger than maximum size" in issue:
                improvements.append("Review size filtering logic to ensure maximum size constraints are properly applied")
            elif "missing statistical information" in issue:
                improvements.append("Add missing statistical calculations (mean, max, min) to the response")
            elif "missing patient details" in issue:
                improvements.append("Include patient IDs and relevant details for 'which patient' queries")
            elif "dates before the start year" in issue:
                improvements.append("Fix date filtering to exclude patients before the start year")
            elif "dates after the end year" in issue:
                improvements.append("Fix date filtering to exclude patients after the end year")
            elif "bladder volumes below minimum threshold" in issue:
                improvements.append("Fix bladder volume filtering to exclude patients below the minimum threshold")
            elif "returned patients without left stones" in issue:
                improvements.append("Fix left side filtering to only include patients with left stones")
            elif "returned patients without right stones" in issue:
                improvements.append("Fix right side filtering to only include patients with right stones")
        
        return improvements
    
    def _calculate_confidence(self, 
                             quality_assessment: Dict[str, Any],
                             issues: List[str],
                             improvements: List[str],
                             processing_time: float) -> float:
        """Calculate confidence in the answer quality."""
        
        # Base confidence on quality assessment
        base_confidence = quality_assessment['overall_score']
        
        # Adjust for number of issues
        issue_penalty = len(issues) * 0.1
        confidence = max(0.0, base_confidence - issue_penalty)
        
        # Adjust for processing time (faster is better, up to a point)
        if processing_time > 5.0:  # More than 5 seconds
            confidence *= 0.95
        elif processing_time < 1.0:  # Less than 1 second
            confidence *= 1.05
        
        # Cap confidence between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _generate_reasoning(self, 
                           quality_assessment: Dict[str, Any],
                           issues: List[str],
                           improvements: List[str],
                           confidence: float) -> str:
        """Generate reasoning for the reflection assessment."""
        
        reasoning_parts = []
        
        # Overall assessment
        overall_score = quality_assessment['overall_score']
        if overall_score >= 0.9:
            reasoning_parts.append("High-quality answer with excellent accuracy and completeness")
        elif overall_score >= 0.7:
            reasoning_parts.append("Good answer with minor areas for improvement")
        elif overall_score >= 0.5:
            reasoning_parts.append("Adequate answer with several areas needing attention")
        else:
            reasoning_parts.append("Answer requires significant improvement")
        
        # Specific criteria
        if quality_assessment['accuracy'] >= 0.9:
            reasoning_parts.append("Filtering logic appears accurate")
        elif quality_assessment['accuracy'] < 0.7:
            reasoning_parts.append("Potential accuracy issues in data filtering")
        
        if quality_assessment['completeness'] >= 0.9:
            reasoning_parts.append("Answer provides comprehensive information")
        elif quality_assessment['completeness'] < 0.7:
            reasoning_parts.append("Answer may be missing requested details")
        
        # Issues summary
        if issues:
            reasoning_parts.append(f"Identified {len(issues)} specific issues")
        
        # Improvements summary
        if improvements:
            reasoning_parts.append(f"Generated {len(improvements)} improvement suggestions")
        
        # Confidence assessment
        if confidence >= 0.8:
            reasoning_parts.append("High confidence in answer quality")
        elif confidence >= 0.6:
            reasoning_parts.append("Moderate confidence in answer quality")
        else:
            reasoning_parts.append("Low confidence in answer quality")
        
        return ". ".join(reasoning_parts) + "."
    
    def _suggest_actions(self, 
                        issues: List[str],
                        improvements: List[str],
                        confidence: float) -> List[str]:
        """Suggest specific actions to improve the answer."""
        
        actions = []
        
        # Actions based on confidence level
        if confidence < 0.6:
            actions.append("Consider re-processing the query with different parameters")
            actions.append("Verify data extraction and filtering logic")
        
        # Actions based on issues
        for issue in issues:
            if "accuracy" in issue.lower():
                actions.append("Review and validate filter application logic")
            elif "completeness" in issue.lower():
                actions.append("Ensure all requested information is included")
            elif "relevance" in issue.lower():
                actions.append("Refine answer to better match query intent")
        
        # Actions based on improvements
        for improvement in improvements:
            if "verify" in improvement.lower():
                actions.append("Double-check data consistency and filter results")
            elif "include" in improvement.lower():
                actions.append("Add missing statistical information")
            elif "focus" in improvement.lower():
                actions.append("Streamline answer to address specific query")
        
        # General actions
        if len(issues) > 2:
            actions.append("Consider breaking down complex query into simpler parts")
        
        if confidence < 0.7:
            actions.append("Provide additional context and explanation")
        
        return actions
    
    def _store_reflection(self, query: str, reflection_result: ReflectionResult):
        """Store reflection result for learning and analysis."""
        
        reflection_entry = {
            'query': query,
            'reflection': reflection_result.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.reflection_history.append(reflection_entry)
        
        # Keep only last 100 reflections to prevent file from growing too large
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
        
        self._save_reflection_history()
    
    def _load_reflection_history(self) -> List[Dict[str, Any]]:
        """Load reflection history from file."""
        try:
            if os.path.exists(self.reflection_history_file):
                with open(self.reflection_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load reflection history: {e}")
        return []
    
    def _save_reflection_history(self):
        """Save reflection history to file."""
        try:
            os.makedirs(os.path.dirname(self.reflection_history_file), exist_ok=True)
            with open(self.reflection_history_file, 'w') as f:
                json.dump(self.reflection_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reflection history: {e}")
    
    def _load_learning_patterns(self) -> Dict[str, Any]:
        """Load learning patterns from file."""
        try:
            if os.path.exists(self.learning_patterns_file):
                with open(self.learning_patterns_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load learning patterns: {e}")
        return {
            'common_issues': {},
            'successful_patterns': {},
            'improvement_suggestions': {}
        }
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection history for analysis."""
        
        if not self.reflection_history:
            return {'message': 'No reflection history available'}
        
        # Calculate average quality scores
        quality_scores = [r['reflection']['quality_score'] for r in self.reflection_history]
        confidence_scores = [r['reflection']['confidence'] for r in self.reflection_history]
        
        # Count common issues
        all_issues = []
        for r in self.reflection_history:
            all_issues.extend(r['reflection']['issues'])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'total_reflections': len(self.reflection_history),
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'recent_reflections': self.reflection_history[-5:] if len(self.reflection_history) >= 5 else self.reflection_history
        }
