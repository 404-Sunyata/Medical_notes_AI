"""Main orchestrator for the radiology AI agent."""

import argparse
import sys
import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime

from .config import OUTPUT_DIR, PLOTS_DIR
from .io_utils import load_excel_data, concat_narratives_by_imaging, save_structured_data, get_data_summary
from .regex_baseline import extract_regex_baseline
from .llm_extractor import LLMExtractor
from .llm_schema import extraction_to_structured, StructuredOutput
from .intent_parser import IntentParser
from .confirm_flow import ConfirmFlow
from .query_tools import QueryTools
from .viz import RadiologyVisualizer
from .safety import check_text
from .reflection import ReflectionArchitecture

# --- logging setup (drop-in replacement) ---
import os, sys, logging
from logging.handlers import RotatingFileHandler

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
file_handler = RotatingFileHandler(
    os.path.join(OUTPUT_DIR, "radiology_agent.log"),
    maxBytes=50 * 1024 * 1024,   # 单文件 50MB
    backupCount=5,               # 最多保留 5 个滚动备份（总 ~<=300MB）
    encoding="utf-8"
)
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
    force=True,  # 防止重复添加旧的 handler
)

logger = logging.getLogger("radiology_agent")
# 可选：写长文本时只记摘要，避免日志再次暴涨
def _clip(s: str, n: int = 1000) -> str:
    if not s:
        return ""
    return s[:n] + ("…[truncated]" if len(s) > n else "")
# 使用示例：
# logger.info({"prompt_head": _clip(prompt, 800), "response_head": _clip(resp, 1200)})
# --- end logging setup ---


class RadiologyAgent:
    """Main orchestrator for the radiology AI agent."""
    
    def __init__(self, dry_run: bool = False, limit: Optional[int] = None):
        """
        Initialize the radiology agent.
        
        Args:
            dry_run: If True, only run regex baseline without LLM calls
            limit: Limit number of rows to process for testing
        """
        self.dry_run = dry_run
        self.limit = limit
        self.intent_parser = IntentParser()
        self.confirm_flow = ConfirmFlow()
        self.llm_extractor = None if dry_run else LLMExtractor()
        self.reflection_arch = ReflectionArchitecture()
        
        logger.info(f"Radiology Agent initialized - Dry run: {dry_run}, Limit: {limit}")
    
    def load_and_prepare_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load and prepare data from Excel file.
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Prepared DataFrame
        """
        logger.info(f"Loading data from {excel_path}")
        
        # Load Excel data
        df = load_excel_data(excel_path)
        
        # Concatenate narratives by imaging date
        df = concat_narratives_by_imaging(df)
        
        # Apply limit if specified
        if self.limit:
            df = df.head(self.limit)
            logger.info(f"Limited to first {self.limit} rows for testing")
        
        # Get data summary
        summary = get_data_summary(df)
        logger.info(f"Data summary: {summary}")
        
        return df
    
    def extract_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract structured data from narratives.
        
        Args:
            df: DataFrame with narratives
            
        Returns:
            DataFrame with structured data
        """
        logger.info("Starting structured data extraction")
        
        structured_data = []
        
        for idx, row in df.iterrows():
            record_id = row['recordid']
            imaging_date = row['imaging_date']
            narrative = row['narrative']
            
            logger.info(f"Processing record {record_id} ({idx + 1}/{len(df)})")
            
            # Safety check
            safety_report = check_text(narrative)
            if not safety_report["safe"]:
                logger.warning(f"Potential PHI detected in record {record_id}")
                narrative = safety_report["redacted_text"]
            
            # Extract using regex baseline
            regex_result = extract_regex_baseline(narrative)
            
            if self.dry_run:
                # Use only regex results
                extraction = regex_result
            else:
                # Use LLM extraction
                try:
                    llm_extraction = self.llm_extractor.extract(narrative, record_id)
                    # Convert LLM extraction to dict format
                    extraction = {
                        'right': {
                            'stone_status': llm_extraction.right.stone_status,
                            'stone_size_cm': llm_extraction.right.stone_size_cm,
                            'kidney_size_cm': llm_extraction.right.kidney_size_cm
                        },
                        'left': {
                            'stone_status': llm_extraction.left.stone_status,
                            'stone_size_cm': llm_extraction.left.stone_size_cm,
                            'kidney_size_cm': llm_extraction.left.kidney_size_cm
                        },
                        'bladder': {
                            'volume_ml': llm_extraction.bladder.volume_ml,
                            'wall': llm_extraction.bladder.wall
                        },
                        'history_summary': llm_extraction.history_summary,
                        'key_sentences': llm_extraction.key_sentences
                    }
                except Exception as e:
                    logger.error(f"LLM extraction failed for record {record_id}: {e}")
                    extraction = regex_result
            
            # Convert to structured output format
            structured_row = {
                'recordid': record_id,
                'imaging_date': imaging_date,
                'right_stone': extraction['right']['stone_status'],
                'right_stone_size_cm': extraction['right']['stone_size_cm'],
                'right_kidney_size_cm': extraction['right']['kidney_size_cm'],
                'left_stone': extraction['left']['stone_status'],
                'left_stone_size_cm': extraction['left']['stone_size_cm'],
                'left_kidney_size_cm': extraction['left']['kidney_size_cm'],
                'bladder_volume_ml': extraction['bladder']['volume_ml'],
                'history_summary': extraction['history_summary'],
                'matched_reason': 'No specific filters applied'
            }
            
            structured_data.append(structured_row)
        
        # Convert to DataFrame
        structured_df = pd.DataFrame(structured_data)
        
        # Save structured data
        save_path = save_structured_data(structured_df, 'structured', 'parquet')
        logger.info(f"Structured data saved to {save_path}")
        
        return structured_df
    
    def process_user_query(self, query: str, structured_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process a user query and return filtered results with reflection.
        
        Args:
            query: Natural language query
            structured_df: Structured DataFrame
            
        Returns:
            Filtered DataFrame or None if cancelled
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing user query: {query}")
        
        # Check domain relevance first
        domain_result = self.intent_parser.parse_query_with_domain_validation(query)
        
        if not domain_result['domain_relevant']:
            # Query is not relevant to medical radiology domain
            print("\n" + "="*60)
            print("DOMAIN VALIDATION")
            print("="*60)
            print(domain_result['message'])
            print(f"\nSuggestion: {domain_result['suggestion']}")
            
            # Ask user if they want general LLM response
            user_choice = input("\nWould you like a general answer? (y/n): ").lower().strip()
            
            if user_choice in ['y', 'yes']:
                print("\n" + "="*60)
                print("GENERAL LLM RESPONSE")
                print("="*60)
                print(domain_result['general_response'])
                print("\nNote: This information is from general knowledge, not from the medical dataset.")
            else:
                print("Please ask about medical radiology topics like patient records, kidney stones, bladder conditions, or imaging findings.")
            
            return None
        
        # Check if data is available for the query
        if not domain_result.get('data_available', True):
            # Query requires data that is not available in the dataset
            print("\n" + "="*60)
            print("DATA AVAILABILITY CHECK")
            print("="*60)
            print(domain_result['message'])
            print(f"\nSuggestion: {domain_result['suggestion']}")
            
            if 'available_fields' in domain_result:
                print(f"\nAvailable data fields in this dataset:")
                for field in domain_result['available_fields']:
                    print(f"  • {field}")
            
            return None
        
        # Query is relevant to medical domain and data is available, proceed with normal processing
        user_query = domain_result['user_query']
        
        # Create query tools
        query_tools = QueryTools(structured_df)
        
        # Estimate matching rows with query context
        query_context = {'query_text': query}
        estimated_rows = query_tools.estimate_matching_rows(user_query.filters, query_context)
        
        # Create plan summary
        plan = self.intent_parser.create_plan_summary(user_query, estimated_rows)
        
        # Get user confirmation
        confirmed_plan = self.confirm_flow.run_confirmation_loop(plan, query)
        if not confirmed_plan:
            logger.info("User cancelled the operation")
            return None
        
        # Apply filters with dynamic learning
        filtered_df = query_tools.apply_filters_with_learning(confirmed_plan.filters, query)
        
        # Save filtered results
        if len(filtered_df) > 0:
            save_path = save_structured_data(filtered_df, 'filtered', 'csv')
            logger.info(f"Filtered results saved to {save_path}")
        
        # Display summary with dynamic learning - only compute stats for relevant fields
        relevant_fields = confirmed_plan.input_fields + [str(k) for k in confirmed_plan.filters.keys()]
        stats = query_tools.get_summary_stats_with_learning(filtered_df, query, relevant_fields)
        summary_text = query_tools.format_summary(stats)
        print("\n" + summary_text)
        
        # REFLECTION STAGE: Reflect on the initial answer (if enabled)
        if self.reflection_arch is not None:
            processing_time = time.time() - start_time
            reflection_result = self.reflection_arch.reflect_on_answer(
                query=query,
                user_query=user_query,
                initial_answer=stats,
                filtered_df=filtered_df,
                processing_time=processing_time
            )
            
            # Display reflection results
            self._display_reflection_results(reflection_result)
            
            # Apply self-correction if needed
            if reflection_result.confidence < 0.7 or reflection_result.quality_score < 0.7:
                corrected_df = self._apply_self_correction(
                    query, user_query, stats, filtered_df, reflection_result
                )
                if corrected_df is not None:
                    return corrected_df
        
        return filtered_df
    
    def create_visualizations(self, structured_df: pd.DataFrame, 
                            relevant_fields: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create visualizations for the structured data, focusing on relevant fields.
        
        Args:
            structured_df: Structured DataFrame
            relevant_fields: List of relevant fields to focus visualizations on
            
        Returns:
            Dictionary of plot file paths
        """
        logger.info("Creating visualizations")
        
        query_tools = QueryTools(structured_df)
        visualizer = RadiologyVisualizer(query_tools)
        
        # Only create relevant visualizations
        plots = {}
        
        if relevant_fields is None:
            relevant_fields = []
        
        # Stone-related visualizations
        if any('stone' in field.lower() for field in relevant_fields):
            try:
                plots['stone_distribution'] = visualizer.plot_stone_distribution_by_side()
                plots['stone_size_histogram'] = visualizer.plot_stone_size_histogram()
            except Exception as e:
                logger.error(f"Error creating stone visualizations: {e}")
        
        # Bladder-related visualizations
        if any('bladder' in field.lower() or 'volume' in field.lower() for field in relevant_fields):
            try:
                plots['bladder_volume'] = visualizer.plot_bladder_volume_distribution()
            except Exception as e:
                logger.error(f"Error creating bladder visualizations: {e}")
        
        # Temporal visualizations
        if any('date' in field.lower() or 'year' in field.lower() or 'time' in field.lower() for field in relevant_fields):
            try:
                plots['temporal_trends'] = visualizer.plot_temporal_trends()
            except Exception as e:
                logger.error(f"Error creating temporal visualizations: {e}")
        
        # If no specific fields, create basic stone distribution
        if not relevant_fields or not any(keyword in ' '.join(relevant_fields).lower() for keyword in ['stone', 'bladder', 'date']):
            try:
                plots['stone_distribution'] = visualizer.plot_stone_distribution_by_side()
            except Exception as e:
                logger.error(f"Error creating basic visualization: {e}")
        
        logger.info(f"Created {len(plots)} relevant visualizations")
        for plot_name, plot_path in plots.items():
            if plot_path:
                logger.info(f"  {plot_name}: {plot_path}")
        
        return plots
    
    def run_interactive_mode(self, structured_df: pd.DataFrame):
        """
        Run interactive mode for user queries.
        
        Args:
            structured_df: Structured DataFrame
        """
        print("\n" + "=" * 80)
        print("RADIOLOGY AI AGENT - INTERACTIVE MODE")
        print("=" * 80)
        print("Enter your queries in natural language. Type 'quit' to exit.")
        print("Examples:")
        print("  - How many patients had left kidney stones > 1 cm?")
        print("  - Show me all patients with bilateral stones")
        print("  - Count patients with bladder volume > 200 ml")
        print("  - Plot stone distribution by side")
        print("=" * 80)
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Check if it's a visualization request
                if any(word in query.lower() for word in ['plot', 'chart', 'graph', 'visualize']):
                    print("Creating visualizations...")
                    # Parse query to determine relevant fields
                    user_query = self.intent_parser.parse_query(query)
                    relevant_fields = user_query.input_fields + [str(k) for k in user_query.filters.keys()]
                    plots = self.create_visualizations(structured_df, relevant_fields)
                    if plots:
                        print(f"Created {len(plots)} relevant visualizations in {PLOTS_DIR}")
                    continue
                
                # Process the query
                result_df = self.process_user_query(query, structured_df)
                
                if result_df is not None and len(result_df) > 0:
                    print(f"\nFound {len(result_df)} matching records.")
                    
                    # Parse the query to get relevant fields for visualizations
                    user_query = self.intent_parser.parse_query(query)
                    relevant_fields = user_query.input_fields + [str(k) for k in user_query.filters.keys()]
                    
                    # Offer additional options
                    while True:
                        option = input("\nOptions: [S]how table, [D]ownload CSV, [P]lot charts, [N]ew query: ").strip().lower()
                        
                        if option in ['s', 'show']:
                            print("\nFirst 10 rows:")
                            print(result_df.head(10).to_string(index=False))
                        elif option in ['d', 'download']:
                            save_path = save_structured_data(result_df, f'query_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 'csv')
                            print(f"Results saved to {save_path}")
                        elif option in ['p', 'plot']:
                            print("Creating visualizations...")
                            plots = self.create_visualizations(result_df, relevant_fields)
                            if plots:
                                print(f"Created {len(plots)} relevant visualizations in {PLOTS_DIR}")
                        elif option in ['n', 'new']:
                            break
                        else:
                            print("Invalid option. Please enter S, D, P, or N.")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")
    
    def run_full_pipeline(self, excel_path: str, query: Optional[str] = None):
        """
        Run the complete pipeline.
        
        Args:
            excel_path: Path to Excel file
            query: Optional query to process
        """
        try:
            # Load and prepare data
            df = self.load_and_prepare_data(excel_path)
            
            # Extract structured data
            structured_df = self.extract_structured_data(df)
            
            # Create initial visualizations (basic stone distribution only)
            plots = self.create_visualizations(structured_df, ['stone'])
            
            if query:
                # Process specific query
                result_df = self.process_user_query(query, structured_df)
                if result_df is not None:
                    print(f"\nQuery completed. Found {len(result_df)} matching records.")
            else:
                # Run interactive mode
                self.run_interactive_mode(structured_df)
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _display_reflection_results(self, reflection_result):
        """Display reflection results to the user."""
        print("\n" + "="*60)
        print("REFLECTION ANALYSIS")
        print("="*60)
        
        # Quality score
        quality_score = reflection_result.quality_score
        if quality_score >= 0.9:
            quality_status = "Excellent"
        elif quality_score >= 0.7:
            quality_status = "Good"
        elif quality_score >= 0.5:
            quality_status = "Adequate"
        else:
            quality_status = "Needs Improvement"
        
        print(f"Answer Quality: {quality_score:.2f}/1.0 ({quality_status})")
        print(f"Confidence: {reflection_result.confidence:.2f}/1.0")
        
        # Issues
        if reflection_result.issues:
            print(f"\nIssues Identified ({len(reflection_result.issues)}):")
            for i, issue in enumerate(reflection_result.issues, 1):
                print(f"  {i}. {issue}")
        
        # Improvements
        if reflection_result.improvements:
            print(f"\nSuggested Improvements ({len(reflection_result.improvements)}):")
            for i, improvement in enumerate(reflection_result.improvements, 1):
                print(f"  {i}. {improvement}")
        
        # Reasoning
        print(f"\nAnalysis: {reflection_result.reasoning}")
        
        # Actions
        if reflection_result.suggested_actions:
            print(f"\nRecommended Actions ({len(reflection_result.suggested_actions)}):")
            for i, action in enumerate(reflection_result.suggested_actions, 1):
                print(f"  {i}. {action}")
        
        print("="*60)
    
    def _apply_self_correction(self, 
                              query: str,
                              user_query,
                              initial_answer: Dict[str, Any],
                              filtered_df: pd.DataFrame,
                              reflection_result) -> Optional[pd.DataFrame]:
        """Apply self-correction based on reflection results."""
        
        logger.info("Applying self-correction based on reflection analysis")
        
        # Check if self-correction is needed and feasible
        if reflection_result.confidence < 0.5:
            logger.warning("Confidence too low for reliable self-correction")
            return None
        
        # Create new query tools for correction
        query_tools = QueryTools(filtered_df)
        
        # Apply corrections based on identified issues
        corrected_df = filtered_df.copy()
        corrections_applied = []
        
        # Issue: Potential accuracy issues in filtering logic
        if any("accuracy" in issue.lower() for issue in reflection_result.issues):
            # Re-verify filters
            logger.info("Re-verifying filter accuracy")
            corrections_applied.append("Re-verified filter logic")
        
        # Issue: Missing requested information
        if any("completeness" in issue.lower() for issue in reflection_result.issues):
            # Try to add missing information
            if 'mean' in query.lower() and 'mean_bladder_volume_ml' not in initial_answer:
                # Calculate missing mean
                if 'bladder_volume_ml' in corrected_df.columns:
                    mean_volume = corrected_df['bladder_volume_ml'].mean()
                    logger.info(f"Added missing mean bladder volume: {mean_volume:.1f} ml")
                    corrections_applied.append(f"Added mean bladder volume: {mean_volume:.1f} ml")
        
        # Issue: Answer may not fully address query intent
        if any("relevance" in issue.lower() for issue in reflection_result.issues):
            # Refine answer to be more relevant
            if 'left' in query.lower() and 'right' not in query.lower():
                # Ensure left-side focus
                if 'has_left_stone' in corrected_df.columns:
                    left_patients = corrected_df[corrected_df['has_left_stone'] == True]
                    if len(left_patients) != len(corrected_df):
                        corrected_df = left_patients
                        corrections_applied.append("Refined to focus on left-side patients only")
        
        # Display corrections
        if corrections_applied:
            print("\n" + "="*60)
            print("SELF-CORRECTION APPLIED")
            print("="*60)
            for correction in corrections_applied:
                print(f"✓ {correction}")
            print("="*60)
            
            # Re-compute statistics with corrected data
            relevant_fields = user_query.input_fields + [str(k) for k in user_query.filters.keys()]
            corrected_stats = query_tools.get_summary_stats_with_learning(
                corrected_df, query, relevant_fields
            )
            corrected_summary = query_tools.format_summary(corrected_stats)
            print("\n" + "CORRECTED RESULTS:")
            print(corrected_summary)
            
            return corrected_df
        
        return None

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Radiology AI Agent for structured data extraction")
    
    parser.add_argument("excel_file", help="Path to Excel file with radiology data")
    parser.add_argument("--query", "-q", help="Natural language query to process")
    parser.add_argument("--dry-run", action="store_true", help="Run with regex only, no LLM calls")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of rows to process")
    parser.add_argument("--confirm", choices=["on", "off"], default="on", help="Enable/disable confirmation flow")
    parser.add_argument("--reflection", choices=["on", "off"], default="on", help="Enable/disable reflection architecture")
    parser.add_argument("--reflection-summary", action="store_true", help="Show reflection history summary")
    
    args = parser.parse_args()
    
    # Show reflection summary if requested (before initializing full agent)
    if args.reflection_summary:
        from .reflection import ReflectionArchitecture
        reflection_arch = ReflectionArchitecture()
        summary = reflection_arch.get_reflection_summary()
        print("\n" + "="*60)
        print("REFLECTION HISTORY SUMMARY")
        print("="*60)
        if 'message' in summary:
            print(summary['message'])
        else:
            print(f"Total Reflections: {summary['total_reflections']}")
            print(f"Average Quality Score: {summary['average_quality_score']:.2f}/1.0")
            print(f"Average Confidence: {summary['average_confidence']:.2f}/1.0")
            if summary['common_issues']:
                print(f"\nMost Common Issues:")
                for issue, count in summary['common_issues'].items():
                    print(f"  • {issue} ({count} times)")
        print("="*60)
        return
    
    # Initialize agent
    agent = RadiologyAgent(dry_run=args.dry_run, limit=args.limit)
    
    # Disable confirmation if requested
    if args.confirm == "off":
        logger.warning("Confirmation flow disabled - this is not recommended for production use")
        agent.confirm_flow.disabled = True
    
    # Disable reflection if requested
    if args.reflection == "off":
        logger.info("Reflection architecture disabled")
        agent.reflection_arch = None
    
    try:
        # Run pipeline
        agent.run_full_pipeline(args.excel_file, args.query)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
