"""User confirmation workflow for plan execution."""

import sys
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
import pandas as pd

from .llm_schema import PlanSummary, UserQuery
from .intent_parser import IntentParser

logger = logging.getLogger(__name__)

class ConfirmationAction(Enum):
    """User confirmation actions."""
    YES = "yes"
    EDIT = "edit"
    CANCEL = "cancel"

class ConfirmFlow:
    """Handle user confirmation workflow."""
    
    def __init__(self):
        self.intent_parser = IntentParser()
        self.disabled = False
        self.structured_df: Optional[pd.DataFrame] = None
    
    def get_user_confirmation(self, plan: PlanSummary, 
                            original_query: str = "") -> ConfirmationAction:
        """
        Get user confirmation for the plan.
        
        Args:
            plan: PlanSummary object
            original_query: Original user query for context
            
        Returns:
            ConfirmationAction enum value
        """
        # Display plan summary
        self._display_plan_summary(plan, original_query)
        
        # Get user input
        while True:
            try:
                response = input("\nDo you want me to proceed? (yes/edit/cancel or y/e/c): ").strip().lower()
                
                if response in ['yes', 'y', 'proceed', 'go']:
                    return ConfirmationAction.YES
                elif response in ['edit', 'e', 'modify', 'change']:
                    return ConfirmationAction.EDIT
                elif response in ['cancel', 'c', 'no', 'quit', 'exit']:
                    return ConfirmationAction.CANCEL
                else:
                    print("Please enter 'yes', 'edit', or 'cancel' (or 'y', 'e', 'c')")
                    continue
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return ConfirmationAction.CANCEL
            except EOFError:
                print("\nOperation cancelled.")
                return ConfirmationAction.CANCEL
    
    def _display_plan_summary(self, plan: PlanSummary, original_query: str = ""):
        """Display the plan summary to the user."""
        print("\n" + "=" * 80)
        print("RADIOLOGY AI AGENT - EXTRACTION PLAN")
        print("=" * 80)
        
        if original_query:
            print(f"Original Query: {original_query}")
            print("-" * 80)
        
        print(f"Goal: {plan.goal}")
        print()
        
        print("Input fields detected:")
        for field in plan.input_fields:
            print(f"  • {field}")
        print()
        
        print("Filters:")
        if plan.filters:
            for key, value in plan.filters.items():
                print(f"  • {key}: {value}")
        else:
            print("  • No specific filters applied")
        print()
        
        print("Outputs:")
        for output in plan.outputs:
            print(f"  • {output}")
        print()
        
        print("Assumptions:")
        for assumption in plan.assumptions:
            print(f"  • {assumption}")
        print()
        
        if plan.estimated_rows:
            print(f"Estimated matching rows: {plan.estimated_rows}")
        
        print(f"Estimated processing time: {plan.processing_time_estimate}")
        print("=" * 80)
    
    def handle_edit_request(self, plan: PlanSummary, 
                          original_query: str = "") -> Optional[PlanSummary]:
        """
        Handle user request to edit the plan.
        
        Args:
            plan: Current plan
            original_query: Original user query
            
        Returns:
            Updated plan or None if cancelled
        """
        print("\n" + "=" * 60)
        print("EDIT PLAN")
        print("=" * 60)
        
        print("Current filters:")
        if plan.filters:
            for i, (key, value) in enumerate(plan.filters.items(), 1):
                print(f"  {i}. {key}: {value}")
        else:
            print("  No filters currently applied")
        
        print("\nOptions:")
        print("1. Modify existing filter")
        print("2. Add new filter")
        print("3. Remove filter")
        print("4. Start over with new query")
        print("5. Cancel editing")
        
        while True:
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    return self._modify_existing_filter(plan)
                elif choice == '2':
                    return self._add_new_filter(plan)
                elif choice == '3':
                    return self._remove_filter(plan)
                elif choice == '4':
                    return self._start_over()
                elif choice == '5':
                    return None
                else:
                    print("Please enter a number between 1 and 5")
                    
            except KeyboardInterrupt:
                print("\nEditing cancelled.")
                return None
            except EOFError:
                print("\nEditing cancelled.")
                return None
    
    def _modify_existing_filter(self, plan: PlanSummary) -> PlanSummary:
        """Modify an existing filter."""
        if not plan.filters:
            print("No filters to modify.")
            return plan
        
        print("\nCurrent filters:")
        filter_items = list(plan.filters.items())
        for i, (key, value) in enumerate(filter_items, 1):
            print(f"  {i}. {key}: {value}")
        
        while True:
            try:
                choice = int(input(f"\nSelect filter to modify (1-{len(filter_items)}): ")) - 1
                if 0 <= choice < len(filter_items):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(filter_items)}")
            except ValueError:
                print("Please enter a valid number")
        
        selected_key = filter_items[choice][0]
        current_value = filter_items[choice][1]
        
        print(f"\nCurrent value for '{selected_key}': {current_value}")
        new_value = input(f"Enter new value for '{selected_key}': ").strip()
        
        if new_value:
            # Update the filter
            updated_filters = plan.filters.copy()
            updated_filters[selected_key] = new_value
            
            # Create new plan with updated filters
            return PlanSummary(
                goal=plan.goal,
                input_fields=plan.input_fields,
                filters=updated_filters,
                outputs=plan.outputs,
                assumptions=plan.assumptions,
                estimated_rows=plan.estimated_rows,
                processing_time_estimate=plan.processing_time_estimate
            )
        else:
            print("No changes made.")
            return plan
    
    def _add_new_filter(self, plan: PlanSummary) -> PlanSummary:
        """Add a new filter."""
        print("\nAvailable filter types:")
        print("1. side (left, right, bilateral)")
        print("2. min_size_cm (minimum stone size in cm)")
        print("3. max_size_cm (maximum stone size in cm)")
        print("4. start_year (start year for date range)")
        print("5. end_year (end year for date range)")
        print("6. stone_presence (present, absent, unclear)")
        print("7. min_bladder_volume_ml (minimum bladder volume)")
        print("8. max_bladder_volume_ml (maximum bladder volume)")
        
        while True:
            try:
                choice = input("\nSelect filter type (1-8): ").strip()
                filter_map = {
                    '1': 'side',
                    '2': 'min_size_cm',
                    '3': 'max_size_cm',
                    '4': 'start_year',
                    '5': 'end_year',
                    '6': 'stone_presence',
                    '7': 'min_bladder_volume_ml',
                    '8': 'max_bladder_volume_ml'
                }
                
                if choice in filter_map:
                    filter_key = filter_map[choice]
                    break
                else:
                    print("Please enter a number between 1 and 8")
            except (KeyboardInterrupt, EOFError):
                print("\nAdding filter cancelled.")
                return plan
        
        # Get filter value
        if filter_key == 'side':
            print("Enter side: left, right, or bilateral")
        elif filter_key in ['min_size_cm', 'max_size_cm']:
            print("Enter size in centimeters (e.g., 1.5)")
        elif filter_key in ['start_year', 'end_year']:
            print("Enter year (e.g., 2020)")
        elif filter_key == 'stone_presence':
            print("Enter stone presence: present, absent, or unclear")
        elif filter_key in ['min_bladder_volume_ml', 'max_bladder_volume_ml']:
            print("Enter volume in milliliters (e.g., 200)")
        
        filter_value = input(f"Enter value for {filter_key}: ").strip()
        
        if filter_value:
            # Validate and convert value
            try:
                if filter_key in ['min_size_cm', 'max_size_cm', 'min_bladder_volume_ml', 'max_bladder_volume_ml']:
                    filter_value = float(filter_value)
                elif filter_key in ['start_year', 'end_year']:
                    filter_value = int(filter_value)
                
                # Update the filters
                updated_filters = plan.filters.copy()
                updated_filters[filter_key] = filter_value
                
                # Create new plan with updated filters
                return PlanSummary(
                    goal=plan.goal,
                    input_fields=plan.input_fields,
                    filters=updated_filters,
                    outputs=plan.outputs,
                    assumptions=plan.assumptions,
                    estimated_rows=plan.estimated_rows,
                    processing_time_estimate=plan.processing_time_estimate
                )
            except ValueError:
                print("Invalid value format. No changes made.")
                return plan
        else:
            print("No value entered. No changes made.")
            return plan
    
    def _remove_filter(self, plan: PlanSummary) -> PlanSummary:
        """Remove a filter."""
        if not plan.filters:
            print("No filters to remove.")
            return plan
        
        print("\nCurrent filters:")
        filter_items = list(plan.filters.items())
        for i, (key, value) in enumerate(filter_items, 1):
            print(f"  {i}. {key}: {value}")
        
        while True:
            try:
                choice = int(input(f"\nSelect filter to remove (1-{len(filter_items)}): ")) - 1
                if 0 <= choice < len(filter_items):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(filter_items)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\nRemoving filter cancelled.")
                return plan
        
        # Remove the selected filter
        selected_key = filter_items[choice][0]
        updated_filters = plan.filters.copy()
        del updated_filters[selected_key]
        
        print(f"Removed filter: {selected_key}")
        
        # Create new plan with updated filters
        return PlanSummary(
            goal=plan.goal,
            input_fields=plan.input_fields,
            filters=updated_filters,
            outputs=plan.outputs,
            assumptions=plan.assumptions,
            estimated_rows=plan.estimated_rows,
            processing_time_estimate=plan.processing_time_estimate
        )
    
    def _start_over(self) -> Optional[PlanSummary]:
        """Start over with a new query."""
        print("\nEnter a new query:")
        try:
            new_query = input("> ").strip()
            if new_query:
                # Use schema-aware parsing if structured_df is available
                user_query = self.intent_parser.parse_query(new_query, self.structured_df)
                return self.intent_parser.create_plan_summary(user_query)
            else:
                print("No query entered.")
                return None
        except (KeyboardInterrupt, EOFError):
            print("\nStarting over cancelled.")
            return None
    
    def run_confirmation_loop(self, initial_plan: PlanSummary, 
                            original_query: str = "",
                            structured_df: Optional[pd.DataFrame] = None) -> Optional[PlanSummary]:
        """
        Run the complete confirmation loop until user confirms or cancels.
        
        Args:
            initial_plan: Initial plan to confirm
            original_query: Original user query
            structured_df: Optional structured DataFrame for schema-aware parsing
            
        Returns:
            Final confirmed plan or None if cancelled
        """
        # Skip confirmation if disabled
        if self.disabled:
            logger.info("Confirmation flow disabled - auto-proceeding")
            return initial_plan
            
        current_plan = initial_plan
        self.structured_df = structured_df  # Store for use in edit/start_over
        
        while True:
            action = self.get_user_confirmation(current_plan, original_query)
            
            if action == ConfirmationAction.YES:
                print("\n✓ Proceeding with extraction...")
                return current_plan
            elif action == ConfirmationAction.EDIT:
                edited_plan = self.handle_edit_request(current_plan, original_query)
                if edited_plan:
                    current_plan = edited_plan
                    print("\n✓ Plan updated. Review the changes above.")
                else:
                    print("\nEditing cancelled. Returning to confirmation.")
            elif action == ConfirmationAction.CANCEL:
                print("\n✗ Operation cancelled by user.")
                return None

