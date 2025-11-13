#!/usr/bin/env python3
"""Test script to verify case sensitivity fixes in confirmation flow."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.confirm_flow import ConfirmFlow, ConfirmationAction
from src.llm_schema import PlanSummary
import unittest.mock

def test_case_sensitivity():
    """Test that confirmation flow accepts various case inputs."""
    print("🧪 Testing Case Sensitivity in Confirmation Flow")
    print("=" * 60)
    
    # Create a test plan
    plan = PlanSummary(
        goal="Test goal",
        input_fields=["narrative"],
        filters={"side": "right"},
        outputs=["recordid"],
        assumptions=["Test assumption"]
    )
    
    flow = ConfirmFlow()
    
    # Test cases for different inputs
    test_cases = [
        # YES variations
        ("yes", ConfirmationAction.YES),
        ("YES", ConfirmationAction.YES),
        ("Yes", ConfirmationAction.YES),
        ("y", ConfirmationAction.YES),
        ("Y", ConfirmationAction.YES),
        ("proceed", ConfirmationAction.YES),
        ("PROCEED", ConfirmationAction.YES),
        ("go", ConfirmationAction.YES),
        ("GO", ConfirmationAction.YES),
        
        # EDIT variations
        ("edit", ConfirmationAction.EDIT),
        ("EDIT", ConfirmationAction.EDIT),
        ("Edit", ConfirmationAction.EDIT),
        ("e", ConfirmationAction.EDIT),
        ("E", ConfirmationAction.EDIT),
        ("modify", ConfirmationAction.EDIT),
        ("MODIFY", ConfirmationAction.EDIT),
        ("change", ConfirmationAction.EDIT),
        ("CHANGE", ConfirmationAction.EDIT),
        
        # CANCEL variations
        ("cancel", ConfirmationAction.CANCEL),
        ("CANCEL", ConfirmationAction.CANCEL),
        ("Cancel", ConfirmationAction.CANCEL),
        ("c", ConfirmationAction.CANCEL),
        ("C", ConfirmationAction.CANCEL),
        ("no", ConfirmationAction.CANCEL),
        ("NO", ConfirmationAction.CANCEL),
        ("quit", ConfirmationAction.CANCEL),
        ("QUIT", ConfirmationAction.CANCEL),
        ("exit", ConfirmationAction.CANCEL),
        ("EXIT", ConfirmationAction.CANCEL),
    ]
    
    print("Testing various case inputs...")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for input_text, expected_action in test_cases:
        # Mock the input to return our test input
        with unittest.mock.patch('builtins.input', return_value=input_text):
            try:
                result = flow.get_user_confirmation(plan, "test query")
                if result == expected_action:
                    print(f"✅ '{input_text}' -> {result.value} (PASS)")
                    passed += 1
                else:
                    print(f"❌ '{input_text}' -> {result.value}, expected {expected_action.value} (FAIL)")
                    failed += 1
            except Exception as e:
                print(f"❌ '{input_text}' -> ERROR: {e} (FAIL)")
                failed += 1
    
    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All case sensitivity tests PASSED!")
        print("The confirmation flow now properly handles:")
        print("  • Uppercase: YES, EDIT, CANCEL")
        print("  • Lowercase: yes, edit, cancel") 
        print("  • Mixed case: Yes, Edit, Cancel")
        print("  • Short forms: y, e, c")
        print("  • Alternative words: proceed, modify, quit, etc.")
    else:
        print("❌ Some tests failed - case sensitivity issue not fully resolved")
    
    return failed == 0

if __name__ == "__main__":
    success = test_case_sensitivity()
    sys.exit(0 if success else 1)

