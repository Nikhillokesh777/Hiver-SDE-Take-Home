"""
Tests for Stage 6: Grounding Verification Module.
Verifies regex hallucination detection (unsupported phone numbers, fake dollar amounts) and grounded claim passing.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.grounding_check import GroundingChecker, GroundingResult


@pytest.fixture(scope="module")
def checker():
    return GroundingChecker()


def test_grounding_catches_unsupported_phone_numbers(checker):
    """Verify regex heuristics detect fabricated customer support phone numbers not in evidence."""
    evidence = "[Historical Case #1]\nResolution: Send us a DM with your registered email."
    hallucinated_reply = "Please call our 24/7 billing support line at 1-800-555-0199 right now."
    
    result = checker.check(hallucinated_reply, evidence, predicted_intent="Fare_Dispute_Or_Refund")
    assert isinstance(result, GroundingResult)
    assert result.grounding_pass is False
    assert result.grounding_score < 0.50
    assert any("phone" in claim.lower() for claim in result.unsupported_claims)


def test_grounding_catches_unsupported_dollar_refunds(checker):
    """Verify regex heuristics detect invented dollar refund guarantees not found in evidence."""
    evidence = "[Historical Case #2]\nResolution: Visit in-app Help to submit a fare review."
    hallucinated_reply = "We are very sorry! We have issued an immediate $150 credit to your Uber balance."
    
    result = checker.check(hallucinated_reply, evidence, predicted_intent="Fare_Dispute_Or_Refund")
    assert result.grounding_pass is False
    assert any("$" in claim for claim in result.unsupported_claims)


def test_grounding_passes_verified_official_reply(checker):
    """Verify compliant reply directing user to official Help links passes grounding check."""
    evidence = "[Historical Case #3]\nResolution: Please send us a DM with your trip details so our team can assist: https://t.co/help"
    grounded_reply = "Hi there, we'd like to help. Please send us a DM with your trip details so our team can assist: https://t.co/help"
    
    result = checker.check(grounded_reply, evidence, predicted_intent="Fare_Dispute_Or_Refund")
    assert result.grounding_pass is True
    assert result.grounding_score >= 0.70
    assert len(result.unsupported_claims) == 0
