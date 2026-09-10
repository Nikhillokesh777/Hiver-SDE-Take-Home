"""
Tests for Stage 3: Intent Classification & Prototype Calibration.
Verifies taxonomy conformance, score calibration, known intent accuracy, and OOD fallback.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.intent_classifier import IntentClassifier, FROZEN_INTENTS, IntentClassificationResult


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


def test_intent_output_schema(classifier):
    """Verify output adheres to IntentClassificationResult schema and frozen taxonomy."""
    result = classifier.classify("I was charged a $5 cancellation fee unfairly.")
    
    assert isinstance(result, IntentClassificationResult)
    assert result.intent in FROZEN_INTENTS
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.all_scores, dict)
    assert len(result.all_scores) >= 8


def test_intent_cancellation_fee_dispute(classifier):
    """Verify cancellation dispute inquiry classifies into Cancellation_Fee_Dispute or Fare_Dispute."""
    result = classifier.classify("My driver never showed up and I got hit with a cancellation fee refund request.")
    assert result.intent in ["Cancellation_Fee_Dispute", "Fare_Dispute_Or_Refund"]
    assert result.confidence > 0.30


def test_intent_lost_item_inquiry(classifier):
    """Verify lost item message classifies into Lost_Item_Inquiry."""
    result = classifier.classify("I left my purse and iPhone in the back seat of the car.")
    assert result.intent == "Lost_Item_Inquiry"
    assert result.confidence > 0.35


def test_intent_empty_or_whitespace_fallback(classifier):
    """Verify empty input gracefully falls back to Other_Or_Unclear with zero confidence."""
    result = classifier.classify("    \n\t   ")
    assert result.intent == "Other_Or_Unclear"
    assert result.confidence == 0.0
    assert result.is_fallback is True


def test_intent_out_of_distribution_fallback(classifier):
    """Verify completely un-related domain queries trigger Other_Or_Unclear fallback."""
    result = classifier.classify("What is the recipe for chocolate chip banana bread?")
    assert result.intent == "Other_Or_Unclear"
    assert result.is_fallback is True
