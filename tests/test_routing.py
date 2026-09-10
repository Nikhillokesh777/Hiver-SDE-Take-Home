"""
Tests for Stage 7 & 8: Routing Engine & Confidence Estimation.
Verifies safety escalation triggers, intent-based policies, auto-handle thresholds, and composite confidence math.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.routing import RoutingEngine, RoutingDecision


@pytest.fixture(scope="module")
def engine():
    return RoutingEngine(auto_handle_threshold=0.70)


def test_routing_safety_keyword_escalation(engine):
    """Verify safety keywords (assault, crash, weapon) immediately trigger ESCALATE regardless of intent confidence."""
    decision = engine.decide(
        intent="Cancellation_Fee_Dispute",
        intent_confidence=0.95,
        retrieval_similarity=0.90,
        grounding_passed=True,
        grounding_score=0.95,
        text="The driver threatened me with assault when I disputed the fee!"
    )
    assert isinstance(decision, RoutingDecision)
    assert decision.action == "escalate"
    assert "safety" in decision.reason.lower()


def test_routing_driver_safety_intent_escalation(engine):
    """Verify Driver_Behavior_Or_Safety intent always triggers human escalation."""
    decision = engine.decide(
        intent="Driver_Behavior_Or_Safety",
        intent_confidence=0.85,
        retrieval_similarity=0.80,
        grounding_passed=True,
        grounding_score=0.90,
        text="The driver was driving erratically and refused to let me exit."
    )
    assert decision.action == "escalate"
    assert "driver_behavior_or_safety" in decision.reason.lower()


def test_routing_auto_handle_on_high_confidence(engine):
    """Verify routine intent with high composite confidence (>0.70) triggers AUTO_HANDLE."""
    decision = engine.decide(
        intent="Cancellation_Fee_Dispute",
        intent_confidence=0.90,
        retrieval_similarity=0.85,
        grounding_passed=True,
        grounding_score=0.95,
        text="My driver cancelled on me, can I get my cancellation fee refunded?"
    )
    assert decision.action == "auto_handle"
    assert decision.composite_confidence >= 0.70


def test_routing_escalates_on_low_confidence(engine):
    """Verify low composite confidence falls back to safe human escalation."""
    decision = engine.decide(
        intent="Lost_Item_Inquiry",
        intent_confidence=0.20,
        retrieval_similarity=0.50,
        grounding_passed=True,
        grounding_score=0.80,
        text="I think maybe I left something somewhere last week."
    )
    assert decision.action == "escalate"
    assert "below auto-handle threshold" in decision.reason.lower()


def test_routing_composite_confidence_formula(engine):
    """Verify composite confidence formula: 0.30*intent + 0.35*retrieval + 0.35*grounding."""
    conf = engine.compute_composite_confidence(
        intent_conf=1.0,
        retrieval_sim=1.0,
        grounding_score=1.0
    )
    assert pytest.approx(conf, 0.001) == 1.0

    conf_half = engine.compute_composite_confidence(
        intent_conf=0.5,
        retrieval_sim=0.5,
        grounding_score=0.5
    )
    assert pytest.approx(conf_half, 0.001) == 0.5
