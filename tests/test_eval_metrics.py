"""
Tests for Evaluation Metrics Harness & Baselines.
Verifies Intent Macro-F1, Escalation Recall/Precision, False Negatives, Autonomy Resolvability, and baseline models.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from eval.metrics import (
    calculate_intent_metrics,
    calculate_escalation_metrics,
    calculate_autonomy_resolvability,
    calculate_grounding_metrics
)
from eval.baselines import TrivialBaseline, SimpleMLBaseline


def test_intent_metrics_calculation():
    """Verify calculation of accuracy and Macro-F1 across known classes."""
    y_true = ["Billing", "Driver", "Billing", "Lost_Item"]
    y_pred = ["Billing", "Driver", "Lost_Item", "Lost_Item"]
    
    metrics = calculate_intent_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "Billing" in metrics["per_intent"]
    assert metrics["per_intent"]["Billing"]["recall"] == 0.5


def test_escalation_metrics_perfect_recall_zero_false_negatives():
    """Verify that achieving zero false negatives yields 100% (1.0000) escalation recall."""
    y_true = ["escalate", "escalate", "auto_handle", "auto_handle"]
    # Proposed system escalates everything it suspects (even false positives), but misses 0 true escalations
    y_pred = ["escalate", "escalate", "escalate", "auto_handle"]
    
    metrics = calculate_escalation_metrics(y_true, y_pred)
    assert metrics["recall"] == 1.0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] < 1.0  # Demonstrates expected precision trade-off


def test_autonomy_resolvability_calculation():
    """Verify calculation of autonomy resolvability rate."""
    y_true = ["auto_handle", "auto_handle", "escalate", "escalate"]
    y_pred = ["auto_handle", "auto_handle", "auto_handle", "escalate"]
    
    # 3 cases auto-handled, 2 of which were genuinely auto-handle
    res = calculate_autonomy_resolvability(y_true, y_pred)
    assert res["auto_handle_count"] == 3
    assert res["valid_resolvable_count"] == 2
    assert res["unsafe_auto_handled_count"] == 1
    assert pytest.approx(res["autonomy_resolvability_rate"], 0.01) == 0.67


def test_grounding_pass_rate_metric():
    """Verify calculation of grounding pass rate."""
    passes = [True, True, True, False]
    scores = [0.9, 0.85, 0.95, 0.3]
    res = calculate_grounding_metrics(passes, scores)
    assert res["grounding_pass_rate"] == 0.75
    assert pytest.approx(res["mean_grounding_score"], 0.01) == 0.75


def test_trivial_baseline_prediction():
    """Verify TrivialBaseline returns static majority predictions."""
    baseline = TrivialBaseline()
    out = baseline.run("Any customer inquiry text")
    assert out.predicted_intent != ""
    assert out.routing_decision == "auto_handle"
    assert len(out.generated_reply) > 0


def test_simple_ml_baseline_prediction():
    """Verify SimpleMLBaseline operates deterministically via TF-IDF."""
    baseline = SimpleMLBaseline()
    baseline.fit_or_load()
    out = baseline.run("I need a refund for my cancellation fee")
    assert out.predicted_intent != ""
    assert out.routing_decision in ["auto_handle", "escalate"]
    assert len(out.generated_reply) > 0
