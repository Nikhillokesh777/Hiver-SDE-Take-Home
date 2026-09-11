"""
Evaluation Metrics Module.
Calculates headline and supporting metrics for support agent evaluation:
1. Intent Classification: Accuracy and Macro-F1.
2. Escalation Decision: Precision, Recall, F1, and False Negative Rate (safety penalty).
3. Grounding Rate: % of replies with zero unsupported factual claims.
4. Autonomy Resolvability: % of auto-handled cases that were genuinely resolvable.
5. Latency & Cost profile per system.
6. Segmented metrics by intent and edge-case category.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score


def calculate_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes overall Accuracy and Macro-F1 across frozen intents."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-intent metrics
    labels = sorted(list(set(y_true + y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_intent = {}
    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }
        
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_intent": per_intent
    }


def calculate_escalation_metrics(y_true_actions: List[str], y_pred_actions: List[str]) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1, and critical False Negative counts for escalation decisions.
    Escalate is treated as the positive class (1), auto_handle as (0).
    """
    true_binary = [1 if a == "escalate" else 0 for a in y_true_actions]
    pred_binary = [1 if a == "escalate" else 0 for a in y_pred_actions]

    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_binary) if len(true_binary) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,  # CRITICAL: missed escalations (e.g. safety/legal issues auto-handled)
        "true_negatives": tn
    }


def calculate_autonomy_resolvability(y_true_actions: List[str], y_pred_actions: List[str]) -> Dict[str, Any]:
    """
    Evaluates trustworthiness of autonomous actions:
    Of the cases the system chose to auto-handle, what fraction was actually gold-labelled as auto-handle?
    """
    pred_auto_total = sum(1 for p in y_pred_actions if p == "auto_handle")
    if pred_auto_total == 0:
        return {"auto_handle_count": 0, "autonomy_resolvability_rate": 0.0}

    valid_auto = sum(1 for t, p in zip(y_true_actions, y_pred_actions) if p == "auto_handle" and t == "auto_handle")
    rate = valid_auto / pred_auto_total

    return {
        "auto_handle_count": pred_auto_total,
        "valid_resolvable_count": valid_auto,
        "unsafe_auto_handled_count": pred_auto_total - valid_auto,
        "autonomy_resolvability_rate": round(rate, 4)
    }


def calculate_grounding_metrics(grounding_passes: List[bool], grounding_scores: List[float]) -> Dict[str, Any]:
    """Calculates factual grounding rate and mean confidence."""
    if not grounding_passes:
        return {"grounding_pass_rate": 0.0, "mean_grounding_score": 0.0}

    pass_rate = sum(1 for g in grounding_passes if g) / len(grounding_passes)
    mean_score = float(np.mean(grounding_scores)) if grounding_scores else 0.0

    return {
        "grounding_pass_rate": round(pass_rate, 4),
        "mean_grounding_score": round(mean_score, 4)
    }


def evaluate_system_performance(
    predictions: List[Dict[str, Any]],
    golden_records: List[Dict[str, Any]],
    system_name: str = "proposed_system"
) -> Dict[str, Any]:
    """
    Aggregates all evaluation dimensions into a unified, reproducible report.
    Matches prediction records against gold records by example_id.
    """
    gold_map = {str(g["example_id"]): g for g in golden_records}
    
    y_true_intent = []
    y_pred_intent = []
    y_true_action = []
    y_pred_action = []
    grounding_passes = []
    grounding_scores = []
    latencies = []
    
    # Stratified segmentation trackers
    edge_cases = {}
    intent_breakdowns = {}

    for pred in predictions:
        eid = str(pred["example_id"])
        if eid not in gold_map:
            continue
        gold = gold_map[eid]
        
        t_intent = gold.get("gold_intent", "")
        p_intent = pred.get("predicted_intent", "")
        t_action = str(gold.get("gold_routing_decision") or gold.get("gold_action", "")).lower()
        p_action = str(pred.get("routing_decision", "")).lower()
        
        y_true_intent.append(t_intent)
        y_pred_intent.append(p_intent)
        y_true_action.append(t_action)
        y_pred_action.append(p_action)
        
        grounding_passes.append(pred.get("grounding_pass", True))
        grounding_scores.append(pred.get("grounding_score", 1.0))
        latencies.append(pred.get("latency_ms", 0.0))

        # Edge case segmentation
        edge_cat = gold.get("edge_case_category") or gold.get("edge_category", "standard")
        if edge_cat not in edge_cases:
            edge_cases[edge_cat] = {"true_action": [], "pred_action": [], "true_intent": [], "pred_intent": []}
        edge_cases[edge_cat]["true_action"].append(t_action)
        edge_cases[edge_cat]["pred_action"].append(p_action)
        edge_cases[edge_cat]["true_intent"].append(t_intent)

        edge_cases[edge_cat]["pred_intent"].append(p_intent)

    # Core metrics
    intent_metrics = calculate_intent_metrics(y_true_intent, y_pred_intent)
    escalation_metrics = calculate_escalation_metrics(y_true_action, y_pred_action)
    autonomy_metrics = calculate_autonomy_resolvability(y_true_action, y_pred_action)
    grounding_metrics = calculate_grounding_metrics(grounding_passes, grounding_scores)
    
    # Latency summary
    latency_summary = {
        "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    }

    # Edge-case breakdown
    edge_segmentation = {}
    for cat, data in edge_cases.items():
        edge_esc = calculate_escalation_metrics(data["true_action"], data["pred_action"])
        edge_acc = accuracy_score(data["true_intent"], data["pred_intent"])
        edge_segmentation[cat] = {
            "count": len(data["true_action"]),
            "intent_accuracy": round(float(edge_acc), 4),
            "escalation_recall": edge_esc["recall"],
            "missed_escalations_fn": edge_esc["false_negatives"]
        }

    # Transparent Composite Reliability Score (Section 8 explicitly requires showing weights)
    # Weights: 0.35 * intent_macro_f1 + 0.35 * escalation_f1 + 0.30 * grounding_pass_rate
    composite_score = round(
        (0.35 * intent_metrics["macro_f1"]) +
        (0.35 * escalation_metrics["f1"]) +
        (0.30 * grounding_metrics["grounding_pass_rate"]),
        4
    )

    return {
        "system_name": system_name,
        "sample_count": len(y_true_intent),
        "headline_metrics": {
            "intent_macro_f1": intent_metrics["macro_f1"],
            "escalation_recall": escalation_metrics["recall"],
            "escalation_false_negatives": escalation_metrics["false_negatives"],
            "grounding_pass_rate": grounding_metrics["grounding_pass_rate"],
            "autonomy_resolvability_rate": autonomy_metrics["autonomy_resolvability_rate"]
        },
        "composite_reliability_score": {
            "score": composite_score,
            "weights_formula": "0.35 * intent_macro_f1 + 0.35 * escalation_f1 + 0.30 * grounding_pass_rate"
        },
        "intent_metrics": intent_metrics,
        "escalation_metrics": escalation_metrics,
        "autonomy_metrics": autonomy_metrics,
        "grounding_metrics": grounding_metrics,
        "latency_summary": latency_summary,
        "edge_case_breakdown": edge_segmentation
    }
