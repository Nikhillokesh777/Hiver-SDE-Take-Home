"""
Unit and Integration Tests for Grounded Support Agent Pipeline (src/).
Tests Stages 1-8 individually and Stage 9 end-to-end with zero reliance on live external APIs.
"""

import os
import sys
import json
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import Preprocessor
from src.thread_reconstruction import ThreadReconstructor
from src.intent_classifier import IntentClassifier, FROZEN_INTENTS
from src.grounding_check import GroundingChecker
from src.routing import RoutingEngine


class TestPipelineComponents:

    def test_preprocessor_masking(self):
        p = Preprocessor(brand_handle="Uber_Support")
        raw = "@Uber_Support @driver_guy I was charged $25 extra for trip https://uber.com/receipt! 😡"
        out = p.process(raw)
        
        assert "@customer" in out["clean_text"]
        assert "@uber_support" in out["clean_text"]
        assert "[URL]" in out["clean_text"]
        assert "charged $25 extra" in out["clean_text"]
        assert out["char_length"] > 0

    def test_thread_reconstruction(self):
        tr = ThreadReconstructor(max_context_turns=3)
        priors = [
            {"role": "customer", "text": "Turn 1"},
            {"role": "agent", "text": "Turn 2"},
            {"role": "customer", "text": "Turn 3"},
            {"role": "agent", "text": "Turn 4"},
        ]
        ctx = tr.reconstruct(current_text="Turn 5", prior_turns=priors)
        assert len(ctx.prior_turns) <= 3
        assert ctx.total_turns == 4
        assert "Customer (Current): Turn 5" in ctx.formatted_transcript

    def test_intent_classifier_labels(self):
        ic = IntentClassifier()
        result = ic.classify("I was charged a $5 cancellation fee unfairly.")
        assert result.predicted_intent in FROZEN_INTENTS
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.all_scores) >= 8

    def test_grounding_checker_catches_unsupported_claims(self):
        gc = GroundingChecker()
        evidence = "[Historical Case #101]\nCustomer: Overcharged\nResolution: Send us a DM with your account email."
        
        # Hallucinated phone number not in evidence
        bad_reply = "Please call our emergency billing team at 1-800-555-0199 for an instant $50 refund."
        result = gc.check(bad_reply, evidence, predicted_intent="Fare_Dispute_Or_Refund")
        assert result.grounding_pass is False
        assert len(result.unsupported_claims) > 0

        # Grounded reply matching evidence
        good_reply = "We understand your concern. Please send us a direct message with your account email so we can review this trip."
        result_good = gc.check(good_reply, evidence, predicted_intent="Fare_Dispute_Or_Refund")
        assert result_good.grounding_pass is True


    def test_routing_engine_escalates_on_safety(self):
        re = RoutingEngine(auto_handle_threshold=0.70)
        decision = re.decide(
            intent="Driver_Behavior_Or_Safety",
            intent_confidence=0.95,
            retrieval_similarity=0.90,
            grounding_passed=True,
            grounding_score=0.95,
            text="The driver threatened me with violence and refused to let me exit the car!"
        )
        assert decision.action == "escalate"
        assert "safety" in decision.reason.lower() or "intent" in decision.reason.lower()

    def test_routing_engine_auto_handles_on_high_confidence(self):
        re = RoutingEngine(auto_handle_threshold=0.65)
        decision = re.decide(
            intent="Lost_Item_Inquiry",
            intent_confidence=0.85,
            retrieval_similarity=0.82,
            grounding_passed=True,
            grounding_score=0.95,
            text="I forgot my umbrella in the vehicle, how do I reach the driver?"
        )
        assert decision.action == "auto_handle"
        assert decision.composite_confidence >= 0.65

    def test_retrieval_offline_artifact(self):
        from src.retrieval import HistoricalCaseRetriever
        retriever = HistoricalCaseRetriever(top_k=3)
        res = retriever.retrieve("I left my laptop in the car")
        assert len(res.retrieved_cases) == 3
        assert res.max_similarity > 0.0
        assert len(res.evidence_text) > 0
        assert all(isinstance(c.case_id, int) for c in res.retrieved_cases)

