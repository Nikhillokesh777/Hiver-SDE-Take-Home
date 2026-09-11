"""
Stage 7 & 8: Confidence Estimation and Routing Module (Auto-Handle vs. Escalate).
Implements transparent, explainable decision boundaries combining:
- Intent classification confidence
- Historical retrieval similarity
- Grounding verification score
- Mandatory safety & legal escalation triggers
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    composite_confidence: float
    intent_confidence: float
    retrieval_similarity: float
    grounding_score: float
    triggered_rules: List[str] = Field(default_factory=list)

    @property
    def action(self) -> str:
        return self.decision.lower()


class RoutingEngine:
    def __init__(
        self,
        auto_handle_threshold: float = 0.70,
        min_retrieval_similarity: float = 0.40,
        min_grounding_score: float = 0.75,
        weight_intent: float = 0.30,
        weight_retrieval: float = 0.35,
        weight_grounding: float = 0.35,
        mandatory_escalate_intents: Optional[List[str]] = None
    ):
        self.auto_handle_threshold = auto_handle_threshold
        self.min_retrieval_similarity = min_retrieval_similarity
        self.min_grounding_score = min_grounding_score
        self.weight_intent = weight_intent
        self.weight_retrieval = weight_retrieval
        self.weight_grounding = weight_grounding
        self.mandatory_escalate_intents = mandatory_escalate_intents or [
            "Support_Status_Or_Escalation_Request",
            "Driver_Behavior_Or_Safety"
        ]

    def compute_composite_confidence(
        self,
        intent_conf: float,
        retrieval_sim: float,
        grounding_score: float
    ) -> float:
        """Computes transparent, weighted composite confidence score."""
        score = (
            (self.weight_intent * intent_conf) +
            (self.weight_retrieval * retrieval_sim) +
            (self.weight_grounding * grounding_score)
        )
        return round(float(score), 4)

    def decide(
        self,
        intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        grounding_passed: bool = True,
        grounding_score: float = 1.0,
        edge_category: str = "standard",
        text: str = "",
        **kwargs
    ) -> RoutingDecision:
        return self.route(
            intent=intent,
            intent_confidence=intent_confidence,
            retrieval_similarity=retrieval_similarity,
            grounding_pass=grounding_passed,
            grounding_score=grounding_score,
            edge_category=edge_category,
            raw_text=text or kwargs.get("raw_text", "")
        )

    def route(
        self,
        intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        grounding_pass: bool,
        grounding_score: float,
        edge_category: str = "standard",
        raw_text: str = ""
    ) -> RoutingDecision:

        """
        Evaluates decision boundaries and returns explainable AUTO_HANDLE vs ESCALATE routing.
        """
        composite_conf = self.compute_composite_confidence(
            intent_confidence, retrieval_similarity, grounding_score
        )

        triggered_rules: List[str] = []

        # 1. Safety & Legal Trigger
        if edge_category == "safety_legal_escalate" or any(w in raw_text.lower() for w in ["lawyer", "police", "sue", "attorney", "assault", "accident"]):
            triggered_rules.append("MANDATORY_SAFETY_OR_LEGAL_ESCALATION")
            return RoutingDecision(
                decision="ESCALATE",
                reason="Mandatory safety/legal trigger: message references legal threat, police, accident, or serious safety hazard.",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )

        # 2. Mandatory Escalation Intent (e.g. repeated ticket loop or severe driver misconduct)
        if intent in self.mandatory_escalate_intents:
            triggered_rules.append(f"MANDATORY_INTENT_ESCALATION_{intent.upper()}")
            return RoutingDecision(
                decision="ESCALATE",
                reason=f"Mandatory escalation policy: intent '{intent}' requires direct human tier-2 supervisor handling.",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )

        # 3. Grounding Failure Trigger
        if not grounding_pass or grounding_score < self.min_grounding_score:
            triggered_rules.append("GROUNDING_VERIFICATION_FAILED")
            return RoutingDecision(
                decision="ESCALATE",
                reason=f"Grounding failure: generated draft contained unsupported factual claims (score: {grounding_score:.2f} < {self.min_grounding_score}).",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )

        # 4. Insufficient Retrieval Evidence Trigger
        if retrieval_similarity < self.min_retrieval_similarity:
            triggered_rules.append("LOW_RETRIEVAL_SIMILARITY")
            return RoutingDecision(
                decision="ESCALATE",
                reason=f"Insufficient evidence: nearest historical case similarity ({retrieval_similarity:.2f}) below confidence floor ({self.min_retrieval_similarity}).",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )

        # 5. Composite Confidence Threshold Check
        if composite_conf >= self.auto_handle_threshold:
            triggered_rules.append("HIGH_CONFIDENCE_AUTO_HANDLE")
            return RoutingDecision(
                decision="AUTO_HANDLE",
                reason=f"High intent confidence ({intent_confidence:.2f}), strong historical evidence ({retrieval_similarity:.2f}), and verified grounded reply ({grounding_score:.2f}).",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )
        else:
            triggered_rules.append("LOW_COMPOSITE_CONFIDENCE")
            return RoutingDecision(
                decision="ESCALATE",
                reason=f"Composite confidence ({composite_conf:.2f}) below auto-handle threshold ({self.auto_handle_threshold}). Preferring safe escalation over uncertain autonomy.",
                composite_confidence=composite_conf,
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                grounding_score=grounding_score,
                triggered_rules=triggered_rules
            )
