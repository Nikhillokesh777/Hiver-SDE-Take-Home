"""
Stage 3: Intent Classification Module.
Classifies customer messages into the frozen 8+1 intent taxonomy (taxonomy.md).
Uses Sentence-Transformers prototype centroid cosine similarity with softmax calibration.
Returns (intent, confidence, score_distribution).
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field

# Intent Prototypes aligned strictly with frozen taxonomy.md
TAXONOMY_PROTOTYPES: Dict[str, List[str]] = {
    "Fare_Dispute_Or_Refund": [
        "I was overcharged for my ride, upfront fare was different from final receipt",
        "Requesting a refund for trip where driver took an unnecessary detour",
        "Charged unexpected toll, cleaning fee, or surge pricing incorrectly",
        "Payment charged twice or deducted multiple times for one ride"
    ],
    "Cancellation_Fee_Dispute": [
        "I was charged a cancellation fee when driver never arrived or cancelled on me",
        "Why was I charged cancellation fee after waiting 15 minutes for driver",
        "Driver cancelled the ride on me and I got charged $5",
        "Cancellation penalty wrongfully applied to my account"
    ],
    "Lost_Item_Inquiry": [
        "I left my phone in the back seat of the car and need to contact driver",
        "Lost my wallet, keys, and backpack in the Uber vehicle, please help recover them",
        "Driver has not returned my forgotten belongings left in the car yesterday",
        "How do I retrieve property or phone left behind in an Uber ride"
    ],
    "Driver_Behavior_Or_Safety": [
        "Driver was rude, aggressive, and yelled at me during the ride",
        "The driver was driving dangerously, speeding, texting, broke traffic laws",
        "Driver called me and refused to take me to destination or harassed me",
        "Safety concern regarding driver physical conduct or reckless vehicle operation"
    ],
    "Pickup_Or_Arrival_Issue": [
        "Driver is at the wrong pickup location and GPS shows them far away",
        "Can't find a ride or no drivers available in my area right now",
        "Driver drove past me and didn't stop at designated pickup spot",
        "Waiting for pickup but driver is not moving on the map"
    ],
    "Account_Access_Or_App_Technical": [
        "Cannot log into my account, password reset link gives an error",
        "The Uber app keeps crashing every time I open it on my phone",
        "Need to update my phone number, email address, or payment method on profile",
        "App notification glitch or technical error requesting rides"
    ],
    "Delivery_Or_Food_Issue": [
        "UberEats delivery was missing items or brought the completely wrong order",
        "My food was cold, spilled, damaged, or never arrived from the restaurant",
        "Order status shows delivered but courier never showed up with food",
        "Food order problem, missing meal, or delivery delay from Uber Eats"
    ],
    "Support_Status_Or_Escalation_Request": [
        "I submitted a support ticket 2 days ago and have received no response",
        "Customer service keeps repeating the same canned response, escalate to manager",
        "Nobody is answering my DMs, I demand immediate resolution or legal action",
        "Ticket ignored for days, frustrated by lack of support response"
    ]
}

FROZEN_INTENTS: List[str] = list(TAXONOMY_PROTOTYPES.keys()) + ["Other_Or_Unclear"]



class IntentClassificationResult(BaseModel):
    intent: str
    confidence: float
    is_fallback: bool = False
    all_scores: Dict[str, float] = Field(default_factory=dict)

    @property
    def predicted_intent(self) -> str:
        return self.intent


class IntentClassifier:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.35,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")
        self._model = None
        self._proto_centroids = None
        self._intent_labels = list(TAXONOMY_PROTOTYPES.keys())

    def _lazy_init(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            
            # Precompute prototype centroids
            centroids = []
            for intent in self._intent_labels:
                proto_texts = TAXONOMY_PROTOTYPES[intent]
                proto_embs = self._model.encode(proto_texts, normalize_embeddings=True, show_progress_bar=False)
                centroid = np.mean(proto_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids.append(centroid)
            self._proto_centroids = np.array(centroids)

    def classify(self, text: str) -> IntentClassificationResult:
        """
        Classifies input text into one of the 8 actionable intents or 'Other_Or_Unclear'.
        """
        self._lazy_init()
        if not text or not text.strip():
            return IntentClassificationResult(
                intent="Other_Or_Unclear",
                confidence=0.0,
                is_fallback=True,
                all_scores={k: 0.0 for k in self._intent_labels}
            )

        text_emb = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        # Cosine similarity against each prototype centroid
        sims = np.dot(self._proto_centroids, text_emb)

        # Softmax calibration for probabilities
        exp_sims = np.exp(sims * 5.0)  # temperature scale of 0.2
        probs = exp_sims / np.sum(exp_sims)

        score_dict = {
            self._intent_labels[i]: round(float(probs[i]), 4)
            for i in range(len(self._intent_labels))
        }

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_prob = float(probs[best_idx])
        best_intent = self._intent_labels[best_idx]

        # Apply boundary threshold for Other/Unclear
        if best_sim < self.confidence_threshold:
            return IntentClassificationResult(
                intent="Other_Or_Unclear",
                confidence=round(1.0 - best_prob, 4),
                is_fallback=True,
                all_scores=score_dict
            )

        return IntentClassificationResult(
            intent=best_intent,
            confidence=round(best_prob, 4),
            is_fallback=False,
            all_scores=score_dict
        )
