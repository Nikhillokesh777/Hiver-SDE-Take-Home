"""
Baselines Implementation.
Implements the two comparison baselines:
1. Trivial Baseline (Majority-class intent + canned template reply + static routing)
2. Simple ML / Engineering Baseline (TF-IDF + Logistic Regression intent + 1-NN historical reply verbatim + keyword escalation)

Both baselines run on the identical golden evaluation set with identical output schemas.
"""

import os
import sys
import time
import json
import pickle
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKING_SET_PATH = os.path.join(ROOT_DIR, "data", "processed", "brand_working_set.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
TFIDF_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "simple_baseline_tfidf.pkl")


class BaselineOutput(BaseModel):
    system_name: str
    example_id: str
    input_text: str
    predicted_intent: str
    intent_confidence: float
    generated_reply: str
    routing_decision: str  # "auto_handle" or "escalate"
    routing_reason: str
    grounding_score: float = 1.0  # Historical verbatim replies are 1.0 by definition
    grounding_pass: bool = True
    latency_ms: float = 0.0


# ==============================================================================
# 1. TRIVIAL BASELINE
# ==============================================================================
class TrivialBaseline:
    """
    Baseline 1: Trivial baseline (no ML at all).
    - Intent: Static majority-class prediction (Fare_Dispute_Or_Refund).
    - Reply: Static canned template per intent.
    - Escalation: Always auto_handle (establishes the absolute performance floor).
    """

    CANNED_TEMPLATES = {
        "Fare_Dispute_Or_Refund": "Thanks for reaching out. We apologize for the billing issue. We are reviewing your trip fare and will update you shortly.",
        "Cancellation_Fee_Dispute": "Thanks for reaching out. We understand your concern regarding the cancellation fee and our support team will look into this.",
        "Lost_Item_Inquiry": "Thanks for contacting us. Please check the Uber app under 'Your Trips' to contact your driver directly regarding lost items.",
        "Driver_Behavior_Or_Safety": "Thanks for reaching out. Safety is our top priority. We are reviewing your report and someone will reach out.",
        "Pickup_Or_Arrival_Issue": "We apologize for the pickup delay. We are actively working to improve driver dispatch times.",
        "Account_Access_Or_App_Technical": "Thanks for reaching out. Please try restarting your app or resetting your password via our help center.",
        "Delivery_Or_Food_Issue": "Thanks for contacting us. We apologize for the order issue and are reviewing the details with our delivery team.",
        "Support_Status_Or_Escalation_Request": "Thanks for following up. Your ticket is currently in our queue and an agent will respond soon.",
        "Other_Or_Unclear": "Thanks for reaching out. Please provide more details about your issue so we can best assist you."
    }

    def __init__(self, majority_intent: str = "Fare_Dispute_Or_Refund"):
        self.majority_intent = majority_intent

    def run(self, raw_text: str, example_id: str = "") -> BaselineOutput:
        start_t = time.perf_counter()
        
        reply = self.CANNED_TEMPLATES.get(
            self.majority_intent,
            "Thanks for reaching out to Uber Support. We are looking into this."
        )
        
        latency = (time.perf_counter() - start_t) * 1000.0

        return BaselineOutput(
            system_name="trivial_baseline",
            example_id=example_id,
            input_text=raw_text,
            predicted_intent=self.majority_intent,
            intent_confidence=0.50,
            generated_reply=reply,
            routing_decision="auto_handle",
            routing_reason="Trivial baseline static policy: always auto-handle",
            grounding_score=0.50,
            grounding_pass=True,
            latency_ms=round(latency, 2)
        )


# ==============================================================================
# 2. SIMPLE ML / ENGINEERING BASELINE
# ==============================================================================
class SimpleMLBaseline:
    """
    Baseline 2: Simple ML / Engineering Baseline.
    - Intent: TF-IDF Vectorizer + Logistic Regression trained on retrieval_pool.
    - Reply: 1-Nearest Neighbor historical brand reply returned verbatim (no LLM, no grounding check).
    - Escalation: Fixed keyword matching rule (e.g. lawyer, police, safety, cancel, refund, emergency).
    """

    ESCALATION_KEYWORDS = {
        "lawyer", "police", "legal", "court", "attorney", "accident",
        "crash", "emergency", "assault", "injured", "injury", "threat",
        "safety", "stolen", "danger", "hospital", "harass"
    }

    def __init__(self, model_path: str = TFIDF_MODEL_PATH):
        self.model_path = model_path
        self.vectorizer: Optional[Any] = None
        self.classifier: Optional[Any] = None
        self.retrieval_corpus: Optional[pd.DataFrame] = None
        self.corpus_tfidf: Optional[Any] = None

    def fit_or_load(self, working_set_path: str = WORKING_SET_PATH, train_size: int = 3000, seed: int = 42):
        """Trains or loads precomputed TF-IDF + Logistic Regression model on retrieval_pool."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.classifier = data["classifier"]
                self.retrieval_corpus = data["retrieval_corpus"]
                self.corpus_tfidf = data["corpus_tfidf"]
            return

        print(f"[SimpleMLBaseline] Training TF-IDF + LogisticRegression on {working_set_path}...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics.pairwise import cosine_similarity
        from src.intent_classifier import IntentClassifier

        df = pd.read_csv(working_set_path, low_memory=False)
        corpus = df[df["split"] == "retrieval_pool"].copy()
        
        if len(corpus) > train_size:
            train_df = corpus.sample(train_size, random_state=seed).reset_index(drop=True)
        else:
            train_df = corpus.reset_index(drop=True)

        # Vectorized batch pseudo-labeling using IntentClassifier prototypes
        ic = IntentClassifier()
        ic._lazy_init()
        print(f"[SimpleMLBaseline] Vectorized batch labeling of {len(train_df)} training samples...", flush=True)
        texts = train_df["customer_text_clean"].astype(str).tolist()
        batch_embs = ic._model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        sim_matrix = np.dot(batch_embs, ic._proto_centroids.T)
        best_indices = np.argmax(sim_matrix, axis=1)
        max_sims = np.max(sim_matrix, axis=1)

        labels = [
            "Other_Or_Unclear" if sim < ic.confidence_threshold else ic._intent_labels[idx]
            for idx, sim in zip(best_indices, max_sims)
        ]
        train_df["pseudo_intent"] = labels


        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(train_df["customer_text_clean"].astype(str))
        y = train_df["pseudo_intent"].values

        self.classifier = LogisticRegression(max_iter=500, C=1.0)
        self.classifier.fit(X, y)

        self.retrieval_corpus = train_df[["customer_tweet_id", "customer_text_clean", "brand_reply_clean"]].copy()
        self.corpus_tfidf = X

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "retrieval_corpus": self.retrieval_corpus,
                "corpus_tfidf": self.corpus_tfidf
            }, f)
        print(f"[SimpleMLBaseline] Model successfully saved to {self.model_path}.")

    def run(self, raw_text: str, example_id: str = "") -> BaselineOutput:
        start_t = time.perf_counter()
        if (
            self.vectorizer is None
            or self.classifier is None
            or self.retrieval_corpus is None
            or self.corpus_tfidf is None
        ):
            self.fit_or_load()

        if (
            self.vectorizer is None
            or self.classifier is None
            or self.retrieval_corpus is None
            or self.corpus_tfidf is None
        ):
            raise RuntimeError(
                "SimpleMLBaseline failed to initialize: one or more model artifacts "
                "(vectorizer, classifier, retrieval_corpus, corpus_tfidf) are None."
            )

        from sklearn.metrics.pairwise import cosine_similarity

        # Stage 1: TF-IDF Intent Classification
        clean_text = raw_text.lower()
        vec = self.vectorizer.transform([clean_text])
        probs = self.classifier.predict_proba(vec)[0]
        max_idx = np.argmax(probs)
        pred_intent = str(self.classifier.classes_[max_idx])
        confidence = float(probs[max_idx])

        # Stage 2: 1-NN Verbatim Historical Reply Retrieval
        sims = cosine_similarity(vec, self.corpus_tfidf)[0]
        best_idx = int(np.argmax(sims))
        verbatim_reply = str(self.retrieval_corpus.iloc[best_idx]["brand_reply_clean"])

        # Stage 3: Keyword-based Escalation Rule
        tokens = set(clean_text.split())
        matched_keywords = tokens.intersection(self.ESCALATION_KEYWORDS)
        
        if matched_keywords:
            routing_decision = "escalate"
            routing_reason = f"Simple ML keyword trigger matched: {', '.join(sorted(matched_keywords))}"
        else:
            routing_decision = "auto_handle"
            routing_reason = "Simple ML baseline: no escalation keywords detected"

        latency = (time.perf_counter() - start_t) * 1000.0

        return BaselineOutput(
            system_name="simple_ml_baseline",
            example_id=example_id,
            input_text=raw_text,
            predicted_intent=pred_intent,
            intent_confidence=round(confidence, 4),
            generated_reply=verbatim_reply,
            routing_decision=routing_decision,
            routing_reason=routing_reason,
            grounding_score=1.0,  # Verbatim historical resolution
            grounding_pass=True,
            latency_ms=round(latency, 2)
        )
