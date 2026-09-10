"""
Semantic Taxonomy Audit using Sentence-Transformers.
Embeds candidate intent definition prototypes and tests zero-shot nearest-centroid
assignment on 100 fresh held-out customer messages to verify real coverage and boundaries.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface")
WORKING_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "brand_working_set.csv")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "taxonomy_coverage_report.json")

# 8 Clear Actionable Intent Prototypes
INTENT_PROTOTYPES = {
    "Fare_Dispute_Or_Refund": [
        "I was overcharged for my ride, the upfront fare was different from the final receipt.",
        "Requesting a refund for a trip where the driver took an unnecessary detour.",
        "Charged unexpected toll, cleaning fee, or surge pricing incorrectly on my ride.",
        "Disputing a charge or promotional discount ride pass not applying to my trip."
    ],
    "Cancellation_Fee_Dispute": [
        "I was charged a cancellation fee when the driver never arrived or cancelled on me.",
        "Why was I charged $5 cancellation fee after waiting 15 minutes for the driver?",
        "Driver accepted and then cancelled, but Uber charged me a cancellation penalty.",
        "Contesting an unfair cancellation charge on my account."
    ],
    "Lost_Item_Inquiry": [
        "I left my phone in the back seat of the car and need to contact the driver.",
        "Lost my wallet, keys, and backpack in the Uber vehicle, please help recover them.",
        "Driver has not returned my forgotten belongings left in the car yesterday.",
        "Need help reaching driver to retrieve my lost property."
    ],
    "Driver_Behavior_Or_Safety": [
        "Driver was rude, aggressive, and yelled at me during the ride.",
        "The driver was driving dangerously, speeding, texting, and broke traffic laws.",
        "Driver called me and refused to take me to my destination or harassed me.",
        "Reporting unsafe driving, verbal abuse, or harassment by the driver."
    ],
    "Pickup_Or_Arrival_Issue": [
        "The driver is at the wrong pickup location and GPS shows them far away.",
        "Can't find a ride or no drivers available in my area right now.",
        "Driver drove past me and didn't stop at the designated pickup spot.",
        "Waiting for pickup but driver is not moving on the map."
    ],
    "Account_Access_Or_App_Technical": [
        "Cannot log into my account, password reset link gives an error.",
        "The Uber app keeps crashing every time I open it on my phone.",
        "Need to update my phone number, email address, or payment method on my profile.",
        "Stop sending promo alerts or technical glitch in the application."
    ],
    "Delivery_Or_Food_Issue": [
        "UberEats delivery was missing items or brought the completely wrong order.",
        "My food was cold, spilled, damaged, or never arrived from the restaurant.",
        "Order status shows delivered but driver never showed up with the food.",
        "Uber Eats driver delivery problem or missing meal."
    ],
    "Support_Status_Or_Escalation_Request": [
        "I submitted a support ticket 2 days ago and have received no response.",
        "Your customer service keeps repeating the same canned response, escalate to a manager.",
        "Nobody is answering my DMs, I demand immediate resolution or I will take legal action.",
        "Frustrated by unhelpful support staff ignoring my open complaint."
    ]
}


def audit_semantic_taxonomy(sample_size=100, seed=123, confidence_threshold=0.35):
    print(f"Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.environ["HF_HOME"])

    # Compute prototype centroid embeddings
    proto_centroids = {}
    for intent, examples in INTENT_PROTOTYPES.items():
        emb = model.encode(examples, normalize_embeddings=True)
        centroid = np.mean(emb, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        proto_centroids[intent] = centroid

    intents = list(proto_centroids.keys())
    proto_matrix = np.array([proto_centroids[i] for i in intents])

    # Load fresh 100-sample from retrieval_pool
    df = pd.read_csv(WORKING_SET_PATH, low_memory=False)
    dev_sample = df[df["split"] == "retrieval_pool"].sample(n=sample_size, random_state=seed).reset_index(drop=True)

    texts = dev_sample["customer_text_clean"].tolist()
    sample_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Cosine similarities (sample_embeddings @ proto_matrix.T)
    sims = np.dot(sample_embeddings, proto_matrix.T)
    max_sim_indices = np.argmax(sims, axis=1)
    max_sim_scores = np.max(sims, axis=1)

    predictions = []
    for i in range(len(dev_sample)):
        best_intent = intents[max_sim_indices[i]]
        best_score = float(max_sim_scores[i])

        if best_score < confidence_threshold:
            assigned = "Other_Or_Unclear"
        else:
            assigned = best_intent

        predictions.append({
            "tweet_id": int(dev_sample.iloc[i]["customer_tweet_id"]),
            "text": texts[i],
            "assigned_intent": assigned,
            "similarity_score": round(best_score, 4),
            "nearest_intent": best_intent
        })

    pred_df = pd.DataFrame(predictions)
    counts = pred_df["assigned_intent"].value_counts().to_dict()
    other_count = counts.get("Other_Or_Unclear", 0)
    other_rate = round(other_count / sample_size * 100, 1)
    coverage_rate = round(100.0 - other_rate, 1)

    summary = {
        "sample_size": sample_size,
        "other_unclear_count": other_count,
        "other_unclear_percentage": other_rate,
        "coverage_rate": coverage_rate,
        "confidence_threshold": confidence_threshold,
        "intent_distribution": counts,
        "sample_classifications": predictions[:25]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Semantic Taxonomy Coverage Audit (N={sample_size}) ---")
    print(f"  Coverage: {coverage_rate}% of messages mapped to actionable intents.")
    print(f"  Other/Unclear Rate: {other_rate}% (Threshold: <10%) -> PASS: {other_rate <= 10.0}")
    print("\nIntent Distribution:")
    for intent, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {cnt} ({cnt/sample_size*100:.1f}%)")

    return summary


if __name__ == "__main__":
    audit_semantic_taxonomy()
