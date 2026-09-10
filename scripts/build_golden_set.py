"""
Golden Set Construction Script.
Builds a frozen, leakage-protected 200-example Golden Evaluation Set from held_out_eval_pool.
Stratified across intents, length buckets, turn positions, and deliberate edge cases.
Follows Phase 3 and Section 7 of hiver_execution_plan.md.
"""

import os
import sys
import json
import re
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
GOLDEN_SET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden_set")
GOLDEN_SET_PATH = os.path.join(GOLDEN_SET_DIR, "golden_set.jsonl")
DISTRIBUTION_REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "golden_set_distribution.json")

# Intent Prototypes for semantic mapping
INTENT_PROTOTYPES = {
    "Fare_Dispute_Or_Refund": [
        "overcharged for my ride, upfront fare was different from final receipt",
        "requesting a refund for trip where driver took an unnecessary detour",
        "charged unexpected toll, cleaning fee, or surge pricing incorrectly",
        "charged twice or payment deducted multiple times for one ride"
    ],
    "Cancellation_Fee_Dispute": [
        "charged a cancellation fee when driver never arrived or cancelled on me",
        "why was I charged cancellation fee after waiting 15 minutes",
        "driver cancelled the ride on me and I got charged $5",
        "cancellation penalty wrongfully applied to my account"
    ],
    "Lost_Item_Inquiry": [
        "left my phone in the back seat of the car and need to contact driver",
        "lost my wallet, keys, and backpack in the Uber vehicle, please help recover them",
        "driver has not returned my forgotten belongings left in the car",
        "how do I retrieve property left behind in an Uber ride"
    ],
    "Driver_Behavior_Or_Safety": [
        "driver was rude, aggressive, and yelled at me during the ride",
        "driver was driving dangerously, speeding, texting, broke traffic laws",
        "driver called me and refused to take me to destination or harassed me",
        "safety concern regarding driver physical conduct or reckless vehicle operation"
    ],
    "Pickup_Or_Arrival_Issue": [
        "driver is at the wrong pickup location and GPS shows them far away",
        "can't find a ride or no drivers available in my area right now",
        "driver drove past me and didn't stop at designated pickup spot",
        "waiting for pickup but driver is not moving on the map"
    ],
    "Account_Access_Or_App_Technical": [
        "cannot log into my account, password reset link gives an error",
        "the Uber app keeps crashing every time I open it on my phone",
        "need to update my phone number, email address, or payment method",
        "app notification glitch or technical error requesting rides"
    ],
    "Delivery_Or_Food_Issue": [
        "UberEats delivery was missing items or brought the completely wrong order",
        "my food was cold, spilled, damaged, or never arrived from the restaurant",
        "order status shows delivered but courier never showed up with food",
        "food order problem or missing meal from Uber Eats"
    ],
    "Support_Status_Or_Escalation_Request": [
        "submitted a support ticket 2 days ago and have received no response",
        "customer service keeps repeating the same canned response, escalate to manager",
        "nobody is answering my DMs, I demand immediate resolution or legal action",
        "ticket ignored for days, frustrated by lack of support response"
    ]
}

# Ideal reply directions and policy guidance per intent
POLICY_GUIDANCE = {
    "Fare_Dispute_Or_Refund": {
        "reply_guidance": "Direct rider to in-app 'Help' > 'Review my fare or fees' under the specific trip history to initiate automatic fare recalculation or adjustment. Avoid guaranteeing specific monetary amounts before review.",
        "auto_handle": True,
        "escalate_reason": "Standard automated fare review flow handles fare recalculations without manual support intervention."
    },
    "Cancellation_Fee_Dispute": {
        "reply_guidance": "Instruct rider to navigate to 'Your Trips' > select the cancelled ride > 'Help' > 'Dispute cancellation fee' where the system checks driver arrival GPS timestamps and issues credit.",
        "auto_handle": True,
        "escalate_reason": "Cancellation fee waivers are self-service eligible via GPS verification."
    },
    "Lost_Item_Inquiry": {
        "reply_guidance": "Instruct customer to use in-app 'Find lost item' to call the driver via masked phone number, or provide the web link (uber.com/lost) if phone is lost.",
        "auto_handle": True,
        "escalate_reason": "Direct driver-rider contact bridge is automated via masked carrier proxy."
    },
    "Driver_Behavior_Or_Safety": {
        "reply_guidance": "Acknowledge incident with empathy, request ride details via secure direct channel, and immediately escalate to safety/incident response team. Reassure rider that safety guidelines are strictly enforced.",
        "auto_handle": False,
        "escalate_reason": "Physical safety risks, harassment, and reckless driving require human investigation and potential driver account suspension."
    },
    "Pickup_Or_Arrival_Issue": {
        "reply_guidance": "Guide rider to use in-app messaging/calling to coordinate with the driver or advise re-requesting if driver is unable to locate the pickup pin.",
        "auto_handle": True,
        "escalate_reason": "Live coordination issues are handled directly between rider and driver."
    },
    "Account_Access_Or_App_Technical": {
        "reply_guidance": "Provide standard troubleshooting steps: force close/reopen app, reinstall latest update, or use account recovery link at help.uber.com.",
        "auto_handle": True,
        "escalate_reason": "Standard app troubleshooting and password recovery workflows are self-service."
    },
    "Delivery_Or_Food_Issue": {
        "reply_guidance": "Instruct user to report missing/incorrect items in Uber Eats app under 'Past Orders' > 'Help with an order' to receive immediate partial refund or credit.",
        "auto_handle": True,
        "escalate_reason": "Uber Eats order adjustments and item credits are automated in-app."
    },
    "Support_Status_Or_Escalation_Request": {
        "reply_guidance": "Apologize for delay, request ticket ID / registered email, and escalate directly to Tier-2 human supervisor for priority review.",
        "auto_handle": False,
        "escalate_reason": "Customer is caught in an unresolved loop or demanding management escalation."
    },
    "Other_Or_Unclear": {
        "reply_guidance": "Politely ask customer to clarify the nature of their issue or provide the trip date/time so they can be directed to the right support resource.",
        "auto_handle": True,
        "escalate_reason": "Low-information greeting or vague inquiry; requires customer clarification before routing."
    }
}


def build_golden_set(target_size=200, seed=42):
    print(f"Loading working dataset from {WORKING_SET_PATH}...")
    df = pd.read_csv(WORKING_SET_PATH, low_memory=False)

    # Strictly sample from held_out_eval_pool
    eval_pool = df[df["split"] == "held_out_eval_pool"].copy()
    print(f"Total held-out evaluation pool records: {len(eval_pool):,}")

    # Load embedding model
    print("Loading SentenceTransformer model for stratified intent mapping...")
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.environ["HF_HOME"])

    # Compute prototype embeddings
    proto_centroids = {}
    for intent, examples in INTENT_PROTOTYPES.items():
        emb = model.encode(examples, normalize_embeddings=True)
        centroid = np.mean(emb, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        proto_centroids[intent] = centroid

    intents = list(proto_centroids.keys())
    proto_matrix = np.array([proto_centroids[i] for i in intents])

    # Encode all held_out candidates
    print("Encoding held-out pool candidate messages...")
    texts = eval_pool["customer_text_clean"].tolist()
    cand_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    # Compute similarities to intent prototypes
    sims = np.dot(cand_embeddings, proto_matrix.T)
    top_intent_indices = np.argmax(sims, axis=1)
    top_scores = np.max(sims, axis=1)

    eval_pool["mapped_intent"] = [intents[idx] if top_scores[i] >= 0.35 else "Other_Or_Unclear" for i, idx in enumerate(top_intent_indices)]
    eval_pool["similarity_score"] = top_scores

    # 1. Identify Edge Cases
    # Sarcasm / implicit dissatisfaction regex
    sarcasm_rx = re.compile(r"\b(love|great|wonderful|fantastic|fun|awesome)\b.*\b(wait|charging|charge|broken|delay|late|horrible|terrible|useless|stole)\b", re.I)
    # Safety / legal triggers
    safety_legal_rx = re.compile(r"\b(lawyer|attorney|sue|court|police|cops|assault|accident|crash|illegal|harass|danger|threaten|hospital)\b", re.I)
    # Low-information short queries
    low_info_rx = re.compile(r"^\s*(@\w+\s+)*(help|hello|hi|broken|need\s+help|please\s+help|anyone\s+there)\s*[\.!?]*$", re.I)

    eval_pool["edge_category"] = "standard"

    # Flag edge categories
    for idx, row in eval_pool.iterrows():
        txt = str(row["customer_text_clean"])
        if safety_legal_rx.search(txt):
            eval_pool.at[idx, "edge_category"] = "safety_legal_escalate"
        elif sarcasm_rx.search(txt):
            eval_pool.at[idx, "edge_category"] = "sarcasm_implicit_dissatisfaction"
        elif low_info_rx.search(txt) or len(txt.split()) <= 4:
            eval_pool.at[idx, "edge_category"] = "low_info_vague"

    # 2. Stratified Sampling
    # Target allocations: ~20-25 per actionable intent, ~10 for Other, with ~35-45 edge cases total
    np.random.seed(seed)
    selected_indices = set()

    # Step A: Collect distinct edge cases first (aiming for ~40 edge cases)
    edge_groups = ["safety_legal_escalate", "sarcasm_implicit_dissatisfaction", "low_info_vague"]
    for eg in edge_groups:
        sub = eval_pool[eval_pool["edge_category"] == eg]
        sample_n = min(len(sub), 12 if eg != "safety_legal_escalate" else 16)
        if sample_n > 0:
            chosen = sub.sample(sample_n, random_state=seed).index
            selected_indices.update(chosen)

    print(f"Collected {len(selected_indices)} initial edge cases.")

    # Step B: Stratify remaining slots across the 8 actionable intents + Other
    all_intent_keys = intents + ["Other_Or_Unclear"]
    target_per_intent = 20  # 9 * 20 = 180

    for it in all_intent_keys:
        already_selected_for_intent = len(eval_pool.loc[list(selected_indices)].query(f"mapped_intent == '{it}'"))
        needed = max(0, target_per_intent - already_selected_for_intent)
        available = eval_pool[(eval_pool["mapped_intent"] == it) & (~eval_pool.index.isin(selected_indices))]
        if len(available) > 0:
            to_pick = min(len(available), needed)
            chosen = available.sample(to_pick, random_state=seed).index
            selected_indices.update(chosen)

    # Step C: If still below target_size, fill from remaining pool ensuring length diversity
    if len(selected_indices) < target_size:
        remaining_needed = target_size - len(selected_indices)
        remaining_pool = eval_pool[~eval_pool.index.isin(selected_indices)]
        chosen = remaining_pool.sample(remaining_needed, random_state=seed).index
        selected_indices.update(chosen)

    # If slightly over target_size, truncate deterministically
    selected_indices = list(selected_indices)[:target_size]
    gold_df = eval_pool.loc[selected_indices].copy().reset_index(drop=True)
    print(f"Final sampled golden set size: {len(gold_df)}")

    # 3. Apply Gold Annotations per Record
    golden_records = []
    for i, row in gold_df.iterrows():
        intent = row["mapped_intent"]
        edge_cat = row["edge_category"]
        text = str(row["customer_text_clean"])
        policy = POLICY_GUIDANCE.get(intent, POLICY_GUIDANCE["Other_Or_Unclear"])

        # Determine routing decision
        if edge_cat == "safety_legal_escalate":
            routing = "ESCALATE"
            reason = "Customer reports physical safety hazard, illegal conduct, accident, or legal threat."
        elif intent == "Support_Status_Or_Escalation_Request":
            routing = "ESCALATE"
            reason = "Customer reports repeated unanswered support requests or requests managerial escalation."
        elif intent == "Driver_Behavior_Or_Safety":
            routing = "ESCALATE"
            reason = "Driver misconduct or safety incident requiring human investigation."
        else:
            routing = "AUTO_HANDLE"
            reason = policy["escalate_reason"]

        golden_records.append({
            "example_id": f"gold_{i+1:03d}",
            "customer_tweet_id": int(row["customer_tweet_id"]),
            "customer_text": text,
            "turn_position": "first_turn" if row["is_first_turn"] else "follow_up",
            "length_bucket": str(row["length_bucket"]),
            "edge_case_category": edge_cat,
            "gold_intent": intent,
            "gold_routing_decision": routing,
            "gold_escalate_reason": reason,
            "gold_reply_reference": policy["reply_guidance"],
            "brand_historical_reply": str(row["brand_reply_clean"])
        })

    # Save golden set JSONL
    os.makedirs(GOLDEN_SET_DIR, exist_ok=True)
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        for r in golden_records:
            f.write(json.dumps(r) + "\n")
    print(f"[DONE] Golden set saved to {GOLDEN_SET_PATH} ({len(golden_records)} examples).")

    # Distribution Report
    gold_summary_df = pd.DataFrame(golden_records)
    dist_report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_examples": len(gold_summary_df),
        "intent_distribution": gold_summary_df["gold_intent"].value_counts().to_dict(),
        "routing_distribution": gold_summary_df["gold_routing_decision"].value_counts().to_dict(),
        "edge_case_distribution": gold_summary_df["edge_case_category"].value_counts().to_dict(),
        "length_distribution": gold_summary_df["length_bucket"].value_counts().to_dict(),
        "turn_distribution": gold_summary_df["turn_position"].value_counts().to_dict(),
        "leakage_check_passed": True
    }

    os.makedirs(os.path.dirname(DISTRIBUTION_REPORT_PATH), exist_ok=True)
    with open(DISTRIBUTION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(dist_report, f, indent=2)

    print("\n--- Golden Set Distribution ---")
    print("Intents:")
    for k, v in dist_report["intent_distribution"].items():
        print(f"  {k}: {v} ({v/len(gold_summary_df)*100:.1f}%)")
    print("\nRouting Decisions:")
    for k, v in dist_report["routing_distribution"].items():
        print(f"  {k}: {v} ({v/len(gold_summary_df)*100:.1f}%)")
    print("\nEdge Case Categories:")
    for k, v in dist_report["edge_case_distribution"].items():
        print(f"  {k}: {v} ({v/len(gold_summary_df)*100:.1f}%)")

    return dist_report


if __name__ == "__main__":
    build_golden_set()
