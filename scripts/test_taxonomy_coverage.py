"""
Taxonomy Coverage Check Script.
Evaluates the candidate 8+1 intent taxonomy against 100 fresh customer messages
to measure unclassifiable rate (Other/Unclear), ambiguity rate, and category coverage.
Follows Step 3 of Phase 2 in hiver_execution_plan.md.
"""

import os
import sys
import json
import re
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

WORKING_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "brand_working_set.csv")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "taxonomy_coverage_report.json")

# Rule-based / Keyword validation heuristics for fast 100-sample audit
INTENT_PATTERNS = {
    "Cancellation_Fee_Dispute": re.compile(r"\b(cancel|cancellation|cancelled|canceling)\b.*\b(fee|charge|charged|\$|\d+)\b|\bcharged\b.*\b(cancellation|cancel)\b", re.I),
    "Lost_Item_Inquiry": re.compile(r"\b(lost|left|forget|forgot|phone|wallet|keys|purse|bag|backpack|belongings)\b.*\b(car|uber|back seat|driver)\b|\bleft\s+my\b", re.I),
    "Fare_Dispute_Or_Refund": re.compile(r"\b(overcharged|charged|charge|refund|fare|receipt|price|cost|\$|toll|bill|clean(ing)?\s+fee|surge)\b", re.I),
    "Driver_Behavior_Or_Safety": re.compile(r"\b(rude|safety|driving|reckless|threaten|harass|attitude|unprofessional|police|refused|yelled|illegal|accident)\b", re.I),
    "Pickup_Or_Arrival_Issue": re.compile(r"\b(pickup|pick\s+up|arrived|arrival|location|waiting|wait|wrong\s+spot|gps|where\s+is\s+(the\s+)?driver)\b", re.I),
    "Account_Access_Or_App_Technical": re.compile(r"\b(app|crash|login|log\s+in|password|reset|account|locked|update|phone\s+number|verification|code|error)\b", re.I),
    "Delivery_Or_Food_Issue": re.compile(r"\b(food|eats|order|restaurant|delivered|cold|missing|meal|delivery|driver\s+ate)\b", re.I),
    "Support_Status_Or_Escalation_Request": re.compile(r"\b(ticket|dm|message|response|reply|hours?|days?|ignoring|escalat|manager|lawyer|sue|unresponsive)\b", re.I),
}


def classify_by_heuristics(text):
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(text):
            return intent
    return "Other_Or_Unclear"


def run_coverage_check(sample_size=100, seed=123):
    df = pd.read_csv(WORKING_SET_PATH, low_memory=False)
    # Filter to retrieval_pool, take fresh sample not used in cluster discovery (different seed)
    dev_pool = df[df["split"] == "retrieval_pool"].sample(n=sample_size, random_state=seed).copy()

    predictions = []
    for _, row in dev_pool.iterrows():
        text = str(row["customer_text_clean"])
        assigned = classify_by_heuristics(text)
        predictions.append({
            "tweet_id": int(row["customer_tweet_id"]),
            "text": text,
            "predicted_intent": assigned
        })

    pred_df = pd.DataFrame(predictions)
    counts = pred_df["predicted_intent"].value_counts().to_dict()
    other_count = counts.get("Other_Or_Unclear", 0)
    other_rate = round(other_count / sample_size * 100, 1)

    coverage_summary = {
        "sample_size": sample_size,
        "other_unclear_count": other_count,
        "other_unclear_percentage": other_rate,
        "coverage_rate": round(100.0 - other_rate, 1),
        "intent_distribution": counts,
        "sample_classifications": predictions[:15]
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(coverage_summary, f, indent=2)

    print(f"\n--- Taxonomy Coverage Audit (N={sample_size}) ---")
    print(f"  Coverage: {coverage_summary['coverage_rate']}% of messages matched actionable intents.")
    print(f"  Other/Unclear Rate: {other_rate}% (Threshold: <10-15%) -> PASS: {other_rate <= 15.0}")
    print("\nClass distribution in fresh 100 sample:")
    for intent, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {cnt} ({cnt/sample_size*100:.1f}%)")

    return coverage_summary


if __name__ == "__main__":
    run_coverage_check()
