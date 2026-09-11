"""
Structural Data Inspection Script for Kaggle Twitter Customer Support Dataset (twcs.csv).
Produces structured telemetry and verifies dataset integrity without making assumptions.
"""

import os
import sys
import json
import time
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "twcs.csv")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "data_inspection_report.json")


def run_structural_inspection(data_path=DATA_PATH, max_rows=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run scripts/download_data.py first.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    print(f"Loading dataset from {data_path} (max_rows={max_rows})...")
    start_t = time.time()

    # Read CSV
    df = pd.read_csv(data_path, nrows=max_rows, low_memory=False)
    load_time = time.time() - start_t
    print(f"Loaded {len(df):,} rows in {load_time:.2f}s.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_percentages": {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns},
    }

    # 1. Inbound vs Outbound breakdown
    inbound_counts = df["inbound"].value_counts().to_dict()
    report["inbound_distribution"] = {str(k): int(v) for k, v in inbound_counts.items()}
    print("\n--- Inbound vs Outbound ---")
    for k, v in inbound_counts.items():
        print(f"  inbound={k}: {v:,} ({v / len(df) * 100:.1f}%)")

    # 2. Candidate Brands Identification
    # Brand tweets are outbound (inbound == False) or authors who reply to customer inquiries
    brand_authors = df[df["inbound"] == False]["author_id"].value_counts()
    report["total_unique_brands"] = int(len(brand_authors))
    report["top_20_brands_by_replies"] = {str(k): int(v) for k, v in brand_authors.head(20).items()}
    print(f"\n--- Total Unique Brand Accounts: {len(brand_authors)} ---")
    print("Top 10 most active brand accounts (by outbound replies):")
    for brand, cnt in brand_authors.head(10).items():
        print(f"  {brand}: {cnt:,} replies")

    # 3. Text Duplication & Near-Duplication Rates
    raw_texts = df["text"].dropna().astype(str)
    exact_duplicates = len(raw_texts) - raw_texts.nunique()
    report["exact_duplicate_texts"] = int(exact_duplicates)
    report["exact_duplicate_pct"] = round(exact_duplicates / len(raw_texts) * 100, 2)
    print(f"\n--- Text Duplication ---")
    print(f"  Exact duplicate text count: {exact_duplicates:,} ({report['exact_duplicate_pct']}%)")

    # 4. Thread Chaining Analysis
    # Analyze in_response_to_tweet_id and response_tweet_id coverage
    has_in_reply_to = int(df["in_response_to_tweet_id"].notna().sum())
    has_response = int(df["response_tweet_id"].notna().sum())
    report["threading"] = {
        "has_in_response_to_tweet_id": has_in_reply_to,
        "has_in_response_to_pct": round(has_in_reply_to / len(df) * 100, 2),
        "has_response_tweet_id": has_response,
        "has_response_pct": round(has_response / len(df) * 100, 2)
    }
    print(f"\n--- Threading Fields Coverage ---")
    print(f"  Has in_response_to_tweet_id: {has_in_reply_to:,} ({report['threading']['has_in_response_to_pct']}%)")
    print(f"  Has response_tweet_id: {has_response:,} ({report['threading']['has_response_pct']}%)")

    # 5. Language check (heuristic English word match vs non-ascii)
    non_ascii_count = int(raw_texts.str.contains(r"[^\x00-\x7F]").sum())
    report["non_ascii_messages"] = non_ascii_count
    report["non_ascii_pct"] = round(non_ascii_count / len(raw_texts) * 100, 2)
    print(f"\n--- Language / Non-ASCII Characters ---")
    print(f"  Messages with non-ASCII characters: {non_ascii_count:,} ({report['non_ascii_pct']}%)")

    # 6. Sample 10 raw threads for qualitative spot-check
    print("\n--- Spot-reading sample customer-brand exchanges ---")
    sample_exchanges = []
    # Find records where a brand replied to a customer tweet
    brand_replies = df[(df["inbound"] == False) & (df["in_response_to_tweet_id"].notna())].sample(
        min(10, len(df)), random_state=42
    )

    tweet_lookup = df.set_index("tweet_id")["text"].to_dict()
    author_lookup = df.set_index("tweet_id")["author_id"].to_dict()

    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row["in_response_to_tweet_id"]
        # parent_id might be float
        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue
        
        parent_text = tweet_lookup.get(parent_id, "[Parent tweet text not found in sample/data]")
        parent_author = author_lookup.get(parent_id, "Customer")
        sample_exchanges.append({
            "brand": str(reply_row["author_id"]),
            "customer_author": str(parent_author),
            "customer_tweet_id": int(parent_id),
            "customer_text": str(parent_text),
            "brand_reply_id": int(reply_row["tweet_id"]),
            "brand_reply_text": str(reply_row["text"])
        })

    report["sample_exchanges"] = sample_exchanges
    for i, ex in enumerate(sample_exchanges[:5], 1):
        print(f"\n[Sample {i}] Brand: {ex['brand']}")
        print(f"  Customer ({ex['customer_author']}): {ex['customer_text']}")
        print(f"  Brand Reply: {ex['brand_reply_text']}")

    # Save structured telemetry
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] Structural inspection report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    max_r = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_structural_inspection(max_rows=max_r)
