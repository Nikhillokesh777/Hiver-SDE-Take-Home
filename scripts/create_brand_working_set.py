"""
Brand Working Set & Disjoint Split Generator.
Extracts, cleans, normalizes, and splits the selected brand's conversations into:
1. Retrieval Corpus Pool / Training Pool (for FAISS index and baseline classifier)
2. Held-out Golden Set Candidate Pool (strictly disjoint, protected from evaluation leakage)
"""

import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "twcs.csv")
BRAND_TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "brand_comparison_table.json")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
OUTPUT_WORKING_SET = os.path.join(PROCESSED_DIR, "brand_working_set.csv")
OUTPUT_SPLIT_META = os.path.join(PROCESSED_DIR, "split_metadata.json")

# Normalization regexes
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
USER_HANDLE_REGEX = re.compile(r"@([a-zA-Z0-9_]+)")


def normalize_text(text, brand_name):
    """
    Normalize tweet text:
    - Keep emoji intact
    - Mask non-brand user handles to @customer / @user
    - Normalize links to [URL]
    - Clean excessive whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize URLs
    cleaned = URL_REGEX.sub("[URL]", text)
    
    # Anonymize user handles, but preserve brand handle if mentioned
    def mask_handle(match):
        h = match.group(1)
        if h.lower() == brand_name.lower() or h.lower() in ["support", "help"]:
            return f"@{brand_name}"
        return "@customer"

    cleaned = USER_HANDLE_REGEX.sub(mask_handle, cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def create_brand_working_set(brand=None, seed=42, working_set_size=5000):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at {DATA_PATH}. Run scripts/download_data.py first.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Determine brand
    if brand is None:
        if os.path.exists(BRAND_TABLE_PATH):
            with open(BRAND_TABLE_PATH, "r", encoding="utf-8") as f:
                bdata = json.load(f)
                brand = bdata["winner"]
        else:
            raise ValueError("No brand specified and brand_comparison_table.json not found.")

    print(f"Extracting and processing conversation threads for brand: {brand} (seed={seed})...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # 1. Filter to brand outbound replies and their inbound parent customer tweets
    brand_replies = df[df["author_id"] == brand].copy()
    print(f"Total brand outbound tweets for {brand}: {len(brand_replies):,}")

    tweet_lookup = df.set_index("tweet_id").to_dict(orient="index")

    pairs = []
    skipped_missing_parent = 0
    skipped_non_inbound = 0

    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue
        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue

        if parent_id not in tweet_lookup:
            skipped_missing_parent += 1
            continue

        parent_tweet = tweet_lookup[parent_id]
        if not parent_tweet.get("inbound", False):
            skipped_non_inbound += 1
            continue

        pairs.append({
            "brand": brand,
            "customer_tweet_id": parent_id,
            "customer_author": str(parent_tweet.get("author_id", "")),
            "customer_created_at": str(parent_tweet.get("created_at", "")),
            "customer_text_raw": str(parent_tweet.get("text", "")),
            "customer_in_response_to_tweet_id": parent_tweet.get("in_response_to_tweet_id", ""),
            "brand_tweet_id": int(reply["tweet_id"]),
            "brand_created_at": str(reply.get("created_at", "")),
            "brand_reply_raw": str(reply.get("text", "")),
        })

    print(f"Extracted {len(pairs):,} paired (customer -> brand resolution) turns.")
    print(f"  Skipped missing parents: {skipped_missing_parent:,}")
    print(f"  Skipped non-inbound parents: {skipped_non_inbound:,}")

    pairs_df = pd.DataFrame(pairs)

    # 2. Text Normalization (preserving raw columns)
    print("Normalizing texts (handles, URLs, whitespace)...")
    pairs_df["customer_text_clean"] = pairs_df["customer_text_raw"].apply(lambda t: normalize_text(t, brand))
    pairs_df["brand_reply_clean"] = pairs_df["brand_reply_raw"].apply(lambda t: normalize_text(t, brand))

    # 3. Deduplication on normalized customer text
    initial_len = len(pairs_df)
    # Deduplicate keeping the earliest customer tweet
    pairs_df = pairs_df.sort_values("customer_tweet_id").drop_duplicates(subset=["customer_text_clean"], keep="first")
    dedup_removed = initial_len - len(pairs_df)
    print(f"Deduplication removed {dedup_removed:,} duplicate/near-duplicate customer queries ({dedup_removed/initial_len*100:.2f}%).")

    # 4. Stratified Length & Turn Segmentation
    pairs_df["customer_char_len"] = pairs_df["customer_text_clean"].str.len()
    pairs_df["is_first_turn"] = pairs_df["customer_in_response_to_tweet_id"].isna() | (pairs_df["customer_in_response_to_tweet_id"] == "")
    pairs_df["length_bucket"] = pd.qcut(pairs_df["customer_char_len"], q=3, labels=["short", "medium", "long"])

    # 5. Strict Train / Retrieval Corpus vs Golden Set Split
    # To prevent evaluation leakage (§7), we partition tweet IDs into:
    # - retrieval_pool_ids: for FAISS RAG index & baseline training
    # - golden_pool_ids: strictly held-out pool from which the golden set is sampled
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(pairs_df))
    
    # 80% retrieval/development, 20% held-out evaluation pool
    split_idx = int(0.80 * len(pairs_df))
    dev_indices = shuffled_indices[:split_idx]
    eval_indices = shuffled_indices[split_idx:]

    pairs_df["split"] = "retrieval_pool"
    pairs_df.iloc[eval_indices, pairs_df.columns.get_loc("split")] = "held_out_eval_pool"

    # Verify 0 overlap between customer tweet IDs in both splits
    dev_tids = set(pairs_df[pairs_df["split"] == "retrieval_pool"]["customer_tweet_id"])
    eval_tids = set(pairs_df[pairs_df["split"] == "held_out_eval_pool"]["customer_tweet_id"])
    assert len(dev_tids.intersection(eval_tids)) == 0, "CRITICAL: Data leakage detected between dev and eval split!"

    # Also verify 0 overlap on normalized texts
    dev_texts = set(pairs_df[pairs_df["split"] == "retrieval_pool"]["customer_text_clean"])
    eval_texts = set(pairs_df[pairs_df["split"] == "held_out_eval_pool"]["customer_text_clean"])
    text_overlap = len(dev_texts.intersection(eval_texts))
    if text_overlap > 0:
        print(f"Removing {text_overlap} exact text matches from held_out_eval_pool to guarantee 0 leakage...")
        pairs_df = pairs_df[~((pairs_df["split"] == "held_out_eval_pool") & (pairs_df["customer_text_clean"].isin(dev_texts)))]

    print(f"Final split breakdown:")
    print(f"  retrieval_pool: {(pairs_df['split'] == 'retrieval_pool').sum():,} records")
    print(f"  held_out_eval_pool: {(pairs_df['split'] == 'held_out_eval_pool').sum():,} records")

    # Save full working set
    pairs_df.to_csv(OUTPUT_WORKING_SET, index=False)
    print(f"Saved working set to {OUTPUT_WORKING_SET} ({os.path.getsize(OUTPUT_WORKING_SET)/(1024*1024):.2f} MB).")

    # Save split metadata for auditability
    split_meta = {
        "brand": brand,
        "random_seed": seed,
        "total_paired_records": len(pairs_df),
        "deduplicated_removed_count": dedup_removed,
        "retrieval_pool_count": int((pairs_df["split"] == "retrieval_pool").sum()),
        "held_out_eval_pool_count": int((pairs_df["split"] == "held_out_eval_pool").sum()),
        "turn_distribution": {
            "first_turn_count": int(pairs_df["is_first_turn"].sum()),
            "follow_up_count": int((~pairs_df["is_first_turn"]).sum())
        },
        "length_distribution": pairs_df["length_bucket"].value_counts().to_dict(),
        "leakage_verified": True
    }
    with open(OUTPUT_SPLIT_META, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Saved split metadata to {OUTPUT_SPLIT_META}")

    return pairs_df


if __name__ == "__main__":
    brand_arg = sys.argv[1] if len(sys.argv) > 1 else None
    create_brand_working_set(brand=brand_arg)
