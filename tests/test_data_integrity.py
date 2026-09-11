"""
Tests for Data Integrity: Schema, Missing Values, Deduplication, and Leakage Prevention.
Tests schema integrity, null value prevention, and strict 0-leakage between data splits.
"""

import os
import sys
import json
import pytest
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

WORKING_SET_PATH = os.path.join(ROOT_DIR, "data", "processed", "brand_working_set.csv")
GOLDEN_SET_PATH = os.path.join(ROOT_DIR, "data", "golden_set", "golden_set.jsonl")
SPLIT_METADATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "split_metadata.json")


def test_working_set_schema_and_columns():
    """Verify processed brand working set has all required schema columns."""
    assert os.path.exists(WORKING_SET_PATH), f"Working set missing at {WORKING_SET_PATH}"
    df = pd.read_csv(WORKING_SET_PATH, nrows=50)
    
    required_cols = [
        "brand", "customer_tweet_id", "customer_author", "customer_text_clean", 
        "brand_tweet_id", "brand_reply_clean", "split"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_working_set_no_critical_missing_values():
    """Verify clean customer queries and agent replies have no null/empty values."""
    df = pd.read_csv(WORKING_SET_PATH, nrows=2000)
    assert df["customer_text_clean"].isnull().sum() == 0
    assert df["brand_reply_clean"].isnull().sum() == 0
    assert (df["customer_text_clean"].astype(str).str.strip() == "").sum() == 0


def test_disjoint_split_leakage_assertion():
    """Verify strictly 0 leakage between retrieval_pool and held_out_eval_pool."""
    if os.path.exists(SPLIT_METADATA_PATH):
        with open(SPLIT_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("leakage_verified") is True

    # Assert 0 intersection across customer tweet IDs
    df = pd.read_csv(WORKING_SET_PATH, usecols=["customer_tweet_id", "split"])
    dev_ids = set(df[df["split"] == "retrieval_pool"]["customer_tweet_id"])
    eval_ids = set(df[df["split"] == "held_out_eval_pool"]["customer_tweet_id"])
    assert len(dev_ids.intersection(eval_ids)) == 0


def test_golden_set_schema_and_volume():
    """Verify Golden Evaluation Set contains between 150-250 validated ground-truth records."""
    assert os.path.exists(GOLDEN_SET_PATH), f"Golden set missing at {GOLDEN_SET_PATH}"
    records = []
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    assert 150 <= len(records) <= 250, f"Golden set size {len(records)} outside required 150-250 range"
    
    required_keys = [
        "example_id", "customer_tweet_id", "customer_text", 
        "gold_intent", "gold_routing_decision", "gold_reply_reference"
    ]
    for r in records[:20]:
        for k in required_keys:
            assert k in r, f"Golden record missing required key: {k}"
        assert r["gold_routing_decision"].upper() in ["AUTO_HANDLE", "ESCALATE"]
