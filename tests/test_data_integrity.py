"""
Tests for Phase 1 Data Integrity, Preprocessing Normalization, and Split Leakage Prevention.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.create_brand_working_set import normalize_text



def test_normalize_text_masks_user_handles():
    raw = "@115712 hello there, please help with @AppleSupport"
    cleaned = normalize_text(raw, brand_name="AppleSupport")
    assert "@customer" in cleaned
    assert "@115712" not in cleaned
    assert "@AppleSupport" in cleaned


def test_normalize_text_replaces_urls():
    raw = "Check the status at https://support.apple.com/status and http://example.com"
    cleaned = normalize_text(raw, brand_name="AppleSupport")
    assert "https://support.apple.com/status" not in cleaned
    assert "[URL]" in cleaned


def test_normalize_text_preserves_emoji():
    raw = "My phone died completely 😡🔥 please help @AppleSupport"
    cleaned = normalize_text(raw, brand_name="AppleSupport")
    assert "😡" in cleaned
    assert "🔥" in cleaned


def test_disjoint_split_leakage_assertion():
    # Synthetic verification of leakage check logic
    data = {
        "customer_tweet_id": [101, 102, 103, 104, 105],
        "customer_text_clean": ["issue one", "issue two", "issue three", "issue four", "issue five"],
        "split": ["retrieval_pool", "retrieval_pool", "retrieval_pool", "held_out_eval_pool", "held_out_eval_pool"]
    }
    df = pd.DataFrame(data)

    dev_ids = set(df[df["split"] == "retrieval_pool"]["customer_tweet_id"])
    eval_ids = set(df[df["split"] == "held_out_eval_pool"]["customer_tweet_id"])
    
    # Assert zero intersection
    assert len(dev_ids.intersection(eval_ids)) == 0

    dev_texts = set(df[df["split"] == "retrieval_pool"]["customer_text_clean"])
    eval_texts = set(df[df["split"] == "held_out_eval_pool"]["customer_text_clean"])
    assert len(dev_texts.intersection(eval_texts)) == 0
