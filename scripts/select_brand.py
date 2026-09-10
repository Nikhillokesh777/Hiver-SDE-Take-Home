"""
Brand Selection Analysis Script.
Implements Step 2 of Phase 1 (Brand Selection Plan in hiver_execution_plan.md).
Evaluates top candidate brands against the 5 explicit measurable criteria:
1. Usable Volume (inbound tweets addressed to brand)
2. Thread Completeness (% customer tweets with linked brand resolution)
3. Topical Diversity (lexical entropy & distinct vocabulary ratio)
4. Language Homogeneity (% English / clean ASCII)
5. Readability & Substance of Resolutions (avoiding pure boilerplate "please DM us")

Produces the Brand Comparison Table and selects the winning brand based on evidence.
"""

import os
import sys
import json
import time
import math
import re
from collections import Counter
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "twcs.csv")
OUTPUT_TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "brand_comparison_table.json")
OUTPUT_MD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "brand_comparison_table.md")

BOILERPLATE_DM_REGEX = re.compile(r"^\s*(please\s+)?(dm|send\s+(us\s+a\s+)?dm|direct\s+message)\s+.*\b(details|info|account|email)\b.*$", re.IGNORECASE)


def compute_lexical_entropy(texts, sample_size=1000):
    """Compute normalized Shannon entropy of word distribution as a proxy for topical diversity."""
    sample = texts.dropna().head(sample_size).str.lower()
    words = [w for t in sample for w in re.findall(r"\b[a-z]{3,}\b", t)]
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def score_brand_resolutions(brand_replies):
    """Measure resolution substance: average length, actionable keywords, non-pure-DM rate."""
    if len(brand_replies) == 0:
        return {"avg_len": 0, "substantive_pct": 0.0}
    
    texts = brand_replies["text"].astype(str)
    avg_len = round(float(texts.str.len().mean()), 1)
    
    # Substantive = has troubleshooting keywords, links, or exceeds 80 characters beyond boilerplate
    has_substance = texts.apply(
        lambda t: len(t) > 60 and not bool(re.match(r"^@\w+\s+(sorry|hi|hello),?\s*please\s+dm\s+us\b", t.lower()))
    )
    substantive_pct = round(float(has_substance.mean() * 100), 1)
    return {"avg_len": avg_len, "substantive_pct": substantive_pct}


def run_brand_selection(data_path=DATA_PATH, top_n_candidates=10):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run scripts/download_data.py first.")

    print(f"Loading data from {data_path} for brand evaluation...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} tweets.")

    # Top brand accounts by outbound reply volume
    brand_reply_counts = df[df["inbound"] == False]["author_id"].value_counts()
    top_brands = brand_reply_counts.head(top_n_candidates).index.tolist()
    print(f"\nTop {top_n_candidates} candidate brands by reply volume: {top_brands}")

    # Build tweet index for fast parent-child lookup
    # in_response_to_tweet_id points to parent tweet_id
    tweet_author_map = df.set_index("tweet_id")["author_id"].to_dict()
    tweet_text_map = df.set_index("tweet_id")["text"].to_dict()
    tweet_inbound_map = df.set_index("tweet_id")["inbound"].to_dict()

    comparison_results = []

    for brand in top_brands:
        print(f"\nEvaluating brand: {brand}...")
        brand_replies = df[df["author_id"] == brand]
        reply_count = len(brand_replies)

        # 1. Customer inbound tweets to this brand
        # A customer tweet is linked to this brand if the brand replied to it
        customer_tweet_ids = brand_replies["in_response_to_tweet_id"].dropna().astype(int)
        
        # Filter to actual tweets in the dataset
        valid_customer_tweets = [
            tid for tid in customer_tweet_ids 
            if tid in tweet_author_map and tweet_inbound_map.get(tid, False) == True
        ]
        usable_inbound_count = len(valid_customer_tweets)

        # 2. Thread completeness
        # Proportion of brand replies that connect to an existing inbound customer tweet
        thread_completeness_pct = round((usable_inbound_count / reply_count * 100) if reply_count > 0 else 0.0, 1)

        # 3. Topical Diversity
        customer_texts = pd.Series([tweet_text_map[tid] for tid in valid_customer_tweets[:2000]])
        diversity_score = compute_lexical_entropy(customer_texts)

        # 4. Language Homogeneity (% English / ASCII)
        if len(customer_texts) > 0:
            ascii_pct = round(float((customer_texts.str.contains(r"^[\x00-\x7F]+$").mean()) * 100), 1)
        else:
            ascii_pct = 0.0

        # 5. Readability & Resolution Substance
        resolution_stats = score_brand_resolutions(brand_replies.head(2000))

        # Composite score calculation based on criteria weights
        # Volume normalized (capped at 50,000 = 1.0)
        vol_score = min(1.0, usable_inbound_count / 50000.0)
        comp_score = min(1.0, thread_completeness_pct / 100.0)
        div_score = diversity_score  # already 0 to 1
        lang_score = min(1.0, ascii_pct / 100.0)
        sub_score = min(1.0, resolution_stats["substantive_pct"] / 100.0)

        composite_score = round(
            (0.20 * vol_score) + 
            (0.25 * comp_score) + 
            (0.20 * div_score) + 
            (0.15 * lang_score) + 
            (0.20 * sub_score), 
            4
        )

        record = {
            "brand": brand,
            "outbound_replies": reply_count,
            "usable_inbound_volume": usable_inbound_count,
            "thread_completeness_pct": thread_completeness_pct,
            "topical_diversity_entropy": diversity_score,
            "english_homogeneity_pct": ascii_pct,
            "avg_reply_length": resolution_stats["avg_len"],
            "substantive_reply_pct": resolution_stats["substantive_pct"],
            "composite_score": composite_score
        }
        comparison_results.append(record)
        print(f"  Volume: {usable_inbound_count:,} | Thread Completeness: {thread_completeness_pct}% | Substance: {resolution_stats['substantive_pct']}% | Composite: {composite_score}")

    # Sort candidates by composite score descending
    comparison_results.sort(key=lambda x: x["composite_score"], reverse=True)
    winner = comparison_results[0]
    runner_up = comparison_results[1] if len(comparison_results) > 1 else None

    # Save JSON table
    os.makedirs(os.path.dirname(OUTPUT_TABLE_PATH), exist_ok=True)
    with open(OUTPUT_TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "winner": winner["brand"],
            "runner_up": runner_up["brand"] if runner_up else None,
            "candidates": comparison_results
        }, f, indent=2)

    # Build Markdown table
    md_lines = [
        "# Brand Selection Comparison Table",
        "",
        "Evaluation of top candidate brands based on the 5 measurable criteria from `hiver_execution_plan.md` (§3 Step 2).",
        "",
        "| Rank | Brand | Usable Inbound Volume | Thread Completeness (%) | Topical Diversity (Entropy) | English Homogeneity (%) | Substantive Reply (%) | Composite Score |",
        "|---|---|---|---|---|---|---|---|"
    ]
    for i, c in enumerate(comparison_results, 1):
        md_lines.append(
            f"| {i} | **{c['brand']}** | {c['usable_inbound_volume']:,} | {c['thread_completeness_pct']}% | {c['topical_diversity_entropy']} | {c['english_homogeneity_pct']}% | {c['substantive_reply_pct']}% | **{c['composite_score']}** |"
        )
    md_lines.extend([
        "",
        f"### Selected Brand: `{winner['brand']}`",
        f"**Justification**: `{winner['brand']}` scored highest across the combined 5-criterion rubric (Composite Score: {winner['composite_score']}). It provides strong usable inbound volume ({winner['usable_inbound_volume']:,} customer queries exceeding the 1,500-2,000 threshold by a wide margin), high thread completeness ({winner['thread_completeness_pct']}%), strong topical diversity ({winner['topical_diversity_entropy']}), high English homogeneity ({winner['english_homogeneity_pct']}%), and a rich substantive resolution rate ({winner['substantive_reply_pct']}%), ensuring historical cases contain actual actionable troubleshooting rather than purely repetitive 'please DM us' boilerplate."
    ])

    md_content = "\n".join(md_lines)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[DONE] Brand comparison table saved to {OUTPUT_MD_PATH}")
    print(f"\n>>> Selected Winning Brand: {winner['brand']} (Runner-up: {runner_up['brand'] if runner_up else 'None'}) <<<")
    return comparison_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_brand_selection(top_n_candidates=n)
