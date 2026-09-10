"""
Intent Discovery Script.
Executes Step 1 of Phase 2 (Intent Discovery Plan in hiver_execution_plan.md).
Embeds a development sample of inbound customer messages using Sentence-Transformers,
runs KMeans clustering across candidate cluster counts (k=6..10),
and extracts cluster medoids and representative messages for taxonomy formulation.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set cache directories to D: drive
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "torch")

WORKING_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "brand_working_set.csv")
OUTPUT_CLUSTER_REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "intent_clusters.json")


def discover_intents(sample_size=2500, k_candidates=(6, 8, 10), seed=42):
    if not os.path.exists(WORKING_SET_PATH):
        raise FileNotFoundError(f"Working set not found at {WORKING_SET_PATH}. Run scripts/create_brand_working_set.py first.")

    print(f"Loading working set from {WORKING_SET_PATH}...")
    df = pd.read_csv(WORKING_SET_PATH, low_memory=False)

    # Use strictly retrieval_pool for discovery to prevent evaluation leakage
    dev_df = df[df["split"] == "retrieval_pool"].copy()
    print(f"Available retrieval_pool records: {len(dev_df):,}")

    if len(dev_df) > sample_size:
        sample_df = dev_df.sample(sample_size, random_state=seed).reset_index(drop=True)
    else:
        sample_df = dev_df.reset_index(drop=True)

    print(f"Selected {len(sample_df)} sample messages for intent discovery.")

    # Embed using SentenceTransformer
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model 'all-MiniLM-L6-v2' (cached on D: drive)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.environ["HF_HOME"])

    texts = sample_df["customer_text_clean"].tolist()
    print("Generating sentence embeddings...")
    start_t = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Computed {len(embeddings)} embeddings in {time.time() - start_t:.1f}s.")

    # Evaluate candidate k values
    print("\nEvaluating KMeans clustering across candidate k values...")
    clustering_results = {}
    best_k = k_candidates[0]
    best_score = -1.0

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=1000, random_state=seed)
        clustering_results[k] = {
            "silhouette_score": round(float(score), 4),
            "inertia": round(float(km.inertia_), 2)
        }
        print(f"  k={k}: Silhouette Score = {score:.4f} | Inertia = {km.inertia_:.1f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nOptimal candidate cluster count based on silhouette: k={best_k}")

    # Run final KMeans with best_k
    final_km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    cluster_labels = final_km.fit_predict(embeddings)
    centroids = final_km.cluster_centers_

    # TF-IDF per cluster to extract top distinguishing keywords
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf.fit(texts)
    feature_names = np.array(tfidf.get_feature_names_out())

    cluster_summaries = []
    for c_id in range(best_k):
        c_mask = cluster_labels == c_id
        c_indices = np.where(c_mask)[0]
        c_size = len(c_indices)
        c_pct = round(c_size / len(sample_df) * 100, 1)

        # Distances to cluster centroid
        c_embeddings = embeddings[c_indices]
        centroid = centroids[c_id]
        distances = np.linalg.norm(c_embeddings - centroid, axis=1)
        nearest_indices = c_indices[np.argsort(distances)[:10]]

        # Top keywords by average TF-IDF in cluster
        c_texts = [texts[i] for i in c_indices]
        c_tfidf = tfidf.transform(c_texts)
        mean_tfidf = np.asarray(c_tfidf.mean(axis=0)).flatten()
        top_keyword_idx = np.argsort(mean_tfidf)[::-1][:8]
        top_keywords = feature_names[top_keyword_idx].tolist()

        representative_examples = [texts[idx] for idx in nearest_indices]

        cluster_summaries.append({
            "cluster_id": c_id,
            "size": c_size,
            "percentage": c_pct,
            "top_keywords": top_keywords,
            "representative_messages": representative_examples
        })

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(sample_df),
        "k_scan_results": clustering_results,
        "selected_k": best_k,
        "clusters": cluster_summaries
    }

    os.makedirs(os.path.dirname(OUTPUT_CLUSTER_REPORT), exist_ok=True)
    with open(OUTPUT_CLUSTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Intent discovery report saved to {OUTPUT_CLUSTER_REPORT}")

    # Print summary of clusters
    for c in cluster_summaries:
        print(f"\n=== Cluster {c['cluster_id']} ({c['size']} msgs, {c['percentage']}%) ===")
        print(f"  Keywords: {', '.join(c['top_keywords'])}")
        print(f"  Example 1: {c['representative_messages'][0][:120]}...")
        print(f"  Example 2: {c['representative_messages'][1][:120]}...")

    return report


if __name__ == "__main__":
    discover_intents()
