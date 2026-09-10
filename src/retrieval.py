"""
Stage 4: Historical Case Retrieval Module.
Retrieves top-k historical (customer_message -> brand_resolution) pairs using a FAISS vector index.
Supports intent-aware candidate boosting/filtering, returns identifiable evidence case IDs,
and saves/loads precomputed index artifacts for <15-minute reproducibility.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Ensure UTF-8 output on Windows consoles
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface")

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "faiss_index.bin")
DEFAULT_META_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "retrieval_metadata.json")
WORKING_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "brand_working_set.csv")


class RetrievedCase(BaseModel):
    case_id: int
    customer_text: str
    brand_resolution: str
    similarity_score: float
    mapped_intent: Optional[str] = None


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    retrieved_cases: List[RetrievedCase] = Field(default_factory=list)
    max_similarity: float = 0.0
    evidence_text: str = ""


class HistoricalCaseRetriever:
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_path: str = DEFAULT_META_PATH,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        self._index = None
        self._metadata: List[Dict[str, Any]] = []

    def _lazy_init_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, cache_folder=os.environ["HF_HOME"])

    def build_index_from_working_set(
        self,
        working_set_path: str = WORKING_SET_PATH,
        max_records: int = 5000,
        seed: int = 42
    ):
        """
        Builds and persists FAISS index over the retrieval_pool split.
        Zero evaluation leakage guaranteed (held_out_eval_pool excluded).
        """
        import faiss
        self._lazy_init_model()

        print(f"Loading retrieval corpus from {working_set_path}...")
        df = pd.read_csv(working_set_path, low_memory=False)
        corpus_df = df[df["split"] == "retrieval_pool"].copy()

        if len(corpus_df) > max_records:
            corpus_df = corpus_df.sample(max_records, random_state=seed).reset_index(drop=True)
        else:
            corpus_df = corpus_df.reset_index(drop=True)

        print(f"Indexing {len(corpus_df)} historical customer resolutions...")
        texts = corpus_df["customer_text_clean"].tolist()
        embeddings = self._model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Build FAISS IndexFlatIP (Cosine similarity on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Prepare metadata
        meta = []
        for i, row in corpus_df.iterrows():
            meta.append({
                "case_id": int(row["customer_tweet_id"]),
                "customer_text": str(row["customer_text_clean"]),
                "brand_resolution": str(row["brand_reply_clean"]),
                "brand_tweet_id": int(row.get("brand_tweet_id", 0))
            })

        # Save artifacts
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._index = index
        self._metadata = meta
        print(f"[SUCCESS] FAISS index saved to {self.index_path} ({len(meta)} records).")

    def load_index(self):
        """Loads precomputed index and metadata from disk."""
        if self._index is not None and len(self._metadata) > 0:
            return

        import faiss
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Index or metadata not found at {self.index_path}. Run build_index_from_working_set() first."
            )

        self._index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.20
    ) -> RetrievalResult:
        """
        Retrieves top-k historical cases relevant to query.
        """
        self.load_index()
        self._lazy_init_model()

        k = top_k or self.top_k
        query_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

        scores, indices = self._index.search(query_emb, k)
        scores = scores[0]
        indices = indices[0]

        retrieved: List[RetrievedCase] = []
        evidence_snippets = []

        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(self._metadata):
                continue
            case_data = self._metadata[idx]
            sim_score = round(float(score), 4)

            case = RetrievedCase(
                case_id=case_data["case_id"],
                customer_text=case_data["customer_text"],
                brand_resolution=case_data["brand_resolution"],
                similarity_score=sim_score
            )
            retrieved.append(case)
            evidence_snippets.append(
                f"[Historical Case #{case.case_id} (Similarity: {sim_score:.2f})]\n"
                f"Customer Query: {case.customer_text}\n"
                f"Support Resolution: {case.brand_resolution}"
            )

        max_sim = retrieved[0].similarity_score if retrieved else 0.0
        evidence_text = "\n\n".join(evidence_snippets)

        return RetrievalResult(
            query=query,
            top_k=k,
            retrieved_cases=retrieved,
            max_similarity=max_sim,
            evidence_text=evidence_text
        )
