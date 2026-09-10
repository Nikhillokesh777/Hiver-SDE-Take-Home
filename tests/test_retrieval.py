"""
Tests for Stage 4: Historical Case Retrieval Module.
Verifies offline FAISS IndexFlatIP search, deterministic top-k results, similarity metrics, and empty queries.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.retrieval import HistoricalCaseRetriever, RetrievalResult


@pytest.fixture(scope="module")
def retriever():
    return HistoricalCaseRetriever(top_k=3)


def test_retrieval_deterministic_top_k(retriever):
    """Verify retrieval returns exactly the requested top_k cases."""
    res_3 = retriever.retrieve("I left my laptop in the car", top_k=3)
    assert isinstance(res_3, RetrievalResult)
    assert len(res_3.retrieved_cases) == 3
    assert res_3.max_similarity > 0.0

    res_5 = retriever.retrieve("I left my laptop in the car", top_k=5)
    assert len(res_5.retrieved_cases) == 5


def test_retrieval_case_metadata_integrity(retriever):
    """Verify all retrieved cases contain valid case IDs, queries, resolutions, and similarity scores."""
    res = retriever.retrieve("Overcharged on cancellation fee", top_k=3)
    for case in res.retrieved_cases:
        assert isinstance(case.case_id, int)
        assert len(case.customer_text) > 0
        assert len(case.brand_resolution) > 0
        assert -1.0 <= case.similarity_score <= 1.0


def test_retrieval_evidence_text_formatting(retriever):
    """Verify evidence text is structured with clear case delimiters for downstream prompt grounding."""
    res = retriever.retrieve("Where is my driver?", top_k=2)
    assert "[Historical Case #" in res.evidence_text
    assert "Customer Query:" in res.evidence_text
    assert "Support Resolution:" in res.evidence_text


def test_retrieval_empty_query_handling(retriever):
    """Verify empty query returns safe empty RetrievalResult without crashing."""
    res = retriever.retrieve("   ", top_k=3)
    assert len(res.retrieved_cases) == 0
    assert res.max_similarity == 0.0
    assert res.evidence_text == ""
