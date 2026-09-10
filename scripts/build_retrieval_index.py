"""
Build and persist the FAISS historical case retrieval index.
Strictly indexes customer_text -> brand_resolution pairs from retrieval_pool (held_out_eval_pool excluded).
Guarantees 0 evaluation leakage and offline index availability for <15-minute evaluation reproduction.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(ROOT_DIR, ".cache", "torch")

sys.path.insert(0, ROOT_DIR)
from src.retrieval import HistoricalCaseRetriever

def main():
    print("=== Building FAISS Historical Retrieval Index ===", flush=True)
    start_t = time.time()
    
    retriever = HistoricalCaseRetriever()
    retriever.build_index_from_working_set(max_records=5000, seed=42)
    
    elapsed = time.time() - start_t
    print(f"=== FAISS Index successfully built in {elapsed:.2f}s ===", flush=True)

if __name__ == "__main__":
    main()
