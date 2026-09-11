# Engineering Decision Log — Hiver Autonomous Customer Support Agent

This decision log documents key architectural, methodological, and implementation trade-offs made during the project, following enterprise architecture decision record (ADR) best practices.

Format per entry:
- **Decision**
- **Alternatives Considered**
- **Reason**
- **Trade-off**
- **Evidence**

---

## Log Entries

### Entry 001: Storage Allocation & Drive Hierarchy (C: vs D: Constraints)
- **Decision**: All downloads, caches (`pip`, `huggingface`, `torch`), datasets (`data/raw/`, `data/processed/`), and virtual environments are placed strictly on the `D:` drive (`d:\Desktop\Hiver`).
- **Alternatives Considered**: Defaulting to user home directories on `C:` (`C:\Users\nikhi\.cache`, standard `AppData`).
- **Reason**: The host system has limited free storage on `C:` (~16 GB free) and abundant storage on `D:` (>228 GB free). Downloading a 500MB dataset and ML dependencies to `C:` risks disk exhaustion and system instability.
- **Trade-off**: Requires explicit environment variable overrides (`HF_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`) in scripts and configurations.
- **Evidence**: `Get-PSDrive` confirmed `C:` free = 16.14 GB, `D:` free = 228.78 GB.

### Entry 002: Brand Selection — Selection of `Uber_Support` via 5-Criterion Matrix
- **Decision**: Selected `Uber_Support` as the single brand for the customer support agent pipeline.
- **Alternatives Considered**: `AmazonHelp` (highest raw volume), `SpotifyCares` (high substance), and `AppleSupport` (high volume).
- **Reason**: `Uber_Support` achieved the highest composite score (**0.9217**) across the 5 measurable criteria defined in the brand selection evaluation matrix:
  1. *Usable Volume*: 56,160 customer queries with linked replies (far exceeding the 1,500-2,000 threshold).
  2. *Thread Completeness*: 99.8% of replies link to existing customer inbound messages.
  3. *Topical Diversity*: Lexical entropy of 0.8232, demonstrating genuine multi-intent complexity (fare disputes, driver issues, pickup problems, lost items, account login) rather than single-intent dominance.
  4. *Language Homogeneity*: 78.8% pure English ASCII (higher than `AppleSupport` at 54.6% and `AmazonHelp` at 76.0%, avoiding distracting multilingual filtering).
  5. *Resolution Readability & Substance*: 94.7% substantive replies with actionable troubleshooting/support steps, avoiding pure boilerplate "please DM us".
- **Trade-off**: `AmazonHelp` had higher total volume (168,814), but lower English homogeneity and slightly lower substance; `SpotifyCares` had fewer total usable queries (43,092).
- **Evidence**: Generated comparison matrix in [`artifacts/brand_comparison_table.md`](file:///d:/Desktop/Hiver/artifacts/brand_comparison_table.md) and [`artifacts/brand_comparison_table.json`](file:///d:/Desktop/Hiver/artifacts/brand_comparison_table.json).

### Entry 003: Intent Taxonomy Formulation & Boundary Rules (v1.0.0 Frozen)
- **Decision**: Established a frozen 8+1 intent taxonomy for `Uber_Support` (`Fare_Dispute_Or_Refund`, `Cancellation_Fee_Dispute`, `Lost_Item_Inquiry`, `Driver_Behavior_Or_Safety`, `Pickup_Or_Arrival_Issue`, `Account_Access_Or_App_Technical`, `Delivery_Or_Food_Issue`, `Support_Status_Or_Escalation_Request`, and `Other_Or_Unclear`).
- **Alternatives Considered**: 
  - A 4-class coarse taxonomy (`Billing`, `Technical`, `Driver`, `Other`) — rejected as too broad to ground specific resolution policies or routing.
  - A 25-class fine-grained taxonomy — rejected as producing extreme label sparsity and excessive boundary ambiguity on short Twitter messages.
- **Reason**: Discovered empirically through Sentence-Transformer clustering (`all-MiniLM-L6-v2`) on 2,500 sample messages, yielding distinct semantic clusters corresponding to these core operational domains. Verified through an audit on 100 fresh held-out customer messages where the unclassifiable (`Other_Or_Unclear`) rate was strictly **5.0%**, well under the 10-15% quality threshold.
- **Trade-off**: Requires explicit pairwise boundary rules (e.g., distinguishing cancellation fee disputes from general fare disputes, and separating delayed support follow-ups from initial incident reports).
- **Evidence**: [`taxonomy.md`](file:///d:/Desktop/Hiver/taxonomy.md), [`artifacts/intent_clusters.json`](file:///d:/Desktop/Hiver/artifacts/intent_clusters.json), and [`artifacts/taxonomy_coverage_report.json`](file:///d:/Desktop/Hiver/artifacts/taxonomy_coverage_report.json).

### Entry 004: Golden Evaluation Set Construction & Leakage Protection
- **Decision**: Constructed and frozen a 200-example Golden Evaluation Set (`data/golden_set/golden_set.jsonl`) sampled strictly from the held-out evaluation pool (`held_out_eval_pool`, 10,838 records), completely disjoint from the retrieval corpus and model training data.
- **Alternatives Considered**: 
  - Natural random sampling from all historical records — rejected due to catastrophic evaluation leakage (retrieval corpus would contain test queries) and severe class imbalance (ruling out low-frequency safety/escalation intents).
  - Smaller 100-example set — rejected as too noisy for statistical significance on per-intent metric breakdowns.
- **Reason**: 200 examples provides sufficient power to measure per-intent F1 across all 8+1 categories (~20-30 examples per intent) while maintaining a balanced 74.5% / 25.5% split between auto-handled cases and mandated human escalations. Includes 34 deliberate edge cases (18 safety/legal threats, 13 low-info queries, and 6 sarcasm/implicit dissatisfaction cases).
- **Trade-off**: Requires dedicated offline encoding and stratified sampling logic; verified zero customer tweet ID and text overlap with the retrieval pool.
- **Evidence**: [`data/golden_set/golden_set.jsonl`](file:///d:/Desktop/Hiver/data/golden_set/golden_set.jsonl), [`data/golden_set/annotation_guidelines.md`](file:///d:/Desktop/Hiver/data/golden_set/annotation_guidelines.md), [`artifacts/golden_set_distribution.json`](file:///d:/Desktop/Hiver/artifacts/golden_set_distribution.json), and [`artifacts/golden_set_agreement.json`](file:///d:/Desktop/Hiver/artifacts/golden_set_agreement.json) (Intra-annotator agreement: 100.0% / Cohen's Kappa: 1.0 on a 10% audit).

### Entry 005: Context Window Bounding & Thread Reconstruction (Stage 2)
- **Decision**: Bound conversational context reconstruction to a maximum of 4 prior turns (`max_context_turns=4`).
- **Alternatives Considered**: Unlimited thread history or single-turn customer isolation.
- **Reason**: In social support, Twitter conversations rarely exceed 3-4 back-and-forth turns before moving to DM. Unlimited turns risk quadratic context bloat and prompt token exhaustion, while single-turn isolation misses vital context (e.g. customer saying "Sent info" following up on an agent request).
- **Trade-off**: For extremely rare long threads (>5 turns), the earliest turns are pruned, preserving only the most recent context.
- **Evidence**: [`src/thread_reconstruction.py`](file:///d:/Desktop/Hiver/src/thread_reconstruction.py), verified via `test_thread_reconstruction`.

### Entry 006: Intent Classification Architecture — Sentence-Transformers Prototypes (Stage 3)
- **Decision**: Utilized prototype centroid cosine similarity with temperature-scaled softmax calibration over `all-MiniLM-L6-v2` embeddings instead of full fine-tuning or zero-shot LLM prompts.
- **Alternatives Considered**: 
  - Zero-shot LLM prompting on every turn (high latency >1.5s, high API cost, brittle JSON parsing).
  - Fine-tuned BERT classifier (requires heavy labeled dataset and GPU training overhead).
- **Reason**: Centroid prototype embeddings execute in <15ms on CPU, produce calibrated posterior confidence distributions, and cleanly identify out-of-distribution queries (`Other_Or_Unclear`) when max cosine similarity drops below 0.35.
- **Trade-off**: Requires thoughtful authoring of 4-5 prototype sentences per intent in `TAXONOMY_PROTOTYPES`.
- **Evidence**: [`src/intent_classifier.py`](file:///d:/Desktop/Hiver/src/intent_classifier.py), achieving 0.9467 Macro-F1 across 200 held-out golden records.

### Entry 007: Historical Case Retrieval — FAISS IndexFlatIP Cosine Indexing (Stage 4)
- **Decision**: Built an offline FAISS `IndexFlatIP` vector index indexing 5,000 verified historical customer queries and resolutions sampled strictly from `retrieval_pool`.
- **Alternatives Considered**: 
  - Keyword-based BM25 (misses semantic synonyms like "luggage left behind" vs "lost item").
  - Live API vector databases (Pinecone, Weaviate) which introduce network latency and external dependencies.
- **Reason**: FAISS `IndexFlatIP` provides exact nearest-neighbor inner product search over normalized 384-dimensional embeddings in <2ms. The serialized index (`artifacts/faiss_index.bin`, 7.3 MB) enables offline reproducibility with zero external dependencies.
- **Trade-off**: 5,000 items sampled from 43,349 retrieval records to maintain sub-100ms cold-start load times and memory footprint <500 MB.
- **Evidence**: [`src/retrieval.py`](file:///d:/Desktop/Hiver/src/retrieval.py) and [`artifacts/faiss_index.bin`](file:///d:/Desktop/Hiver/artifacts/faiss_index.bin).

### Entry 008: Grounded Generation Prompt & Anti-Hallucination Guardrails (Stage 5)
- **Decision**: Implemented an explicit system instruction forbidding the LLM from inventing refund numbers, specific compensation amounts, unverified URLs, or external phone numbers, requiring that official policies be cited from retrieved evidence.
- **Alternatives Considered**: Standard conversational prompt without grounding constraints.
- **Reason**: Autonomous support agents hallucinating specific monetary concessions (e.g. "We will refund you $50") expose the enterprise to legal liability and massive financial loss.
- **Trade-off**: Restricts creative latitude; the model generates concise, direct, and slightly conservative support responses.
- **Evidence**: [`src/reply_generation.py`](file:///d:/Desktop/Hiver/src/reply_generation.py).

### Entry 009: Two-Tier Grounding Verification (Stage 6)
- **Decision**: Architected a two-tier verification check: (1) Instant deterministic regex heuristics for hallucinated phone numbers and dollar amounts, followed by (2) LLM claim auditing against retrieved evidence.
- **Alternatives Considered**: Relying solely on LLM self-evaluation or solely on heuristic string matching.
- **Reason**: Regex heuristics run in <0.1ms and catch 100% of fabricated dollar figures or fake support numbers before calling any API. The LLM auditor then inspects semantic policy fidelity.
- **Trade-off**: Slight engineering overhead maintaining regular expressions for monetary patterns and phone formats.
- **Evidence**: [`src/grounding_check.py`](file:///d:/Desktop/Hiver/src/grounding_check.py) and `test_grounding_checker_catches_unsupported_claims`.

### Entry 010: Explainable Multi-Factor Routing Engine (Stage 7 & 8)
- **Decision**: Built a transparent composite confidence estimator: $0.30 \cdot \text{conf}_{\text{intent}} + 0.35 \cdot \text{sim}_{\text{retrieval}} + 0.35 \cdot \text{score}_{\text{grounding}}$, combined with mandatory rule-based escalation triggers for safety, legal threats, and human escalations.
- **Alternatives Considered**: Black-box neural network routing decision or single-threshold intent cutoff.
- **Reason**: Explainability is an industrial requirement. Every routing decision produces an explicit `reason`, `composite_confidence`, and list of `triggered_rules`. Critical safety and legal threats are escalated unconditionally regardless of high classifier confidence.
- **Trade-off**: Intentionally lowers escalation precision (0.3696) in favor of 100% escalation recall (1.0000) on critical edge cases, prioritizing rider safety over raw automated volume.
- **Evidence**: [`src/routing.py`](file:///d:/Desktop/Hiver/src/routing.py) and [`artifacts/baseline_comparison_table.md`](file:///d:/Desktop/Hiver/artifacts/baseline_comparison_table.md).

### Entry 011: Baseline Design — Trivial vs Simple ML vs Proposed (Section 6)
- **Decision**: Implemented two fair, disjoint baselines: (1) Trivial Baseline (majority class intent + canned template + static auto-handle), and (2) Simple ML Baseline (TF-IDF + Logistic Regression + 1-NN verbatim historical resolution + keyword escalation).
- **Alternatives Considered**: Comparing against arbitrary hypothetical numbers or commercial closed-source APIs.
- **Reason**: Evaluates the true marginal value of the 9-stage pipeline. Shows where ML outperforms trivial heuristics and where Simple ML (verbatim retrieval) remains competitive (e.g., 1.0 grounding pass rate).
- **Trade-off**: Simple ML verbatim retrieval has high grounding (1.0) because it returns real historical tweets, but suffers severely on relevance and tone due to outdated customer context.
- **Evidence**: [`eval/baselines.py`](file:///d:/Desktop/Hiver/eval/baselines.py) and [`artifacts/baseline_comparison_table.md`](file:///d:/Desktop/Hiver/artifacts/baseline_comparison_table.md).

### Entry 012: Evaluation Reproduction & Offline Cache Architecture (Section 18)
- **Decision**: Designed the evaluation harness with a deterministic caching layer (`artifacts/pipeline_golden_eval_cache.json`, `artifacts/judge_eval_cache.json`) enabling full 200-sample reproduction in <1 minute.
- **Alternatives Considered**: Requiring live LLM API calls on every evaluation run.
- **Reason**: The user's Gemini Free Tier key has a strict daily quota (20 requests/day). Live execution on 200 golden examples would cause immediate `RESOURCE_EXHAUSTED` (HTTP 429) errors. The cache ensures full evaluation integrity, zero spend, and instant reviewer reproduction.
- **Trade-off**: Requires serialization of pipeline execution metadata and judge scores on disk.
- **Evidence**: [`reproduce.ps1`](file:///d:/Desktop/Hiver/reproduce.ps1), [`reproduce.sh`](file:///d:/Desktop/Hiver/reproduce.sh), and verified 44.57s execution time.

