# Autonomous Customer Support Agent & Industrial Evaluation Harness
**Deliverable**: Hiver SDE Intern Take-Home Project  
**Target Brand**: `Uber_Support`  
**Dataset**: Kaggle Twitter Customer Support Dataset (2.81M records)  
**Evaluation Set**: 200 Stratified Ground-Truth Golden Examples  
**Reproduction Time**: **44.57 seconds** (Guaranteed <15 minutes offline)

---

## 📋 Take-Home Deliverables Verification Matrix

| # | Deliverable | Primary File / Artifact | Key Highlights |
| :- | :--- | :--- | :--- |
| **1** | **Runnable Pipeline & Reproduction** | [`reproduce.ps1`](reproduce.ps1) / [`reproduce.sh`](reproduce.sh) | Full modular test suite (38 tests) & 3-system benchmark execute in **~90s** (<15m requirement) with zero API spend. Interactive CLI in [`scripts/demo.py`](scripts/demo.py). |
| **2** | **Golden Evaluation Set (150–250 examples)** | [`data/golden_set/golden_set.jsonl`](data/golden_set/golden_set.jsonl) | **200 hand-labelled examples** strictly from `held_out_eval_pool` + [sampling & labelling note](data/golden_set/annotation_guidelines.md) + [100% agreement audit](artifacts/golden_set_agreement.json). |
| **3** | **Evaluation Harness & LLM-as-Judge** | [`eval/metrics.py`](eval/metrics.py) & [`eval/llm_judge.py`](eval/llm_judge.py) | Automated metrics (Macro-F1, Escalation Recall) + 5-dimension rubric + [human-judge agreement validation](artifacts/judge_validation_report.json). |
| **4** | **Comprehensive Report** | [`report.md`](report.md) | Problem framing & non-goals, 3-system comparison, Top 5 concrete failure modes with real examples, mandatory *"What is misleading about my headline number?"* critique, and 1-week roadmap. |
| **5** | **Engineering Decision Log** | [`decision_log.md`](decision_log.md) | **15 non-obvious engineering decisions** with explicit alternatives considered, trade-offs, and empirical evidence. |

---

## 1. Quick Reproduction (< 2 Minutes)

To verify the entire evaluation harness, all 38 unit/integration tests across 9 test modules, and generate all benchmark comparison tables offline with zero API spend:

### On Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File reproduce.ps1
```

### On Linux / macOS (Bash):
```bash
chmod +x reproduce.sh
./reproduce.sh
```

### Manual Run:
```bash
# Run full automated modular test suite (9 test files, 38 tests)
pytest tests/ -v

# Run 3-system evaluation harness on 200 golden examples
python eval/run_eval.py --use-cache --limit 200
```

---

## 2. Benchmark Summary & Headline Results

Evaluated across **200 strictly held-out golden records** disjoint from all training and retrieval data:

| Metric | Role | Trivial Baseline | Simple ML Baseline | Proposed Grounded System | Marginal Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Macro-F1** | Headline | 0.0211 | 0.4305 | **0.9467** | **+0.5162 (+120%)** |
| **Escalation Recall** | Safety Headline | 0.0000 | 0.2941 | **1.0000 (100%)** | **+0.7059 (+240%)** |
| **Missed Escalations (FN)** | Risk Headline | 51 (Critical) | 36 (Moderate) | **0 (Zero Risk)** | **-36 Dangerous Misses** |
| **Escalation Precision** | Operational Trade-off | 0.0000 | **1.0000** | 0.3696 | -0.6304 (Intentional trade-off) |
| **Autonomy Resolvability** | Trust Headline | 0.7450 | 0.8054 | **1.0000 (100%)** | **+0.1946** |
| **Grounding Pass Rate** | Trust Metric | 1.0000 | **1.0000 (Verbatim)** | 1.0000 | Baseline Parity |
| **LLM-Judge Quality (1-5)** | Quality Metric | 2.10 | 4.08 | **4.67 / 5.0** | **+0.59 points** |
| **Composite Reliability** | Weighted Overall | 0.3074 | 0.6098 | **0.8202** | **+0.2104** |
| **P50 Latency (ms)** | Compute Overhead | **0.0 ms** | 1.31 ms | 27.6 ms | Baseline Advantage |

*Composite Reliability Formula*: $0.35 \cdot \text{Macro-F1} + 0.35 \cdot \text{Escalation-F1} + 0.30 \cdot \text{Grounding-Rate}$.

---

## 3. Industrial Pipeline Architecture

The system operates as a cohesive 9-stage modular pipeline:

1. **Stage 1: Preprocessing & PII Masking** ([`src/preprocessing.py`](file:///d:/Desktop/Hiver/src/preprocessing.py))  
   Normalizes URLs, collapses redundant whitespace, and masks user handles to `@customer` while preserving `@uber_support` and sentiment emojis.
2. **Stage 2: Bounded Context Windowing** ([`src/thread_reconstruction.py`](file:///d:/Desktop/Hiver/src/thread_reconstruction.py))  
   Bounds conversational history to a maximum of 4 turns (`max_context_turns=4`) to eliminate context window bloat while maintaining referential coherence.
3. **Stage 3: Calibrated Intent Classification** ([`src/intent_classifier.py`](file:///d:/Desktop/Hiver/src/intent_classifier.py))  
   Cosine similarity against frozen prototype centroids using `all-MiniLM-L6-v2` with temperature-scaled softmax calibration ($T=0.20$). Detects out-of-distribution queries (`Other_Or_Unclear`) when max similarity $< 0.35$.
4. **Stage 4: FAISS Historical Case Retrieval** ([`src/retrieval.py`](file:///d:/Desktop/Hiver/src/retrieval.py))  
   Exact inner product cosine search (`IndexFlatIP`) over 5,000 verified historical customer resolutions sampled strictly from the disjoint `retrieval_pool`.
5. **Stage 5: Grounded Reply Generation** ([`src/reply_generation.py`](file:///d:/Desktop/Hiver/src/reply_generation.py))  
   Grounded generation with Gemini 2.5 Flash with explicit anti-hallucination prompt guardrails prohibiting fake refund numbers or unverified support phone lines.
6. **Stage 6: Two-Tier Grounding Verification** ([`src/grounding_check.py`](file:///d:/Desktop/Hiver/src/grounding_check.py))  
   Sub-millisecond regex heuristics checking for unverified dollar amounts and contact phone numbers, backed by semantic claim auditing against retrieved evidence.
7. **Stage 7 & 8: Confidence Estimation & Routing Engine** ([`src/routing.py`](file:///d:/Desktop/Hiver/src/routing.py))  
   Multi-factor composite confidence estimation ($0.30 \cdot \text{intent} + 0.35 \cdot \text{retrieval} + 0.35 \cdot \text{grounding}$) combined with mandatory rule-based escalation triggers for safety hazards, legal threats, and human escalations.
8. **Stage 9: Telemetry & Structured Logging** ([`src/pipeline.py`](file:///d:/Desktop/Hiver/src/pipeline.py))  
   Comprehensive JSON Lines telemetry logging with execution IDs, stage latencies, and explainable decision paths.

> 📖 **Deep Dive**: For a complete file-by-file execution walkthrough tracing which files activate, where the query is solved, and the full role of every file in the project, see [**`QUERY_LIFECYCLE.md`**](QUERY_LIFECYCLE.md).

---

## 4. Leakage Prevention Guarantee

To guarantee strict evaluation integrity:
- The dataset is partitioned into **two strictly disjoint pools**:
  - `retrieval_pool`: 43,349 records used exclusively for index creation and classifier development.
  - `held_out_eval_pool`: 10,838 records reserved exclusively for evaluation.
- The 200-example Golden Evaluation Set was drawn entirely from `held_out_eval_pool`.
- Zero overlap in Tweet IDs and customer query text is enforced and asserted by automated tests in [`tests/test_data_integrity.py`](file:///d:/Desktop/Hiver/tests/test_data_integrity.py).

---

## 5. Directory Structure & Key Deliverables

```
D:\Desktop\Hiver\
├── artifacts\                          # Persisted reproducible artifacts
│   ├── baseline_comparison_table.md    # Final markdown comparison table
│   ├── evaluation_report.json          # Full evaluation report across 3 systems
│   ├── faiss_index.bin                 # Serialized FAISS IndexFlatIP (5,000 records)
│   ├── retrieval_metadata.json         # Retrieval corpus metadata
│   ├── judge_validation_report.json    # Human vs Judge agreement stats
│   ├── pipeline_golden_eval_cache.json # Cached pipeline outputs (200 records)
│   └── judge_eval_cache.json           # Cached LLM-Judge rubric scores
├── data\
│   ├── golden_set\
│   │   ├── golden_set.jsonl            # 200 ground-truth evaluation records
│   │   └── annotation_guidelines.md    # Explicit labelling rules & rubrics
│   └── processed\
│       ├── brand_working_set.csv       # Normalized Uber_Support dataset (54,187 rows)
│       └── split_metadata.json         # Split hashes & leakage verification
├── eval\
│   ├── baselines.py                    # Trivial & Simple ML Baselines
│   ├── metrics.py                      # Accuracy, Macro-F1, Recall, Autonomy rates
│   ├── llm_judge.py                    # 5-dimension rubric LLM-as-Judge
│   ├── judge_validation.py             # Human agreement & Spearman correlation
│   └── run_eval.py                     # Single-command evaluation runner
├── src\                                # 9-stage support pipeline modules
│   ├── preprocessing.py
│   ├── thread_reconstruction.py
│   ├── intent_classifier.py
│   ├── retrieval.py
│   ├── reply_generation.py
│   ├── grounding_check.py
│   ├── routing.py
│   └── pipeline.py
├── tests\
│   ├── test_data_integrity.py          # Schema, missing values, deduplication, 0-leakage assertion
│   ├── test_preprocessing.py           # Normalization, PII handle masking, URL replacing, emoji preservation
│   ├── test_thread.py                  # Multi-turn conversation handling & context turn bounding
│   ├── test_intent.py                  # Intent classification, calibration & OOD fallback
│   ├── test_retrieval.py               # Deterministic FAISS top-k retrieval & empty query handling
│   ├── test_grounding.py               # Unsupported claim detection & regex hallucination filters
│   ├── test_routing.py                 # Safety triggers, escalation rules & composite confidence math
│   ├── test_pipeline_e2e.py            # End-to-end execution, schema, resilience & telemetry logging
│   └── test_eval_metrics.py            # Baselines, Macro-F1, Escalation Recall/Precision & autonomy rates
├── QUERY_LIFECYCLE.md                  # File-by-file execution walkthrough & query lifecycle
├── decision_log.md                     # 15 industrial engineering decisions
├── report.md                           # Comprehensive technical report
├── taxonomy.md                         # Frozen 8+1 intent taxonomy v1.0.0
├── reproduce.ps1                       # Windows reproduction script
├── reproduce.sh                        # Linux/macOS reproduction script
└── requirements.txt
```

---

## 6. Setup & Interactive Usage

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell or Unix Bash

### Installation
```bash
# Clone the repository
git clone https://github.com/Nikhillokesh777/Hiver-SDE-Take-Home.git
cd Hiver-SDE-Take-Home

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up your Gemini API key for live generation
cp .env.example .env
```

### Interactive Query Execution

You can test any customer inquiry through the full 9-stage pipeline using either the command line or Python:

#### Method 1: Interactive CLI Menu
Launch the interactive terminal interface:
```bash
python scripts/demo.py
```
This lets you pick from pre-configured edge cases (safety hazards, cancellation disputes, lost items) or type any custom query.

#### Method 2: Direct Single-Query Command
Test any inquiry directly in one line:

```bash
# On Windows (PowerShell) - Note: Use single quotes to preserve '$' characters:
python scripts/demo.py 'My driver demanded that I pay him in cash even though the ride was already paid'

# Another example: Cancellation fee dispute
python scripts/demo.py 'My driver cancelled my ride and charged me a $5 cancellation fee, can I get a refund?'
```

#### Method 3: Programmatic Python API
```python
from src.pipeline import SupportAgentPipeline

# Initialize the end-to-end pipeline
pipeline = SupportAgentPipeline()

# Process an incoming support tweet
result = pipeline.run("My driver cancelled my trip and charged me a cancellation fee, can I get a refund?")

print(f"Predicted Intent:    {result.predicted_intent}")
print(f"Confidence Score:    {result.intent_confidence:.4f}")
print(f"Routing Decision:    {result.routing_decision}")
print(f"Routing Reason:      {result.routing_reason}")
print(f"Grounding Passed:    {result.grounding_pass}")
print(f"Generated Response:  {result.generated_reply}")
```

---

### 💡 Practical Input & Expected Output Examples

The table and breakdown below illustrate representative inputs across distinct operational scenarios, demonstrating how the agent classifies, grounds, and routes customer inquiries:

#### Example 1: Standard Auto-Handled Cancellation Fee Dispute
* **Query:** `My driver cancelled my ride and charged me a $5 cancellation fee, can I get a refund?`
* **Predicted Intent:** `Cancellation_Fee_Dispute` (Confidence: `0.88`)
* **Routing Decision:** `AUTO_HANDLE`
* **Routing Reason:** `High intent confidence (0.88), strong historical evidence (0.92), and verified grounded reply (1.00).`
* **Grounding Check:** `PASSED` (Score: `1.00`, 0 unsupported claims)
* **Expected Answer:**
  > *"We understand your concern regarding the cancellation fee. You can request a fee review directly in the Uber app by going to 'Help' > 'Review my fare or fees' for that specific trip. Our team will verify the driver's arrival time and assist you."*

#### Example 2: Standard Auto-Handled Lost Property Recovery
* **Query:** `I left my black backpack with my laptop in the back of my Uber last night, please help me contact the driver`
* **Predicted Intent:** `Lost_Item_Inquiry` (Confidence: `0.91`)
* **Routing Decision:** `AUTO_HANDLE`
* **Routing Reason:** `High intent confidence (0.91), strong historical evidence (0.94), and verified grounded reply (1.00).`
* **Grounding Check:** `PASSED` (Score: `1.00`, 0 unsupported claims)
* **Expected Answer:**
  > *"We're here to help you get your item back! The fastest way to contact your driver is through the Uber app: navigate to 'Help' > 'Find lost item' > 'Contact driver about lost item'. If you cannot reach the driver, visit [URL] so our support team can assist."*

#### Example 3: Critical Safety Hazard / Driver Threat (100% Safety Recall)
* **Query:** `The driver was driving recklessly at 90mph, ran two red lights, and threatened me when I asked him to slow down!`
* **Predicted Intent:** `Driver_Behavior_Or_Safety` (Confidence: `0.94`)
* **Routing Decision:** `ESCALATE`
* **Routing Reason:** `Mandatory safety/legal trigger: message references reckless driving, threat, or physical safety hazard.`
* **Grounding Check:** `PASSED` (Score: `0.85`)
* **Expected Answer:**
  > *"Your safety is our absolute priority. We take reports of dangerous driving and threatening behavior extremely seriously and have escalated this incident directly to our Specialized Safety Response Team. Please send us a DM with your account email and trip details so we can investigate immediately."*

#### Example 4: Managerial Escalation / Repeated Unanswered Requests
* **Query:** `I have opened 3 tickets about my incorrect charge and your support bot keeps closing them without any response! I need to speak to a human supervisor now.`
* **Predicted Intent:** `Support_Status_Or_Escalation_Request` (Confidence: `0.89`)
* **Routing Decision:** `ESCALATE`
* **Routing Reason:** `Mandatory escalation policy: intent 'Support_Status_Or_Escalation_Request' requires direct human tier-2 supervisor handling.`
* **Grounding Check:** `PASSED` (Score: `0.85`)
* **Expected Answer:**
  > *"We sincerely apologize for the delay and frustration with your previous tickets. We have flagged this thread and routed it directly to a Tier-2 customer support supervisor for priority review. Please send us a DM with your registered email and ticket numbers."*

#### Example 5: Low-Information / Ambiguous Greeting
* **Query:** `@Uber_Support hey are you there? hello???`
* **Predicted Intent:** `Other_Or_Unclear` (Confidence: `0.78`)
* **Routing Decision:** `AUTO_HANDLE`
* **Routing Reason:** `Low-information greeting or vague inquiry; requires customer clarification before routing.`
* **Grounding Check:** `PASSED` (Score: `1.00`)
* **Expected Answer:**
  > *"Hi there! We are here and ready to help. Could you please share more details about your trip or the issue you are experiencing so our team can assist you?"*

---

## 7. Extended Technical Documentation

Detailed deep-dives, architectural analyses, and engineering decisions are documented in dedicated reports:

- [**Technical Evaluation Report (`report.md`)**](report.md): In-depth failure mode analysis (top 5 failure modes), misleading headline metric critiques, and production deployment recommendations.
- [**Engineering Decision Log (`decision_log.md`)**](decision_log.md): Architectural decision records (ADRs) covering model selection, FAISS index configuration, grounding verification tiers, and safety threshold trade-offs.
- [**Frozen Intent Taxonomy (`taxonomy.md`)**](taxonomy.md): Formal frozen 8+1 intent taxonomy specification (`v1.0.0`) with explicit negative boundaries, trigger phrases, and semantic clustering rationale.
- [**Annotation Guidelines (`data/golden_set/annotation_guidelines.md`)**](data/golden_set/annotation_guidelines.md): Rigorous labelling rubrics and edge case handling rules used for constructing the 200-sample Golden Set.

