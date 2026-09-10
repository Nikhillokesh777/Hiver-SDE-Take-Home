# Hiver SDE Intern Take-Home — Execution Plan

**Guiding principle:** *the proof is worth more than the system.* Every architectural choice below is deliberately boring; every evaluation choice below is deliberately rigorous. If you only have time for one, cut the architecture, not the evaluation.

This is a **plan**, not code. Anything that depends on what's actually in the data is flagged **[DECIDE AFTER DATA INSPECTION]** — do not pre-commit to those choices.

---

## 1. Executive Strategy

You are not being scored on building the most sophisticated agent. You are being scored on whether you can:

1. Make a defensible, evidence-based decision under ambiguity (which brand, which intents, which architecture).
2. Build something that actually runs end-to-end on a subsample, reproducibly, in <15 minutes.
3. Measure it honestly — including telling the evaluator where it's weak.
4. Explain every line of it live, under questioning.

The single biggest risk is **spending 80% of your time on the agent and 20% on evaluation**. Invert that ratio. A mediocre agent with a rigorous, honest evaluation and a sharp failure analysis will beat an impressive-looking agent with a single "94% accuracy!" headline number and no scrutiny of what that number means.

Concretely, your time budget should look like:

| Phase | % of total time |
|---|---|
| Data inspection & brand selection | 10% |
| Intent taxonomy | 10% |
| Golden set construction (labelling) | 20% |
| Agent build (all 3 stages) | 25% |
| Evaluation harness + LLM judge + validation | 20% |
| Failure analysis, report, decision log, README | 15% |

If you notice yourself deep in architecture tinkering while the golden set doesn't exist yet, stop — you're optimizing the wrong thing.

**Non-negotiables baked into the plan:**
- Every claim in your report must trace to a number you actually computed.
- The golden set is built and frozen *before* you look at how your final system performs on it (leakage prevention — see §7).
- The "misleading headline number" section is not a formality; treat it as seriously as the headline itself.

---

## 2. What the Evaluator Will Care About

Evaluators reading dozens of these submissions are pattern-matching for a small number of signals, fast.

**What separates average from excellent:**

| Average submission | Excellent submission |
|---|---|
| Picks the brand with the most rows | Picks the brand using a stated, measurable criterion, and shows the comparison table |
| Intents feel like a guess ("Billing", "Other", "Complaint") | Intents are derived from actual clustering/reading of the brand's data, with a documented discovery process |
| One aggregate accuracy number | Headline number + a section actively arguing why it overstates quality |
| "We used an LLM as judge" | Judge rubric shown, judge validated against ~30-50 human-labelled samples, agreement number (e.g. Cohen's kappa or % exact match) reported honestly, including where they disagree |
| Failure analysis is 2 generic bullet points | 5 concrete failures, each with a real transcript, root-cause hypothesis, and an explicit "is this worth fixing" judgment |
| README says "run main.py" | README reproduces headline numbers in <15 min from a cached/precomputed artifact, with exact commands |
| Baselines are token gestures | Baselines are described precisely enough that another engineer could reimplement them, and the comparison is apples-to-apples (same eval set, same metrics) |
| Decision log is generic ("used Python") | Decision log has real trade-offs with alternatives that were seriously considered and rejected for a stated reason |

**What evaluators are implicitly testing beyond the deliverables:**
- Can you tell the difference between "the model got it right" and "the eval is lenient"?
- Do you know the limits of your own system? (This is the #1 signal of engineering maturity.)
- Can you explain *why* you made each choice, not just *what* you chose?
- Did you use AI coding assistants as a tool (fine, expected) or did you let them make architectural decisions you can't defend (bad — you'll be asked to modify your own code live)?

---

## 3. Data Exploration & Brand Selection Plan

**Step 1 — Structural inspection (before any modeling).**
The public Kaggle "Customer Support on Twitter" dataset is widely documented to have a row-per-tweet structure with fields resembling: `tweet_id`, `author_id`, `inbound` (whether the tweet is customer→brand or brand→customer), `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`. **[DECIDE AFTER DATA INSPECTION]** — confirm the exact column names, types, and any nulls/encoding issues once you load the actual file; do not assume this schema is complete or correct until verified.

Run, and log the output of, at minimum:
- `df.shape`, `df.dtypes`, `df.isna().sum()`
- Distribution of tweets per `author_id` that is a brand (i.e., candidate "brands") — count of inbound tweets addressed to each
- Distribution of thread lengths (how many tweets link via `response_tweet_id`/`in_response_to_tweet_id` chains)
- Language distribution (a quick langdetect pass) — the dataset is known to contain non-English tweets
- Duplicate/near-duplicate text rate (exact dupes, and near-dupes via a cheap hash like MinHash or simple normalized-text match)
- Spot-read 30–50 raw threads manually, not just summary stats — you cannot pick a good taxonomy or brand from histograms alone

**Step 2 — Brand selection criteria (define these *before* looking at which brand "looks best"; otherwise you're fitting the criteria to your favorite brand after the fact).**

Score each of the top 8–10 candidate brands by inbound-tweet volume on:

1. **Volume** — enough inbound customer tweets to support a 150–250 example golden set *and* leave a disjoint pool for few-shot/retrieval examples, with room to spare. A rough floor: don't pick a brand with <1,500–2,000 usable inbound tweets after cleaning — you'll starve both the retrieval corpus and the golden set.
2. **Thread completeness** — high proportion of customer tweets that actually have a linked brand response (you need historical resolutions to ground replies on; a brand with mostly unanswered tweets is unusable for step 2 of the assignment).
3. **Topical diversity vs. coherence** — enough variety that intent classification is a real problem (not 95% one intent), but not so scattered that no small taxonomy can cover it.
4. **Language homogeneity** — overwhelmingly English, to avoid needing multilingual handling as a side-quest.
5. **Readability of resolutions** — spot-check 10 threads per candidate: do the brand's replies look like genuine resolutions (steps taken, links, "DM us") or mostly boilerplate ("we're sorry, please DM us")? A brand whose only reply pattern is "please DM us" gives you nothing to ground a reply generator on.

Build a small comparison table (brand × the 5 criteria, with actual computed numbers) and pick the winner with a one-paragraph justification. This table *is* a decision-log entry and doubles as evidence in your report.

**Step 3 — Handling noisy/duplicated/incomplete/multi-turn/ambiguous data (methodology, not final numbers):**

- **Noisy text:** normalize handles/URLs/emoji into placeholders rather than stripping them (emoji and links can be semantically meaningful, e.g. a link to a status page). Keep a raw-text column alongside the cleaned column — never destroy the original.
- **Duplicates:** dedupe on normalized text within the same brand; keep the earliest occurrence; log how many rows were removed and why (this number goes in the report, not invented).
- **Incomplete threads:** a customer tweet with no linked response can still be used for intent classification, but cannot be used for reply-grounding (no ground truth) or for baseline "brand actually said X" comparison — exclude those from the golden set's reply-quality slice, but you may still use them for the intent-taxonomy discovery step.
- **Multi-turn threads:** reconstruct a thread as an ordered list of turns using the `response_tweet_id`/`in_response_to_tweet_id` chain (or root-tweet grouping) — **[DECIDE AFTER DATA INSPECTION]** whether chains are reliably reconstructable or noisy/broken in practice; if broken often, document the fallback (e.g., group by conversation_id if present, or by time-window + author pair).
- **Ambiguous/multi-intent messages:** don't force a single label at the discovery stage — allow a message to carry a primary intent + optional secondary intent tag during annotation; decide the taxonomy's stance on this in §4.

**Step 4 — Reproducible sampling strategy.**
Fix a random seed. Define the exact sampling procedure in code/README terms:
- Filter to the selected brand's inbound tweets with a valid first response.
- Deduplicate.
- Stratify by (a) whether it's the first message in a thread vs. a follow-up, and (b) message length bucket (short/medium/long), so your sample isn't dominated by one message shape.
- Sample size for the "working set" you develop against (distinct from the frozen golden set — see §7) should be big enough to see variety but small enough to iterate fast, e.g. a few thousand rows, capped for compute/API cost.
Document the seed and exact filter/sort order so a second run reproduces the identical sample.

---

## 4. Intent Discovery Plan

**Do not start with a taxonomy in your head.** Discover it from data, then freeze it.

**Step 1 — Unsupervised pass.** Embed a large sample (e.g. 2,000–5,000) of the selected brand's inbound messages with a sentence embedding model, cluster (k-means with a scan over k, or HDBSCAN if you want to let the data suggest the count), and read 10–15 representative messages per cluster. This gives you a first-pass sense of the natural groupings — don't over-trust cluster boundaries, use them as reading guides.

**Step 2 — Manual synthesis into a small taxonomy.** From the clusters plus your own manual reading (you already spot-read 30–50 threads in §3 — extend that reading here), write down candidate intents as short, mutually-distinguishable categories. Target **6–10 intents** — small enough that each has real support (dozens of examples minimum), large enough to be meaningfully more useful than "complaint / question / other." Typical shapes for a support-Twitter brand (illustrative only — **[DECIDE AFTER DATA INSPECTION]** what actually applies to your brand): service outage/status inquiry, billing/refund dispute, account access/login issue, delivery/order status, product defect/return, how-to/informational question, complaint with no actionable ask (venting), escalation request/threat (e.g. "cancelling," "lawyer"), positive feedback/other.

**Step 3 — Rigor checklist for the taxonomy before freezing it:**
- Each intent has a one-sentence definition and 3 example messages, written down in a taxonomy doc.
- Intents are pairwise distinguishable — for any two intents, can you write a sentence explaining how an annotator tells them apart? If not, merge them.
- Coverage check: sample 100 fresh messages (not the ones used to build the taxonomy) and manually label them; if >10–15% don't fit any category or fit ambiguously into 3+, revise the taxonomy once — then freeze.
- Include an explicit "Other/Unclear" bucket, but track its rate; if it exceeds ~10%, that's a sign the taxonomy is missing something.

**Step 4 — Handling ambiguous/multi-intent messages.** Decide and document a policy, e.g.: label the *primary* actionable intent (the thing the agent must act on) as the single ground-truth label; optionally record a secondary intent as metadata for error analysis, but don't train/evaluate the classifier on multi-label unless you have the bandwidth — a single clean label your classifier and judge can agree on is worth more than a "correct" but unwieldy multi-label scheme.

**Step 5 — Freeze.** Once frozen, the taxonomy doesn't change again except to fix a definition typo. Any change after golden-set labelling starts invalidates prior labels — put the taxonomy in its own versioned file (`taxonomy.md` or `taxonomy.json`) and reference a version number in the decision log.

**Optional secondary use of Banking77:** only for calibrating your *methodology* for defining a small intent taxonomy from a larger one (Banking77 already has 77 fine-grained intents you could study for definition style), not as ground truth for the brand's actual intents — the assignment explicitly scopes Banking77 to intent methodology, not to Twitter-brand data.

---

## 5. System Architecture

Lean, sequential pipeline — no unnecessary agent frameworks, no unnecessary services. Each stage is a plain function with a typed input/output so it can be tested and logged independently.

```
raw tweet/thread
     │
     ▼
[1] Preprocessing ── normalize text, resolve mentions/links/emoji, detect language, strip PII-like handles
     │
     ▼
[2] Thread Reconstruction ── join customer + prior turns into an ordered conversation object (customer's current message + up to N prior turns of context)
     │
     ▼
[3] Intent Classification ── classifier over the frozen taxonomy; outputs (intent, confidence)
     │
     ▼
[4] Historical-Case Retrieval ── retrieve top-k similar past (customer message → brand resolution) pairs for this brand, filtered/boosted by matching intent
     │
     ▼
[5] Reply Generation ── LLM call grounded on the retrieved cases + current message + thread context; outputs a draft reply + a list of which retrieved case(s) it drew from
     │
     ▼
[6] Grounding / Evidence Check ── verify the draft reply's claims are actually supported by the retrieved cases (not hallucinated); attach an evidence score
     │
     ▼
[7] Confidence Estimation ── combine intent confidence, retrieval similarity score, and grounding score into a single confidence signal
     │
     ▼
[8] Auto-handle vs. Escalate ── threshold(s) on confidence + intent-specific rules (e.g. always escalate "legal threat" or "safety" intents regardless of confidence) → decision + stated reason
     │
     ▼
[9] Logging ── every stage's input/output, model/version, prompt version, and timing written to a structured log (JSONL) keyed by a run ID, for reproducibility and failure analysis
```

**Design notes per stage:**

- **[1] Preprocessing:** deterministic, no LLM calls — keep it cheap and fast so it can run over the full subsample instantly.
- **[2] Thread reconstruction:** bound context window (e.g. last 3–5 turns) to keep prompts small and costs predictable; **[DECIDE AFTER DATA INSPECTION]** the right window size based on how long real threads run.
- **[3] Intent classification:** start with the simplest thing that could work (see baselines in §6) before reaching for an LLM-based classifier; only escalate complexity if the simple approach demonstrably underperforms on your golden set. Given you've already built a FAISS-based retrieval pipeline before, a lightweight embedding + nearest-centroid or embedding + small classifier (e.g. logistic regression on embeddings) is a natural, explainable middle ground between "keyword rules" and "LLM classifies everything."
- **[4] Retrieval:** this is exactly the retrieval step from a RAG pipeline — embed the brand's historical (customer message → resolution) pairs once, offline, into a vector index (FAISS is a reasonable, familiar choice here), and query it per new message. Cache the index as a build artifact so reproduction doesn't require re-embedding the whole corpus.
- **[5] Reply generation:** one LLM call, given the message + thread context + retrieved cases; the prompt should explicitly instruct the model to *only* use information present in the retrieved cases or general policy language, and to say so if it's uncertain — this is what makes stage [6] checkable.
- **[6] Grounding check:** cheapest viable version — ask the same or a smaller LLM "is every factual claim in this draft supported by the provided cases? list any unsupported claims" and turn that into a numeric/boolean grounding score. This does not need to be a separate fancy NLI model unless time permits.
- **[7]-[8] Confidence & routing:** keep this a simple, explainable rule (e.g. weighted sum or a small decision table), not a second ML model — a black-box confidence model is very hard to defend live, and the assignment rewards a *stated reason*, which favors interpretable rules.
- **[9] Logging:** this single piece of infrastructure pays for the entire evaluation harness, the failure analysis, and the "reproduce in 15 minutes" requirement — do not skip it or bolt it on late.

**What to explicitly *not* build** (call these out in the report as scoped-out, which itself signals judgment): a fine-tuned model, a full multi-agent framework with autonomous planning/looping, a database/service layer, a UI beyond a minimal demo script — none of these are asked for and all of them eat time better spent on evaluation.

---

## 6. Baselines

At least three systems must be compared on the *same* golden set with the *same* metrics.

1. **Trivial baseline** — no ML at all. E.g.: (a) intent = majority-class prediction (whatever intent is most frequent in the golden set's label distribution, computed *without peeking at test-time features*), and (b) reply = a single canned template per intent (e.g. "Thanks for reaching out — we're looking into this and will follow up"), and (c) escalation decision = always escalate (or always auto-handle) — pick whichever makes the contrast most informative. This establishes the floor.
2. **Simple ML/engineering baseline** — e.g.: (a) intent classification via TF-IDF + logistic regression (or embeddings + nearest-centroid) trained on a labelled subset disjoint from the golden set, (b) reply = nearest-neighbor retrieval only, i.e. return the single most similar historical brand reply verbatim with no generation/grounding step, (c) escalation = a simple fixed rule based on keyword flags (e.g. "refund", "lawyer", "cancel").
3. **Proposed system** — the full pipeline from §5.

**Fair comparison rules:**
- All three run on the identical golden set, never on different samples.
- All three are scored with the identical metrics/rubric from §8–9.
- Latency/cost should also be reported per system, since "proposed system is better but 40x the cost and 10x the latency" is itself an important, honest data point — don't hide it.
- State explicitly in the report by how much the proposed system beats each baseline, on which metrics, and where a baseline is *not* beaten (there will likely be at least one metric where the trivial or simple baseline is competitive — report that too; it's a feature of honest evaluation, not a flaw).

---

## 7. Golden Set Construction

**Size:** 150–250 examples per the assignment. Aim for the higher end (≈200–250) if time allows — larger golden sets give more stable, harder-to-game numbers.

**Sampling strategy:**
- Sample from the *held-out* portion of the brand's data — i.e., strictly disjoint from whatever pool you used for retrieval-corpus construction and for developing/tuning the classifier or prompts. This is the leakage-prevention step: if a message (or its near-duplicate) is in both your retrieval index and your golden set, your grounding/reply-quality scores will look artificially strong.
- Stratify the sample across: (a) each intent in your frozen taxonomy (don't let the golden set mirror the brand's natural — likely skewed — intent distribution exactly, or your rarer intents get 2–3 examples and produce noisy per-intent metrics), (b) message length buckets, (c) first-turn vs. follow-up-turn messages, (d) an intentionally oversampled slice of edge cases (see below).

**Edge cases to deliberately include (don't leave these to chance):**
- Ambiguous/multi-intent messages.
- Sarcasm or implicit dissatisfaction (no explicit complaint words but clearly unhappy).
- Messages that should clearly escalate (legal threats, safety, repeated unresolved issue).
- Very short/low-information messages ("this is broken").
- Messages referencing something outside the retrieval corpus's coverage (nothing similar historically) — tests whether the system correctly expresses low confidence rather than hallucinating a grounded-looking answer.

**Annotation guidelines:**
- Write down, before labelling starts, a short guideline doc: the taxonomy definitions (from §4), what counts as a "good" reply for this brand (tone, must/must-not include, e.g. never promise a refund amount that wasn't in a historical resolution), and what counts as a correct escalation call.
- Label three fields per example: gold intent, a gold "ideal reply direction" (can be a short reference note, not necessarily full prose, given time constraints — but note in the report if you did short-form references vs full gold replies), gold escalate/auto-handle decision + reason.

**Quality checks / handling disagreement:**
- Self-consistency check: re-label a random 10% of the set after a delay (e.g. a day later, without looking at your first pass) and measure intra-annotator agreement — report this number honestly, don't skip it because you're a single annotator.
- If a second person (classmate, friend) is available even briefly, have them independently label a 20–30 example subset and compute agreement (Cohen's kappa for intent; simple % agreement for escalate/no-escalate) — this materially strengthens your "golden set is trustworthy" claim, and doubles as your judge-validation data source in §9 if scoped well.
- Where you personally disagree with your first-pass label on review, log the correction and the reason — a small "annotation notes" appendix is cheap to produce and reads very well to an evaluator.

**Representativeness, not cherry-picking:**
- Fix the stratified-sampling code and seed *first*, then label whatever comes out — do not swap out examples because they're "hard to label" or make your system look bad. If you must exclude something (e.g. a non-English tweet that slipped through), log the exclusion rule and apply it uniformly, not case-by-case.
- Report the final realized distribution (counts per intent, per edge-case category) in the report/README so the evaluator can see it wasn't hand-picked to flatter the system.

---

## 8. Evaluation Framework

Design this *before* the agent is finished, so the agent isn't built to game a metric you invent after seeing its outputs.

| Dimension | Metric(s) | Role |
|---|---|---|
| Intent classification | Accuracy, macro-F1 (macro, not just micro/accuracy, because intents are likely imbalanced) | Headline for this sub-task |
| Reply quality | LLM-judge rubric score (see §9), broken into sub-scores (helpfulness, tone-fit, conciseness) | Headline for this sub-task |
| Grounding / factual correctness | % of replies with zero unsupported factual claims (from stage [6]'s check), spot-validated by hand on a subsample | Headline — this is the trust metric |
| Relevance | LLM-judge sub-score: does the reply actually address what the customer asked? | Supporting |
| Escalation decisions | Precision/recall against gold escalate labels, with attention to false negatives (should-have-escalated-but-didn't) as the costlier error class | Headline for this sub-task |
| Auto-handled cases | Of the cases the system chose to auto-handle, what fraction would a human judge as actually resolvable without escalation? | Headline — this is the "can we trust autonomy" metric |
| Overall reliability | A single combined score is optional and, if included, must be explicitly labelled as a weighted composite with the weights shown — never present a composite as if it were a natively meaningful unit | Supporting only, never headline alone |

**Headline vs. supporting, explicitly:** pick 2–3 headline numbers max (suggest: reply-quality judge score, grounding rate, and escalation recall on the "should-escalate" edge cases) and demote everything else to supporting detail in an appendix table. A report with 15 undifferentiated metrics reads as if you don't know which one matters — that itself is a negative signal.

**Segment your metrics, don't just report the mean:** break every headline metric down by intent and by the edge-case categories from §7. A system that's 92% good on easy cases and 40% good on edge cases has a very different headline story than a flat 85% — and this segmentation is exactly the input to §11 (misleading headline number).

---

## 9. LLM-as-Judge Framework

**Rubric (define per-dimension, 1–5 or 0–2 scales, evaluator's choice, but be consistent):**
- *Grounding:* does the reply only make claims supported by the retrieved historical cases / general non-fabricated policy language?
- *Relevance:* does it address the actual customer message?
- *Helpfulness:* does it move the customer meaningfully toward resolution (vs. generic platitudes)?
- *Tone fit:* does it match the brand's observed tone in its historical replies?
- *Conciseness/appropriateness for the channel:* Twitter replies are short — penalize bloated responses.

**Judge prompt structure:**
- Provide the judge: the customer message + thread context, the retrieved cases actually used, the candidate reply, and the rubric with the scale spelled out per dimension.
- Ask for a structured output (JSON) with a score per dimension plus a one-sentence justification per score — the justification is what makes disagreements diagnosable later, don't skip it to save tokens.
- Judge one reply at a time, blind to which system (trivial/simple/proposed) produced it, and blind to any prior scores — randomize/shuffle order across systems to avoid position or system-identity bias.

**Preventing judge bias:**
- Never show the judge which system generated a reply.
- Watch for length bias (LLM judges are documented to favor longer answers) — explicitly instruct the rubric to penalize unnecessary length, and check post-hoc whether judge scores correlate suspiciously with reply length; report that correlation number.
- Use a fixed judge prompt/temperature (low, e.g. 0) across the whole run so you're not comparing a judge that behaved differently between systems.
- Consider a small self-consistency check: re-judge a subsample twice and check score stability before trusting a single pass.

**Validating the judge against humans:**
- Take a subsample (aim for at least 30–50, more if time allows — ties back to §7's second-annotator effort) where you (or a second annotator) also score the same replies on the same rubric, blind to the judge's scores.
- Compute agreement: exact-match rate, and/or a correlation coefficient (Spearman is reasonable for ordinal 1–5 scores), and/or Cohen's kappa if you bucket scores into pass/fail bands.
- **Report this number honestly, wherever it lands.** Moderate agreement (say, 0.4–0.6 correlation) reported honestly with a discussion of *where* the judge and human disagreed (e.g. judge is more lenient on tone than a human) is a stronger submission than a suspiciously perfect agreement number with no discussion — evaluators are specifically looking for whether you understand judge limitations, not whether your judge is flawless.

---

## 10. Failure Analysis

Find the top 5 failure modes via both automated and manual passes:

**Automated pass:** sort golden-set examples by lowest judge/grounding scores and by escalation-decision mismatches; cluster the low-scoring examples by intent and by which pipeline stage's confidence was lowest — this usually surfaces 3–4 clear clusters fast.

**Manual pass:** read through *every* golden-set example your system got meaningfully wrong (not just a sample of the failures — at this scale, ~150–250 examples, you can and should read all of the low scorers) to catch failure types the automated sort misses (e.g. a whole intent category is systematically mis-defined, not just individually mis-classified).

**For each of the 5 failure modes, document:**
- **Real example** — the actual message + system output (verbatim, from your own data — never invented).
- **Expected behavior** — what a correct system should have done.
- **Actual behavior** — what happened instead.
- **Root cause hypothesis** — e.g. "retrieval returned a case from a different intent because embeddings weight surface wording over intent," stated as a hypothesis, not a proven fact, unless you've actually tested it.
- **Potential fix** — concrete, e.g. "filter retrieval candidates by predicted intent before ranking by similarity."
- **Worth implementing? (yes/no + why)** — an honest cost/benefit call; "no, this affects <2% of cases and the fix risks breaking the more common case" is a legitimate and good answer.

---

## 11. Misleading Headline Number

This section is mandatory and evaluators will specifically check for intellectual honesty here — do not write a generic disclaimer.

Structure it as a real critique of your *actual* strongest metric, e.g.:
- **Sampling bias:** your golden set, however carefully stratified, is still drawn from historical resolved threads — it systematically excludes cases the brand never resolved on Twitter at all (escalated to phone/email, or simply abandoned), so your "reliability" number says nothing about that unmeasured slice.
- **Judge leniency/bias:** if your judge-human agreement (§9) wasn't perfect, your headline judge score inherits that judge's specific blind spots (e.g. tone leniency) at whatever rate you measured.
- **Distributional mismatch:** if one intent dominates your golden set's realized distribution even after stratification effort, your headline is disproportionately a measure of performance on that intent.
- **Retrieval corpus overlap effects:** if the historical resolution used for grounding happens to closely match the golden example (because both come from a similar time period/campaign), grounding may look better than it would on a genuinely novel future message.
- **Static evaluation vs. live conversation:** your escalation-decision metric evaluates a single message in isolation; a live agent would face multi-turn back-and-forth where errors compound — your number doesn't capture that.

Pick whichever 2–3 of these (or others you find specific to your actual results) genuinely apply, back each with a number or observation from your own run, and be concrete about the direction and rough size of the bias (overstated by how much, in which direction) rather than a vague "there are limitations."

---

## 12. Decision Log

Record 10–15 decisions as you go (don't reconstruct them from memory at the end — you'll lose the real alternatives-considered detail). Each entry: **Decision / Alternatives considered / Reason / Trade-off / Evidence.** Candidate decisions likely to arise from this plan (fill in your actual reasoning once made):

1. Which brand was selected, and why (over the runner-up).
2. Number and definition of intents in the frozen taxonomy.
3. How ambiguous/multi-intent messages are labelled.
4. Thread context window size (how many prior turns included).
5. Classifier approach for intent (rule-based vs. embeddings+classifier vs. LLM-zero-shot) and why.
6. Retrieval method and embedding model choice.
7. Number of retrieved cases (k) fed into reply generation.
8. Reply-generation model choice (cost/latency/quality trade-off, especially given API budget constraints).
9. How the grounding check is implemented (LLM self-check vs. a separate lighter model).
10. Confidence/escalation thresholding scheme and why those thresholds.
11. Golden set size within the 150–250 range and stratification scheme.
12. LLM judge model choice and why (cost vs. quality vs. availability).
13. How judge-human agreement was measured and what threshold was considered "good enough" to trust the judge.
14. What was deliberately scoped out (fine-tuning, multi-agent orchestration, a UI/service layer) and why.
15. Any deduplication/cleaning rule that measurably changed the data (with the before/after row counts).

---

## 13. Repository Structure

```
hiver-support-agent/
├── README.md                  # setup, exact commands, <15-min repro instructions
├── report.md                  # or report.pdf — the max-6-page report
├── decision_log.md
├── taxonomy.md                # frozen intent definitions + version
├── requirements.txt
├── config.yaml                # thresholds, model names, k, window size — single source of truth
├── data/
│   ├── raw/                   # (gitignored) pointer/README only — don't commit the full Kaggle dump
│   ├── processed/             # cleaned/sampled subsample used for the demo (small, committed)
│   └── golden_set/            # the frozen 150-250 labelled examples (committed, this is precious)
├── artifacts/                 # precomputed: embeddings index, trained baseline classifier, cached LLM outputs
├── src/
│   ├── preprocessing.py
│   ├── thread_reconstruction.py
│   ├── intent_classifier.py
│   ├── retrieval.py
│   ├── reply_generation.py
│   ├── grounding_check.py
│   ├── routing.py             # confidence + auto-handle/escalate decision
│   └── pipeline.py            # wires stages 1-9 together, writes structured logs
├── eval/
│   ├── baselines.py
│   ├── metrics.py
│   ├── llm_judge.py
│   ├── judge_validation.py    # judge-vs-human agreement computation
│   └── run_eval.py            # single entry point producing the report's numbers/tables
├── notebooks/                 # exploration only — nothing load-bearing lives only in a notebook
├── logs/                      # JSONL run logs from pipeline.py, used for failure analysis
└── scripts/
    └── reproduce.sh           # the literal script the README tells the evaluator to run
```

Keep `src/` free of notebook-only exploration code; anything the pipeline depends on must be a tested, importable module.

---

## 14. 15-Minute Reproduction Plan

**Precompute and commit (small enough to check in, or fetch via a script):**
- The cleaned/sampled subsample used for the demo (not the full 3M-tweet dump).
- The FAISS/embedding index over the retrieval corpus.
- The trained simple baseline classifier (a small pickled/serialized model).
- The frozen golden set with gold labels.
- Cached LLM outputs (draft replies + judge scores) for the exact evaluation run reported in the report, keyed by example ID and prompt version, so re-running `run_eval.py` with `--use-cache` reproduces the identical headline numbers without needing live API keys or spending money.

**What should run locally in the 15 minutes:**
- Loading the precomputed artifacts.
- Running the trivial and simple baselines fresh (cheap, deterministic, no API calls) to reproduce their numbers.
- Recomputing the aggregate metrics/tables from the cached proposed-system outputs.
- Optionally, a `--live` flag that re-calls the LLM APIs for a handful of examples if the evaluator wants to sanity-check the caching wasn't hiding anything — but this shouldn't be required for the headline numbers.

**README commands (exact, e.g.):**
```
pip install -r requirements.txt
python scripts/reproduce.sh          # rebuilds all headline tables from cached artifacts, <15 min
python eval/run_eval.py --use-cache  # equivalent, more granular
python eval/run_eval.py --live --n 10  # optional: re-run 10 examples live against the real API
```

**Determinism:**
- Fix all random seeds (sampling, any stochastic clustering, classifier training).
- Pin package versions in `requirements.txt`.
- Set LLM temperature to 0 (or as low as the API allows) for anything the report's numbers depend on, and note the exact model version string used (model versions drift over time — record it).
- Version the taxonomy and prompts (e.g. `prompt_v1.txt`) and log which version produced each cached result.

---

## 15. Day-by-Day Implementation Roadmap

*(Scale the number of days to your actual deadline — this is a sequence, not a fixed calendar; compress or stretch phases proportionally, but do not skip phase order.)*

**Phase 1 — Data Inspection & Brand Selection**
- Objective: pick a brand with evidence, not vibes.
- Tasks: load dataset, run structural inspection (§3 step 1), score candidate brands (§3 step 2), spot-read threads.
- Output: brand-comparison table + one-paragraph justification, committed to the repo.
- Checkpoint: can you name, with a number, why this brand beats the runner-up?
- Definition of done: brand locked, comparison table in decision log.

**Phase 2 — Intent Taxonomy**
- Objective: a frozen, evidence-based intent set.
- Tasks: embed + cluster, manual synthesis, coverage check on a fresh 100-sample, freeze.
- Output: `taxonomy.md` with definitions + examples, versioned.
- Checkpoint: can two different messages be confidently assigned without you having to think hard?
- Definition of done: taxonomy frozen, coverage/"other" rate documented.

**Phase 3 — Golden Set Construction**
- Objective: 150–250 trustworthy labelled examples, held out from everything else.
- Tasks: stratified sampling script, annotation pass, self-consistency re-check, (if possible) second-annotator pass.
- Output: `data/golden_set/`, annotation guideline doc, agreement numbers.
- Checkpoint: agreement numbers computed and acceptable-or-honestly-reported.
- Definition of done: golden set frozen and *not touched again* except to fix a labelling bug you can justify in writing.

**Phase 4 — Pipeline Build (stages 1–9)**
- Objective: an end-to-end runnable agent, however simple.
- Tasks: build each stage as an independent, testable module; wire into `pipeline.py`; get structured logging working from day one of this phase, not bolted on later.
- Output: pipeline runs on the working sample and produces logged, inspectable outputs.
- Checkpoint: pick 5 random outputs and manually sanity-check them before moving on.
- Definition of done: pipeline runs deterministically on the full working sample without crashing, with logs.

**Phase 5 — Baselines**
- Objective: trivial + simple baselines implemented, run on the same golden set.
- Tasks: implement both, run once, sanity-check outputs are actually "trivial"/"simple" as intended (not accidentally strong or accidentally broken).
- Output: baseline outputs cached alongside proposed-system outputs.
- Definition of done: all three systems' outputs exist for every golden-set example.

**Phase 6 — Evaluation Harness + LLM Judge**
- Objective: metrics + judge implemented and validated.
- Tasks: implement `metrics.py`, implement `llm_judge.py` with the rubric from §9, run judge on all three systems' outputs (blind, shuffled), run judge-validation against a human-labelled subsample.
- Output: full metrics table (headline + supporting, segmented by intent/edge-case), judge-human agreement number.
- Checkpoint: do the numbers make qualitative sense against your Phase 4 manual sanity check? If the proposed system scores *worse* than the trivial baseline on something, investigate before writing it off as a bug or accepting it as a real finding.
- Definition of done: all metrics tables committed, cached.

**Phase 7 — Failure Analysis, Misleading-Number Section, Decision Log, Report**
- Objective: turn the numbers into an honest, well-argued report.
- Tasks: sort/cluster failures, write up top 5, write the misleading-headline-number critique, finalize the decision log from your running notes, write the ≤6-page report.
- Output: `report.md`.
- Definition of done: every claim in the report has a corresponding number/artifact in the repo.

**Phase 8 — Reproducibility Polish & README**
- Objective: a stranger can reproduce your headline numbers in <15 minutes.
- Tasks: write `scripts/reproduce.sh`, time it yourself end-to-end on a clean checkout/environment, fix anything that breaks, pin versions.
- Definition of done: you personally timed a clean-environment reproduction and it was under 15 minutes.

**Phase 9 — Submission**
- Tasks: final repo cleanup (remove dead notebooks/secrets), confirm repo access settings, submit via the Notion form with repo link + report, per the stated submission instructions.

---

## 16. Final Submission Checklist

- [ ] Repo is public or access is explicitly granted to the evaluators.
- [ ] README reproduces headline numbers in <15 minutes on a clean environment — actually timed, not estimated.
- [ ] No API keys or secrets committed.
- [ ] Golden set (150–250 examples) is present, frozen, and clearly disjoint from retrieval-corpus/training data.
- [ ] Evaluation harness runs from cached artifacts without requiring live API spend.
- [ ] LLM-judge rubric is documented, and judge-human agreement is reported with an actual number.
- [ ] Report is ≤6 pages (or README section) and covers all 5 required sub-sections, including the mandatory misleading-headline-number section.
- [ ] Decision log has 10–15 real entries with alternatives and trade-offs, not generic statements.
- [ ] Failure analysis has 5 real, documented failure modes.
- [ ] Anything borrowed (code snippets, prompts, tutorials) is cited.
- [ ] You can explain and modify every file in `src/` live, without notes.

---

## 17. Interview Preparation

Be ready to answer, live and without your slides:

- Why this brand and not the top-3 alternatives — walk through the actual comparison numbers.
- Why this taxonomy and not a coarser/finer one — what would break if you merged two intents?
- Walk through one real example end-to-end through all 9 pipeline stages, including a case where it fails.
- Why this retrieval method and embedding choice, and what would change if the corpus were 100x bigger?
- Defend the confidence/escalation thresholds — what would happen at a stricter or looser threshold, and did you check?
- Explain the judge-human agreement number and what specifically it does and doesn't tell you.
- Pick your weakest headline number and explain, unprompted, why it might be misleading (this is effectively §11 rehearsed live).
- Be ready to modify a piece of your own code on the spot — e.g. "change k in retrieval from 3 to 5, what do you expect to happen, and why?"
- Know the cost/latency profile of your system and what you'd change to make it 10x cheaper.

---

## 18. Biggest Risks and How to Avoid Them

| Risk | Mitigation |
|---|---|
| Spending too long on architecture, running out of time for evaluation | Follow the time-budget in §1; build the golden set before finishing the pipeline if forced to choose |
| Golden set contaminated by retrieval-corpus overlap | Enforce a strict train/eval split at the sampling stage (§7), verify with an explicit dedup/overlap check before freezing |
| LLM judge silently favors your proposed system (system-identity or length bias) | Blind, shuffle, and explicitly test for length correlation (§9) |
| "15-minute reproduction" fails on a clean machine because of an untracked local file/env variable | Actually test reproduction from a fresh clone/venv before submitting, not just on your dev machine |
| Report makes claims not backed by a number in the repo | Do a final pass matching every sentence in the report to an artifact/log |
| Taxonomy drifts mid-project (relabelling under a changed definition) | Freeze and version the taxonomy file explicitly (§4); treat any post-freeze change as a logged decision, not a silent edit |
| Over-scoped agent (multi-agent frameworks, unnecessary services) eats time and is hard to defend live | Explicitly scope these out in the report (§5) as a stated, deliberate choice |
| API cost/rate limits derail development close to the deadline | Cache aggressively from day one (§14), prefer a cheaper/local model for iteration and reserve the strongest model for the final judged/reported run |
