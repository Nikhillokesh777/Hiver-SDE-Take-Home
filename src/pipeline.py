"""
Stage 9: End-to-End Support Agent Pipeline & Structured Observability Logger.
Wires Stages 1 through 8 into a single, cohesive, observable execution pipeline:
Input Message -> Preprocessing -> Thread Reconstruction -> Intent Classification ->
Historical Case Retrieval -> Reply Generation -> Grounding Check -> Confidence & Routing ->
Structured JSONL Telemetry Logging.
Follows Sections 5 and 18 of hiver_execution_plan.md.
"""

import os
import sys
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.preprocessing import Preprocessor
from src.thread_reconstruction import ThreadReconstructor, ThreadContext
from src.intent_classifier import IntentClassifier, IntentClassificationResult
from src.retrieval import HistoricalCaseRetriever, RetrievalResult
from src.reply_generation import ReplyGenerator, GeneratedReply
from src.grounding_check import GroundingChecker, GroundingResult
from src.routing import RoutingEngine, RoutingDecision

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


class PipelineExecutionOutput(BaseModel):
    execution_id: str
    example_id: Optional[str] = None
    input_text: str
    clean_text: str
    turn_position: str = "first_turn"
    predicted_intent: str
    intent_confidence: float
    retrieved_case_ids: List[int] = Field(default_factory=list)
    max_retrieval_similarity: float
    generated_reply: str
    grounding_pass: bool
    grounding_score: float
    routing_decision: str
    routing_reason: str
    composite_confidence: float
    latency_ms: float
    error: Optional[str] = None


class SupportAgentPipeline:
    def __init__(
        self,
        brand_handle: str = "Uber_Support",
        max_context_turns: int = 4,
        retrieval_top_k: int = 3,
        auto_handle_threshold: float = 0.70,
        logs_dir: str = DEFAULT_LOGS_DIR,
        cache: Optional[Dict[str, Any]] = None
    ):
        self.brand_handle = brand_handle
        self.logs_dir = logs_dir
        self.cache = cache
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize modular stages
        self.preprocessor = Preprocessor(brand_handle=brand_handle)
        self.thread_reconstructor = ThreadReconstructor(max_context_turns=max_context_turns)
        self.intent_classifier = IntentClassifier()
        self.retriever = HistoricalCaseRetriever(top_k=retrieval_top_k)
        self.reply_generator = ReplyGenerator()
        self.grounding_checker = GroundingChecker()
        self.routing_engine = RoutingEngine(auto_handle_threshold=auto_handle_threshold)

        # Active run log file
        run_date = time.strftime("%Y%m%d")
        self.log_file_path = os.path.join(self.logs_dir, f"pipeline_run_{run_date}.jsonl")

    def run(
        self,
        raw_text: str,
        example_id: Optional[str] = None,
        prior_turns: Optional[List[Dict[str, Any]]] = None,
        edge_category: str = "standard"
    ) -> PipelineExecutionOutput:
        """
        Executes the full 9-stage support pipeline end-to-end.
        """
        start_t = time.perf_counter()
        exec_id = f"exec_{uuid.uuid4().hex[:8]}"

        try:
            # Stage 1: Preprocessing
            prep_out = self.preprocessor.process(raw_text)
            clean_text = prep_out["clean_text"]

            # Stage 2: Thread Reconstruction
            thread: ThreadContext = self.thread_reconstructor.reconstruct(
                current_text=clean_text,
                prior_turns=prior_turns
            )

            # Stage 3: Intent Classification
            intent_res: IntentClassificationResult = self.intent_classifier.classify(clean_text)

            # Stage 4: Historical Case Retrieval
            retrieval_res: RetrievalResult = self.retriever.retrieve(
                query=clean_text,
                top_k=3
            )
            case_ids = [c.case_id for c in retrieval_res.retrieved_cases]

            # Stage 5: Grounded Reply Generation
            reply_res: GeneratedReply = self.reply_generator.generate(
                customer_message=clean_text,
                thread_transcript=thread.formatted_transcript,
                retrieved_evidence=retrieval_res.evidence_text,
                retrieved_case_ids=case_ids,
                cache=self.cache
            )

            # Stage 6: Grounding Verification Check
            grounding_res: GroundingResult = self.grounding_checker.check(
                reply_text=reply_res.reply_text,
                evidence_text=retrieval_res.evidence_text,
                evidence_ids=case_ids,
                cache=self.cache
            )

            # Stage 7 & 8: Confidence Estimation and Routing Engine
            routing_res: RoutingDecision = self.routing_engine.route(
                intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
                retrieval_similarity=retrieval_res.max_similarity,
                grounding_pass=grounding_res.grounding_pass,
                grounding_score=grounding_res.grounding_score,
                edge_category=edge_category,
                raw_text=raw_text
            )

            latency = round((time.perf_counter() - start_t) * 1000, 2)

            output = PipelineExecutionOutput(
                execution_id=exec_id,
                example_id=example_id,
                input_text=raw_text,
                clean_text=clean_text,
                turn_position="follow_up" if prior_turns else "first_turn",
                predicted_intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
                retrieved_case_ids=case_ids,
                max_retrieval_similarity=retrieval_res.max_similarity,
                generated_reply=reply_res.reply_text,
                grounding_pass=grounding_res.grounding_pass,
                grounding_score=grounding_res.grounding_score,
                routing_decision=routing_res.decision,
                routing_reason=routing_res.reason,
                composite_confidence=routing_res.composite_confidence,
                latency_ms=latency,
                error=None
            )

        except Exception as e:
            latency = round((time.perf_counter() - start_t) * 1000, 2)
            output = PipelineExecutionOutput(
                execution_id=exec_id,
                example_id=example_id,
                input_text=raw_text,
                clean_text=raw_text,
                predicted_intent="Other_Or_Unclear",
                intent_confidence=0.0,
                retrieved_case_ids=[],
                max_retrieval_similarity=0.0,
                generated_reply="We apologize for the inconvenience. Please send us a DM so our team can assist.",
                grounding_pass=False,
                grounding_score=0.0,
                routing_decision="ESCALATE",
                routing_reason=f"Pipeline error encountered: {str(e)}",
                composite_confidence=0.0,
                latency_ms=latency,
                error=str(e)
            )

        # Stage 9: Structured JSONL Logging
        self._log_execution(output)
        return output

    def _log_execution(self, output: PipelineExecutionOutput):
        """Appends structured execution telemetry to JSONL log."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(output.model_dump_json() + "\n")
        except Exception as e:
            print(f"[Warning] Failed to write log: {e}", file=sys.stderr)
