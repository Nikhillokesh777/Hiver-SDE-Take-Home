"""
Tests for Stage 9: End-to-End Pipeline Execution & Telemetry Logging.
Verifies full integration through Stages 1-8, output schema, structured JSONL telemetry, and error resilience.
"""

import os
import sys
import json
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.pipeline import SupportAgentPipeline, PipelineExecutionOutput


@pytest.fixture(scope="module")
def pipeline():
    return SupportAgentPipeline()


def test_pipeline_e2e_successful_execution(pipeline):
    """Verify complete end-to-end execution on a standard support inquiry."""
    query = "My driver cancelled my trip and charged me a $5 cancellation fee, can I get a refund?"
    result = pipeline.run(query)

    assert isinstance(result, PipelineExecutionOutput)
    assert result.execution_id.startswith("exec_")
    assert result.input_text == query
    assert len(result.clean_text) > 0
    assert result.predicted_intent != ""
    assert result.routing_decision in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(result.grounding_pass, bool)
    assert len(result.generated_reply) > 0
    assert result.latency_ms > 0.0


def test_pipeline_telemetry_logging(pipeline):
    """Verify pipeline writes structured telemetry JSONL logs with execution metadata."""
    query = "Test telemetry log entry for support pipeline verification."
    result = pipeline.run(query)

    log_file = pipeline.log_file_path
    assert os.path.exists(log_file), f"Telemetry log file not found at {log_file}"

    # Read last log entry
    last_entry = None
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_entry = json.loads(line)

    assert last_entry is not None
    assert last_entry["execution_id"] == result.execution_id
    assert last_entry["predicted_intent"] == result.predicted_intent
    assert last_entry["routing_decision"] == result.routing_decision


def test_pipeline_empty_input_resilience(pipeline):
    """Verify empty or whitespace query is handled gracefully without crashing the pipeline."""
    result = pipeline.run("    ")
    assert isinstance(result, PipelineExecutionOutput)
    assert result.routing_decision == "ESCALATE"
    assert result.predicted_intent == "Other_Or_Unclear"
