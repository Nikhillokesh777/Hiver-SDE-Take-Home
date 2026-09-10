"""
Interactive Live Demo CLI for Hiver Autonomous Support Agent.
Allows testing standard queries, edge cases, safety escalations, and custom input.
"""

import sys
import os
import logging

# Suppress HuggingFace / Transformers progress bars and warnings for clean console formatting
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgentPipeline

SAMPLE_QUERIES = [
    (
        "Standard Auto-Handle (Cancellation Fee)",
        "My driver cancelled my ride and charged me a $5 cancellation fee, can I get a refund?"
    ),
    (
        "Critical Safety Escalation (100% Safety Recall)",
        "The driver was driving recklessly on the highway and threatened to assault me. I need urgent help!"
    ),
    (
        "Colloquial Trigger Edge Case (Report.md Failure Mode 5)",
        "I requested this ride on accident because my phone was lagging, please refund the fee."
    ),
    (
        "Lost Property Inquiry",
        "I left my black backpack and laptop in the back seat of the car after my ride to JFK airport."
    ),
    (
        "Driver Conduct / Unprofessional Service",
        "The driver was extremely rude, refused to turn on the AC, and made inappropriate comments."
    ),
]


def display_result(res):
    print("\n" + "=" * 70)
    print(f" INPUT QUERY:        {res.input_text}")
    print("-" * 70)
    print(f" PREDICTED INTENT:   {res.predicted_intent} (Confidence: {res.intent_confidence:.4f})")
    print(f" ROUTING DECISION:   {res.routing_decision}")
    print(f" ROUTING REASON:     {res.routing_reason}")
    print(f" GROUNDING CHECK:    {'PASSED' if res.grounding_pass else 'FAILED'} (Score: {res.grounding_score:.2f})")
    print(f" LATENCY:            {res.latency_ms:.2f} ms")
    print("-" * 70)
    print(f" GENERATED REPLY:")
    print(f" \"{res.generated_reply}\"")
    print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("  HIVERA SUPPORT AGENT - LIVE INTERACTIVE DEMO")
    print("=" * 70)
    print("Initializing pipeline (loading FAISS index and embeddings)...")
    pipeline = SupportAgentPipeline()
    print("Pipeline ready!\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Processing query: {query}")
        result = pipeline.run(query)
        display_result(result)
        return

    while True:
        print("\nSelect a sample scenario to test, or enter your own:")
        for idx, (title, text) in enumerate(SAMPLE_QUERIES, 1):
            print(f"  [{idx}] {title}")
            print(f"      \"{text}\"")
        print("  [6] Enter your own custom inquiry")
        print("  [0] Exit")

        choice = input("\nEnter choice (0-6): ").strip()

        if choice == "0":
            print("Exiting demo. Goodbye!")
            break
        elif choice in {"1", "2", "3", "4", "5"}:
            idx = int(choice) - 1
            title, query = SAMPLE_QUERIES[idx]
            print(f"\n---> Running Scenario {choice}: {title}")
            result = pipeline.run(query)
            display_result(result)
        elif choice == "6":
            query = input("\nEnter your support tweet / inquiry: ").strip()
            if not query:
                print("Empty input. Try again.")
                continue
            print(f"\n---> Processing Custom Inquiry...")
            result = pipeline.run(query)
            display_result(result)
        else:
            print("Invalid choice, please enter 0 through 6.")


if __name__ == "__main__":
    main()
