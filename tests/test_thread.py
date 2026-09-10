"""
Tests for Stage 2: Thread Reconstruction & Context Window Bounding.
Verifies multi-turn conversation handling, turn-limit truncation, and transcript formatting.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.thread_reconstruction import ThreadReconstructor, ThreadContext


def test_thread_single_turn_first_message():
    """Verify first-turn customer message without prior turns produces valid single-turn context."""
    tr = ThreadReconstructor(max_context_turns=4)
    ctx = tr.reconstruct(current_text="Where is my ride?", prior_turns=None)
    
    assert isinstance(ctx, ThreadContext)
    assert ctx.total_turns == 1
    assert len(ctx.prior_turns) == 0
    assert "Customer (Current): Where is my ride?" in ctx.formatted_transcript


def test_thread_multi_turn_ordering():
    """Verify conversational turns are preserved in chronological order."""
    tr = ThreadReconstructor(max_context_turns=4)
    priors = [
        {"role": "customer", "text": "My driver hasn't arrived."},
        {"role": "agent", "text": "Can you check the live GPS map in your app?"},
        {"role": "customer", "text": "The map shows him 10 miles away moving in reverse."},
    ]
    ctx = tr.reconstruct(current_text="Now he cancelled the ride!", prior_turns=priors)
    
    assert ctx.total_turns == 4
    assert "Customer: My driver hasn't arrived." in ctx.formatted_transcript
    assert "Support Agent: Can you check the live GPS map in your app?" in ctx.formatted_transcript
    assert "Customer (Current): Now he cancelled the ride!" in ctx.formatted_transcript


def test_thread_bounds_maximum_context_turns():
    """Verify context history is strictly bounded by max_context_turns (preventing context bloat)."""
    tr = ThreadReconstructor(max_context_turns=2)
    priors = [
        {"role": "customer", "text": "Turn 1 (oldest)"},
        {"role": "agent", "text": "Turn 2"},
        {"role": "customer", "text": "Turn 3"},
        {"role": "agent", "text": "Turn 4 (most recent prior)"},
    ]
    ctx = tr.reconstruct(current_text="Turn 5 (current)", prior_turns=priors)
    
    # max_context_turns=2 should keep only Turn 3 and Turn 4
    assert len(ctx.prior_turns) == 2
    assert ctx.prior_turns[0].text == "Turn 3"
    assert ctx.prior_turns[1].text == "Turn 4 (most recent prior)"
    assert "Turn 1 (oldest)" not in ctx.formatted_transcript
    assert "Turn 5 (current)" in ctx.formatted_transcript
