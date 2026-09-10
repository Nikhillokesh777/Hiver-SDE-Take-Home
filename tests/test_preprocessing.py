"""
Tests for Stage 1: Preprocessing and Text Normalization.
Verifies PII masking, URL normalization, emoji preservation, and boundary whitespace stripping.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import Preprocessor


def test_preprocessor_masks_customer_handles():
    """Verify customer handles (@115873, @john_doe) are masked to @customer while preserving brand handle."""
    p = Preprocessor(brand_handle="Uber_Support")
    raw = "@115873 @jane_doe my trip was charged twice! Please help @Uber_Support"
    out = p.process(raw)
    
    assert "@customer" in out["clean_text"]
    assert "@115873" not in out["clean_text"]
    assert "@jane_doe" not in out["clean_text"]
    assert "@uber_support" in out["clean_text"]


def test_preprocessor_replaces_urls():
    """Verify URLs are normalized to [URL] tokens to prevent downstream hallucination of fake links."""
    p = Preprocessor(brand_handle="Uber_Support")
    raw = "Here is my trip receipt https://trip.uber.com/receipt/12345 and help page http://t.co/help"
    out = p.process(raw)
    
    assert "https://trip.uber.com" not in out["clean_text"]
    assert "http://t.co/help" not in out["clean_text"]
    assert "[URL]" in out["clean_text"]


def test_preprocessor_preserves_emojis():
    """Verify emotional and safety emojis are strictly preserved for sentiment & escalation signals."""
    p = Preprocessor(brand_handle="Uber_Support")
    raw = "Driver was aggressive 😡 and speeding 🔥 please help 🙏"
    out = p.process(raw)
    
    assert "😡" in out["clean_text"]
    assert "🔥" in out["clean_text"]
    assert "🙏" in out["clean_text"]


def test_preprocessor_strips_redundant_whitespace():
    """Verify multiple spaces, tabs, and newlines are collapsed to a single space."""
    p = Preprocessor(brand_handle="Uber_Support")
    raw = "   Where   is    my\n\n\nrefund???   "
    out = p.process(raw)
    
    assert out["clean_text"] == "Where is my refund???"
    assert out["char_length"] == len("Where is my refund???")


def test_preprocessor_empty_input_handling():
    """Verify empty or whitespace-only input returns safe empty representation."""
    p = Preprocessor(brand_handle="Uber_Support")
    out = p.process("   \n\t  ")
    assert out["clean_text"] == ""
    assert out["char_length"] == 0
