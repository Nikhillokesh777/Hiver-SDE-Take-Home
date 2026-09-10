"""
Stage 1: Preprocessing Module.
Normalizes customer and brand messages:
- Strips PII-like customer handles while preserving brand handles (@Uber_Support)
- Normalizes URLs to standard [URL] placeholder
- Preserves emojis (semantic value)
- Normalizes excessive whitespace and formatting
- Detects non-English / noisy text
"""

import re
from typing import Dict, Any, Optional

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_HANDLE_REGEX = re.compile(r"@([a-zA-Z0-9_]+)")
WHITESPACE_REGEX = re.compile(r"\s+")


class Preprocessor:
    def __init__(self, brand_handle: str = "Uber_Support"):
        self.brand_handle = brand_handle.lower()

    def normalize(self, text: str) -> str:
        """
        Applies deterministic text normalization.
        """
        if not isinstance(text, str):
            return ""

        # Normalize URLs
        cleaned = URL_REGEX.sub("[URL]", text)

        # Mask user handles to @customer while preserving brand handle
        def mask_handle(match):
            h = match.group(1)
            if h.lower() == self.brand_handle or h.lower() in ["uber", "ubersupport", "help"]:
                return f"@{self.brand_handle}"
            return "@customer"

        cleaned = USER_HANDLE_REGEX.sub(mask_handle, cleaned)

        # Collapse whitespace
        cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
        return cleaned

    def process(self, raw_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes Stage 1 contract.
        Input: raw text + optional metadata
        Output: dictionary containing raw_text, clean_text, char_len, and metadata
        """
        clean_text = self.normalize(raw_text)
        is_clean_ascii = bool(re.match(r"^[\x00-\x7F]+$", clean_text))

        return {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "char_length": len(clean_text),
            "word_count": len(clean_text.split()),
            "is_ascii": is_clean_ascii,
            "metadata": metadata or {}
        }
