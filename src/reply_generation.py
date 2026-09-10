"""
Stage 5: Reply Generation Module.
Generates an empathetic, concise, grounded response based strictly on:
- Customer's current message and conversation thread context
- Retrieved historical support resolution cases
- Explicit anti-hallucination instructions

Supports Gemini and OpenAI with transparent prompt versioning and artifact caching.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

PROMPT_VERSION = "v1.0"

SYSTEM_INSTRUCTION = """You are an official customer support representative for @Uber_Support on Twitter.
Your goal is to write a helpful, empathetic, concise, and professional Twitter reply (under 280 characters).

STRICT GROUNDING RULES:
1. ONLY provide instructions, links, or solutions that are directly supported by the PROVIDED HISTORICAL CASES.
2. NEVER invent refund amounts, guarantees, policy exceptions, or fake contact numbers.
3. If the retrieved historical cases suggest directing the customer to in-app Help or asking for details via Direct Message, advise them accordingly.
4. If you do not have enough specific evidence to resolve the issue directly, apologize politely and instruct them to send a Direct Message with their trip details.
5. Keep the response concise and formatted appropriately for social media.
"""


class GeneratedReply(BaseModel):
    reply_text: str
    evidence_ids_used: List[int] = Field(default_factory=list)
    model_name: str
    prompt_version: str = PROMPT_VERSION
    is_grounded_draft: bool = True
    cached: bool = False


class ReplyGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None

    def _lazy_init(self):
        if self._client is None:
            if not self.api_key:
                # Try loading from .env
                import dotenv
                dotenv.load_dotenv()
                self.api_key = os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        customer_message: str,
        thread_transcript: str,
        retrieved_evidence: str,
        retrieved_case_ids: List[int],
        cache: Optional[Dict[str, Any]] = None
    ) -> GeneratedReply:
        """
        Generates a grounded reply. Checks cache first before calling API.
        """
        # Check cache if provided
        cache_key = f"reply_{hashlib.md5((customer_message + PROMPT_VERSION).encode()).hexdigest()}"
        if cache and cache_key in cache:
            entry = cache[cache_key]
            return GeneratedReply(
                reply_text=entry["reply_text"],
                evidence_ids_used=entry.get("evidence_ids_used", retrieved_case_ids),
                model_name=self.model_name,
                prompt_version=PROMPT_VERSION,
                cached=True
            )

        self._lazy_init()

        user_prompt = f"""[CUSTOMER CONVERSATION THREAD]
{thread_transcript}

[RETRIEVED HISTORICAL RESOLUTIONS EVIDENCE]
{retrieved_evidence}

Please draft the official @Uber_Support response following all strict grounding rules:"""

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self.temperature,
                max_output_tokens=200,
            )
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config
            )
            reply_text = response.text.strip()
        except Exception as e:
            # Fallback deterministic grounded template in case of API network failure
            reply_text = (
                "Hi there, we'd like to look into this for you. Please send us a DM with your registered "
                "email address and trip details so our team can assist: https://t.co/help"
            )

        result = GeneratedReply(
            reply_text=reply_text,
            evidence_ids_used=retrieved_case_ids,
            model_name=self.model_name,
            prompt_version=PROMPT_VERSION,
            cached=False
        )

        if cache is not None:
            cache[cache_key] = {
                "reply_text": result.reply_text,
                "evidence_ids_used": result.evidence_ids_used
            }

        return result
