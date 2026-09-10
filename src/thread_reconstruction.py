"""
Stage 2: Thread Reconstruction Module.
Joins current customer turn with bounded prior conversation turns (up to max_turns)
into a structured, typed conversation thread object.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str  # "customer" or "agent"
    text: str
    tweet_id: Optional[int] = None
    created_at: Optional[str] = None


class ThreadContext(BaseModel):
    current_message: str
    current_tweet_id: Optional[int] = None
    prior_turns: List[Turn] = Field(default_factory=list)
    total_turns: int = 1
    formatted_transcript: str = ""


class ThreadReconstructor:
    def __init__(self, max_context_turns: int = 4):
        self.max_context_turns = max_context_turns

    def reconstruct(
        self,
        current_text: str,
        current_tweet_id: Optional[int] = None,
        prior_turns: Optional[List[Dict[str, Any]]] = None
    ) -> ThreadContext:
        """
        Reconstructs an ordered, bounded conversation thread.
        Limits historical context to max_context_turns to control token budget.
        """
        parsed_turns: List[Turn] = []
        if prior_turns:
            for pt in prior_turns[-self.max_context_turns:]:
                parsed_turns.append(Turn(
                    role=pt.get("role", "customer"),
                    text=pt.get("text", ""),
                    tweet_id=pt.get("tweet_id"),
                    created_at=pt.get("created_at")
                ))

        # Build human-readable formatted transcript
        transcript_lines = []
        for t in parsed_turns:
            speaker = "Customer" if t.role == "customer" else "Support Agent"
            transcript_lines.append(f"{speaker}: {t.text}")

        transcript_lines.append(f"Customer (Current): {current_text}")
        formatted = "\n".join(transcript_lines)

        return ThreadContext(
            current_message=current_text,
            current_tweet_id=current_tweet_id,
            prior_turns=parsed_turns,
            total_turns=len(parsed_turns) + 1,
            formatted_transcript=formatted
        )
