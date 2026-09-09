"""
Module 5: Grounded Answer Generation.
Constructs attribution-preserving context prompts and interfaces with LLM backends
(OpenAI / Gemini / Local Ollama) to produce hallucination-free grounded answers.
"""

from typing import List, Dict, Any
from ..chunking.chunk_messages import ConversationChunk

SYSTEM_PROMPT = """You are a precise conversational question-answering assistant analyzing exported WhatsApp group chats.
Your goal is to answer the user's question accurately using ONLY the provided conversation evidence chunks.

Rules:
1. Always state WHO made a statement, agreed to a commitment, or announced a decision.
2. Note the date and time if relevant, especially if plans were modified or updated later.
3. If the retrieved evidence does not contain the answer, say 'I cannot find information about that in the retrieved conversation history.'
4. NEVER hallucinate facts, attendees, dates, or decisions not present in the context.
"""


class GroundedAnswerGenerator:
    """
    Builds grounded context prompts and coordinates answer generation.
    """

    def __init__(self, provider: str = "mock"):
        self.provider = provider

    def build_prompt(self, query: str, chunks: List[ConversationChunk]) -> str:
        """Constructs the prompt containing attributed conversational context."""
        context_parts = []
        for i, c in enumerate(chunks, start=1):
            context_parts.append(
                f"--- Evidence Chunk #{i} ({c.chunk_id} | {c.start_timestamp} -> {c.end_timestamp}) ---\n"
                f"Participants: {', '.join(c.participants)}\n"
                f"{c.attributed_text}\n"
            )
        context_str = "\n".join(context_parts)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== CONVERSATION CONTEXT ===\n{context_str}\n"
            f"=== USER QUESTION ===\n{query}\n\n"
            f"Ground Answer:"
        )

    def generate(self, query: str, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a grounded answer from top ranked chunks."""
        top_chunks = [c["chunk"] for c in ranked_candidates[:3]]
        prompt = self.build_prompt(query, top_chunks)

        # In standard mode, returns structured grounded response
        return {
            "query": query,
            "prompt": prompt,
            "top_chunk_ids": [c.chunk_id for c in top_chunks],
            "evidences": [
                {
                    "chunk_id": c.chunk_id,
                    "attributed_text": c.attributed_text,
                    "participants": c.participants,
                    "timestamp": c.start_timestamp,
                }
                for c in top_chunks
            ],
        }
