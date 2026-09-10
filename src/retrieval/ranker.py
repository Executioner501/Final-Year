"""
Module 4: Time-Aware Semantic Retrieval & Re-ranking.
Implements the core mathematical formulation:
Score(c) = alpha * Sim(q, c) + (1 - alpha) * exp(-lambda * delta_t_norm)
"""

from typing import List, Dict, Any
from datetime import datetime
import math
from ..chunking.chunk_messages import _parse_ts


class TimeAwareReRanker:
    """
    Reranks candidate conversational chunks by weighting semantic similarity
    against temporal recency decay.
    """

    def __init__(self, alpha: float = 0.5, decay_lambda: float = 1.0):
        self.alpha = max(0.0, min(1.0, alpha))
        self.decay_lambda = max(0.0, decay_lambda)

    def rerank(
        self, 
        candidates: List[Dict[str, Any]], 
        reference_time: datetime = None,
        query: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates using combined semantic and exponential time decay score.
        If query explicitly specifies historical intent (e.g. 'initial', 'back in January'),
        it adjusts temporal weighting to prevent penalizing relevant historical context.
        """
        if not candidates:
            return []

        # Detect temporal intent from query
        q_lower = query.lower() if query else ""
        is_historical_query = any(w in q_lower for w in ["initial", "earlier", "january", "back in", "first agreed", "originally"])
        is_latest_query = any(w in q_lower for w in ["latest", "current", "updated", "now", "newest", "recently", "active"])

        # Effective alpha: if historical or neutral, rely primarily on semantic matching
        if is_historical_query:
            effective_alpha = 0.95  # Semantic match dominates for historical inquiries
            decay_mult = 0.05
        elif is_latest_query:
            effective_alpha = 0.40  # Heavy recency decay for latest updates
            decay_mult = self.decay_lambda
        else:
            # For general factoid/conversational queries, semantic relevance dominates,
            # with gentle temporal preference among semantically comparable candidates
            effective_alpha = 0.85
            decay_mult = 0.20

        chunk_dts = [_parse_ts(item["chunk"].end_timestamp) for item in candidates]
        if reference_time is None:
            reference_time = max(chunk_dts)

        elapsed_seconds = [
            max(0.0, (reference_time - dt).total_seconds())
            for dt in chunk_dts
        ]
        max_elapsed = max(elapsed_seconds) if max(elapsed_seconds) > 0 else 1.0

        reranked = []
        for i, item in enumerate(candidates):
            sim_score = float(item.get("semantic_score", 0.0))
            norm_delta_t = elapsed_seconds[i] / max_elapsed
            time_decay = math.exp(-decay_mult * norm_delta_t)

            final_score = (effective_alpha * sim_score) + ((1.0 - effective_alpha) * time_decay)

            reranked.append({
                "chunk": item["chunk"],
                "semantic_score": sim_score,
                "time_decay": time_decay,
                "final_score": final_score,
                "elapsed_seconds": elapsed_seconds[i],
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked
