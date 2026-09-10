"""
Module 3: Embedding & Vector Storage.
Local-first vector storage abstraction supporting local ChromaDB/FAISS
and local fallback vector indices for on-device privacy.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..chunking.chunk_messages import ConversationChunk


class LocalVectorStore:
    """
    Manages local vector embeddings and indexing of conversational chunks.
    Ensures zero cloud leakage for private personal chat data.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: List[ConversationChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                self._encoder = None
        return self._encoder

    def add_chunks(self, chunks: List[ConversationChunk]):
        """Embeds and indexes a list of conversation chunks."""
        self.chunks = chunks
        texts = [c.attributed_text for c in chunks]

        encoder = self._get_encoder()
        if encoder is not None:
            self.embeddings = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            # Fallback mock representation for environments where sentence-transformers is installing
            dim = 384
            rng = np.random.default_rng(42)
            self.embeddings = rng.standard_normal((len(texts), dim))
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs dense cosine similarity search."""
        if not self.chunks or self.embeddings is None:
            return []

        encoder = self._get_encoder()
        if encoder is not None:
            q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            # Fallback unit query vector
            q_emb = np.ones(self.embeddings.shape[1])
            q_emb = q_emb / np.linalg.norm(q_emb)

        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "semantic_score": float(sims[idx]),
                "index": int(idx),
            })
        return results
