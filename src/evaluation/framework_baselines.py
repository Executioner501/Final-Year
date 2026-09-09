"""
Framework Baselines for Comparative Study:
1. Naive Baseline (30-min time gap silence chunking)
2. LangChain Standard RAG (RecursiveCharacterTextSplitter document chunking)
3. LlamaIndex Standard RAG (SentenceSplitter token chunking)
4. Proposed Context-Aware RAG (Speaker-turn preservation + time-decay re-ranking)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from ..parser.parse_whatsapp import MessageRecord, parse_whatsapp_chat
from ..chunking.chunk_messages import (
    naive_time_gap_chunking,
    context_aware_chunking,
    ConversationChunk,
    _parse_ts
)
from ..retrieval.ranker import TimeAwareReRanker

# Embedding encoder cache
_GLOBAL_MODEL = None


def get_embedding_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _GLOBAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _GLOBAL_MODEL


class BaseRetriever:
    """Base class providing shared embedding and vector similarity capabilities."""

    def __init__(self, name: str):
        self.name = name
        self.chunk_texts: List[str] = []
        self.chunk_metas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def _index_texts(self, texts: List[str], metas: List[Dict[str, Any]]):
        self.chunk_texts = texts
        self.chunk_metas = metas
        model = get_embedding_model()
        self.embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunk_texts or self.embeddings is None:
            return []

        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "text": self.chunk_texts[idx],
                "meta": self.chunk_metas[idx],
                "score": float(sims[idx]),
            })
        return results


# ==========================================
# 1. Naive Time-Gap Baseline
# ==========================================
class NaiveBaselineRAG(BaseRetriever):
    """
    Naive baseline using 30-minute inactivity threshold.
    Converts raw messages into unstructured text blocks.
    """

    def __init__(self, messages: List[MessageRecord], threshold_minutes: int = 30):
        super().__init__("Naive Time-Gap Baseline")
        chunks = naive_time_gap_chunking(messages, threshold_minutes=threshold_minutes)
        texts = [c.raw_text for c in chunks]
        metas = [c.to_dict() for c in chunks]
        self._index_texts(texts, metas)


# ==========================================
# 2. LangChain Standard RAG Framework
# ==========================================
class LangChainRAG(BaseRetriever):
    """
    Standard LangChain document retrieval pipeline.
    Uses RecursiveCharacterTextSplitter treating the chat log as plain prose.
    """

    def __init__(self, raw_text: str, chunk_size: int = 350, chunk_overlap: int = 70):
        super().__init__("LangChain Standard RAG")
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = splitter.split_text(raw_text)
        metas = [{"chunk_id": f"LC-CHUNK-{i:03d}", "source": "langchain_recursive"} for i in range(len(docs))]
        self._index_texts(docs, metas)


# ==========================================
# 3. LlamaIndex Standard RAG Framework
# ==========================================
class LlamaIndexRAG(BaseRetriever):
    """
    Standard LlamaIndex retrieval pipeline.
    Uses SentenceSplitter (token-based sentence splitting).
    """

    def __init__(self, raw_text: str, chunk_size: int = 150, chunk_overlap: int = 25):
        super().__init__("LlamaIndex Standard RAG")
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.split_text(raw_text)
        metas = [{"chunk_id": f"LLAMA-NODE-{i:03d}", "source": "llama_index_sentence"} for i in range(len(nodes))]
        self._index_texts(nodes, metas)


# ==========================================
# 4. Proposed Context-Aware Framework
# ==========================================
class ContextAwareRAG(BaseRetriever):
    """
    Proposed adapted conversational retrieval framework:
    - Explicit speaker turn attribution: [Date Time] Sender: Message
    - Topic and bounded turn chunking
    - Exponential time-decay re-ranking: Score = alpha*Sim + (1-alpha)*exp(-lambda*delta_t)
    """

    def __init__(
        self, 
        messages: List[MessageRecord], 
        alpha: float = 0.45, 
        decay_lambda: float = 1.2
    ):
        super().__init__("Proposed Context-Aware RAG")
        self.alpha = alpha
        self.decay_lambda = decay_lambda
        self.reranker = TimeAwareReRanker(alpha=alpha, decay_lambda=decay_lambda)

        chunks = context_aware_chunking(messages, max_inactivity_minutes=20, max_chunk_size=8)
        self.conv_chunks = chunks
        texts = [c.attributed_text for c in chunks]
        metas = [c.to_dict() for c in chunks]
        self._index_texts(texts, metas)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        candidates = super().retrieve(query, top_k=len(self.conv_chunks))
        
        # Prepare candidates for time-decay reranking
        formatted_candidates = []
        for c in candidates:
            idx = int(c["meta"]["chunk_id"].split("-")[-1]) - 1
            chunk_obj = self.conv_chunks[idx]
            formatted_candidates.append({
                "chunk": chunk_obj,
                "semantic_score": c["score"],
                "text": c["text"],
                "meta": c["meta"]
            })

        reranked = self.reranker.rerank(formatted_candidates, query=query)
        
        results = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            results.append({
                "rank": rank,
                "text": item["chunk"].attributed_text,
                "meta": item["chunk"].to_dict(),
                "score": item["final_score"],
                "semantic_score": item["semantic_score"],
                "time_decay": item["time_decay"],
            })
        return results
