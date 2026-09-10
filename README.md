# Retrieval Beyond Text: Context-Aware Retrieval Framework for Multi-Party, Time-Dependent Conversations

[![Project Phase](https://img.shields.io/badge/Project-Phase%20II-blue.svg)](https://github.com/)
[![Academic Year](https://img.shields.io/badge/Academic%20Year-2026--2027-green.svg)](https://github.com/)
[![Department](https://img.shields.io/badge/Amrita%20School%20of%20Computing-CSE-orange.svg)](https://github.com/)

**Course**: 23CSE498 – Project Phase II  
**Institution**: Amrita School of Computing, Amritapuri Campus, Amrita Vishwa Vidyapeetham  
**Project Guide**: Ms. Bindhya Bhadran  
**Team Members (Team B10)**:
- K. Sai Dinesh (AM.SC.U4CSE23130)
- Satya Sri Dheeraj Motupalli (AM.SC.U4CSE23150)
- Y. Sai Nikhil (AM.SC.U4CSE23171)

---

## 📖 1. Project Overview

Conversational data generated in messaging applications like WhatsApp is fundamentally different from traditional structured documents:
- **Fragmented utterances** ("Yeah", "Okay", "Tomorrow") lacking independent semantic context
- **Multi-party interleaved dialogues** with multiple speakers and roles
- **Abrupt topic switches** within short timeframes
- **Time-varying decisions** where newer messages supersede older commitments
- **Privacy constraints** demanding local/on-device query processing rather than cloud ingestion

Generic RAG systems (built on LangChain, LlamaIndex, Haystack) and conversational memory systems (MemGPT, SECOM) fail on multi-party chats because they were designed either for structured prose or 1-on-1 human-to-AI dialogues.

This project designs, develops, and benchmarks a **Context-Aware Conversational Retrieval Pipeline** that introduces:
1. **Speaker-Aware Context Preservation**: Preserving message author attribution within each chunk.
2. **Topic-Aware Semantic Chunking**: Detecting topic shift boundaries to prevent unrelated topic bleed.
3. **Thread/Reply Approximation**: Reconstructing conversational link graphs using semantic similarity and timing proximity.
4. **Time-Weighted Re-Ranking**: Balancing semantic relevance with temporal freshness via exponential time decay:
   $$\text{Score}(c) = \alpha \cdot \text{sim}(q, c) + (1 - \alpha) \cdot e^{-\lambda \cdot \Delta t(c)}$$
5. **Diagnostic Failure Benchmarking**: Evaluating precision, recall, and categorizing failure modes (Bad Chunk Boundary, Lost Speaker Context, Wrong Time Window, Irrelevant Retrieval) against naive baselines and general RAG frameworks.

---

## 🏛️ 2. System Architecture

```text
Raw WhatsApp Chat Export (.txt)
             │
             ▼
   [Chat Parsing Module]
   • Parse timestamps, senders, text
   • Merge multi-line continuation messages
   • Filter system notifications & media placeholders
             │
             ▼
   [Chunking Module]
   ├── Baseline: Naive Time-Gap Chunking (30-min threshold)
   └── Proposed: Context-Aware Chunking (Speaker + Topic + Thread Approximation)
             │
             ▼
   [Embedding & Vector Store]
   • Dense embeddings (Sentence-Transformers / Multilingual)
   • Vector indexing (ChromaDB / FAISS)
             │
             ▼
   [Retrieval & Time-Weighted Re-ranking]
   • Semantic vector similarity
   • Dynamic time-decay re-ranking (alpha, lambda)
             │
             ▼
   [LLM Generation & Grounding]
   • Context-grounded synthesis with speaker attribution & timestamps
             │
             ▼
   [Evaluation & Diagnostic Benchmark]
   • Benchmark query sets with ground truth
   • Comparative analysis vs. Baselines & Frameworks
   • Failure Mode Classification
```

---

## 📂 3. Repository Structure

```tree
├── docs/                     # System design specifications and architecture documentation
│   ├── SYSTEM_DESIGN.md      # Detailed technical architecture, schemas, and algorithms
│   └── UI_UX_SPECIFICATION.md# UI/UX design specs, layouts, interactive prototypes
├── data/
│   ├── raw/                  # Raw WhatsApp export files (.txt) [git-ignored]
│   │   └── sample_chat.txt   # Synthetic benchmark chat export for testing
│   ├── processed/            # Structured parsed message JSONs [git-ignored]
│   └── benchmarks/           # Hand-labeled query sets & ground-truth answers
├── src/                      # Source code modules
│   ├── parser/               # WhatsApp export parsing logic
│   ├── chunking/             # Naive & Context-Aware chunking implementations
│   ├── indexing/             # Vector store & embedding managers
│   ├── retrieval/            # Semantic search & time-decay re-ranker
│   ├── generation/           # LLM prompting & grounded response generator
│   └── evaluation/           # Evaluation benchmark runner & failure diagnostic
├── ui/                       # Interactive Design Prototype & Visualizer Dashboard
├── eval_results/             # Evaluation logs and comparative benchmark reports
├── Detailed_Project_Explanation.md # Comprehensive conceptual project guide
├── Design_Proposal (1) (1).pdf    # Department submission: System Design Proposal
├── proposal-1.pdf                 # Department submission: Initial Project Idea
├── requirements.txt          # Python dependencies
└── .gitignore                # Privacy & environment security exclusions
```

---

## 🚀 4. Roadmap & Milestone Progress Log

| Stage | Milestone / Task | Status | Script / Artifact |
| :--- | :--- | :---: | :--- |
| **Stage 1** | Chat Parsing & Multi-line Sanitization | ✅ Completed | `src/parser/parse_whatsapp.py` |
| **Stage 2** | Baseline vs. Context-Aware Chunking Engines | ✅ Completed | `src/chunking/chunk_messages.py` |
| **Stage 3** | Embedding & Vector Storage (Local-First) | ✅ Completed | `src/indexing/vector_store.py` |
| **Stage 4** | Benchmark Query Sets (25-query & 18-query suites) | ✅ Completed | `data/benchmarks/` |
| **Stage 5** | Baseline Retrieval Test (Naive Time-Gap 30m) | ✅ Completed | `scripts/run_naive_baseline.py` |
| **Stage 6** | Framework Benchmarking (LangChain & LlamaIndex) | ✅ Completed | `scripts/run_langchain.py`, `scripts/run_llamaindex.py` |
| **Stage 7** | Adapted Retrieval Design (Attributed + Time-Decay) | ✅ Completed | `scripts/run_context_aware.py`, `src/retrieval/ranker.py` |
| **Stage 8** | Empirical Comparative Study & Failure Taxonomy | ✅ Completed | `src/evaluation/run_comparative_study.py`, `scripts/compare_results.py` |
| **Stage 9** | Interactive Prototype Visualizer Studio | ✅ Completed | `ui/index.html` (Vanilla HTML/CSS/JS) |

---

## 📊 5. Empirical Benchmark Results

### 5.1 Comparative Study: Standard vs. Proposed Frameworks (25-Query Ground Truth Suite)

Evaluated on multi-turn dialogue containing multi-party continuations, temporal decision reversals (January vs. August), and speaker attributions using `sentence-transformers/all-MiniLM-L6-v2`:

| Framework | Hit Rate @ 1 | Hit Rate @ 3 | MRR | Speaker Attrib Acc | Temporal Validity Rate | Clean (No Failure) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Time-Gap Baseline** | **92.0%** | **100.0%** | **0.960** | **100.0%** | 66.7% | 23 / 25 |
| **LangChain Standard RAG** | 84.0% | 96.0% | 0.887 | 92.0% | 33.3% | 21 / 25 |
| **LlamaIndex Standard RAG** | 84.0% | **100.0%** | 0.913 | **100.0%** | 66.7% | 23 / 25 |
| **Proposed Context-Aware RAG** | 84.0% | 96.0% | 0.893 | 96.0% | **100.0%** | **24 / 25** |

### 5.2 Failure Mode Distribution

| Framework | Bad Boundary | Lost Speaker Context | Wrong Time Window | Total Failures |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Time-Gap Baseline** | 0 | 0 | 2 | 2 |
| **LangChain Standard RAG** | 1 | 1 | 2 | 4 |
| **LlamaIndex Standard RAG** | 0 | 0 | 2 | 2 |
| **Proposed Context-Aware RAG** | 1 | 0 | **0** | **1** |

> [!TIP]
> **Key Finding**: Generic document-oriented frameworks (LangChain / LlamaIndex) suffer up to a **66.7% temporal failure rate**, repeatedly retrieving outdated January decisions rather than current August decisions. The Proposed Context-Aware RAG achieves **100.0% temporal validity** with zero time-inversion failures.

---

## 📚 6. Literature Review Summary

| # | Paper | Year | Methodology | Gap Relative to Conversational RAG |
|---|-------|------|-------------|-----------------------------------|
| 1 | **SECOM** (Pang et al., ICLR) | 2025 | Segment-level memory + compression-based denoising | Single-user AI dialogue only; lacks multi-speaker attribution |
| 2 | **MemGPT** (Packer et al.) | 2023 | OS-inspired hierarchical memory paging | Assumes single 1-on-1 dialogue; does not handle multi-party chats |
| 3 | **Alonso et al.** | 2024 | Time-sensitive long-term memory | Time-awareness applied only within single-user assistant memory |
| 4 | **Gutiérrez et al.** | 2025 | Non-parametric continual learning | Built for general knowledge/QA corpora, not personal chat |
| 5 | **MultiHop-RAG** (Tang & Yang) | 2024 | Multi-hop QA benchmark | Evaluates structured news/document prose, not fragmented messaging |

---

## 💻 7. Reproducing the Comparative Study

Run the benchmark suite using the self-contained evaluation scripts in `scripts/`:

```powershell
# 1. Verify Python dependencies
python scripts/check_env.py

# 2. Run individual framework benchmark evaluations
python scripts/run_naive_baseline.py
python scripts/run_langchain.py
python scripts/run_llamaindex.py
python scripts/run_context_aware.py

# 3. View the aggregated comparison table and category breakdown
python scripts/compare_results.py

# 4. Run the comprehensive automated 25-query diagnostic evaluation
python -m src.evaluation.run_comparative_study
```
