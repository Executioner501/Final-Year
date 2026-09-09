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

## 🚀 4. Roadmap & Deliverables

- [x] Initial Project Formulation & Literature Survey
- [x] Git Repository Setup & Privacy Sanitization
- [x] Comprehensive System & UI/UX Architecture Design
- [ ] Chat Parser Implementation (`parse_whatsapp.py`)
- [ ] Baseline vs. Context-Aware Chunking Engines (`chunk_messages.py`)
- [ ] Vector Embedding & Storage Integration (FAISS / ChromaDB)
- [ ] Time-Aware Re-ranking Engine
- [ ] Grounded Answer Generation with Citation Traceability
- [ ] Diagnostic Evaluation Benchmarking & Failure Taxonomy
- [ ] Interactive Prototype Dashboard & Presentation Deliverables
