# PDF RAG Chatbot — Functional & Technical Specification

## 1. Purpose

A local Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions
over a multi-page PDF document (e.g. a resume or a technical product manual). The system indexes
the PDF on startup, stores embeddings in an in-memory FAISS vector store, and serves a
conversational Streamlit interface backed by Google Gemini. The project is structured into
decoupled development milestones for clean incremental testing and GitHub delivery.

---

## 2. Functional Requirements

| ID   | Requirement                                                                                                                                        |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| F-01 | On startup, automatically load and index `./data/document.pdf` — no user action required.                                                         |
| F-02 | Accept natural-language questions via a Streamlit chat input widget.                                                                               |
| F-03 | Before querying FAISS, rewrite ambiguous follow-up questions into a complete standalone query using a history-aware retriever chain.               |
| F-04 | Retrieve the top-5 most semantically relevant chunks from the FAISS index using the rewritten query.                                              |
| F-05 | Generate answers using Google Gemini conditioned on retrieved context + conversation history.                                                      |
| F-06 | If retrieved chunks do not contain the answer, respond with exactly: "I do not know the answer based on the provided documentation." No extrapolation permitted. |
| F-07 | Display each answer with inline citations: source page number + a short text excerpt.                                                              |
| F-08 | Maintain full multi-turn conversation memory within a browser session.                                                                             |
| F-09 | Provide a "Clear Chat" sidebar button that resets history without restarting or re-indexing.                                                       |

---

## 3. Non-Functional Requirements

| ID    | Requirement                                                                       |
|-------|-----------------------------------------------------------------------------------|
| NF-01 | Startup indexing completes in under 30 seconds.                                  |
| NF-02 | Per-query latency (retrieval + generation) target is under 5 seconds.            |
| NF-03 | All secrets (`GOOGLE_API_KEY`) are loaded from a `.env` file — never hard-coded.                    |
| NF-04 | A single `requirements.txt` reproduces the full environment with `pip install`.  |
| NF-05 | No external database, no Docker, no cloud infrastructure beyond the Gemini API.  |

---

## 4. Project File Tree

```
pdf_chatbot_rag/
├── data/
│   ├── document.pdf              # Input PDF (user-supplied, gitignored)
│   └── .gitkeep                  # Keeps the data/ folder tracked in git
├── src/
│   ├── __init__.py               # Marks src/ as a Python package
│   ├── ingest.py                 # Pillar 1+2+3+4: load, chunk, embed, index
│   ├── retriever.py              # Pillar 5: FAISS retriever wrapper
│   ├── llm.py                    # Pillar 6: Gemini LLM client
│   ├── chain.py                  # RAG orchestration (history-aware chain)
│   └── app.py                    # Streamlit UI entry point
├── config.py                     # Central constants (all tuneable values)
├── .env                          # Runtime secrets — never committed
├── .env.example                  # Committed template for .env
├── .gitignore                    # Excludes: .env, data/document.pdf, __pycache__, .venv
├── requirements.txt              # Pinned Python dependencies
└── specs.md                      # This document
```

---

## 5. Architectural Boundaries

The system is divided into four strictly decoupled layers. No layer may import from a layer above it.

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4 — Presentation Layer                           │
│  src/app.py                                             │
│  Responsibility: UI rendering, session state, user I/O  │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Orchestration Layer                          │
│  src/chain.py                                           │
│  Responsibility: question rewriting, prompt assembly,   │
│                  chain execution, source extraction      │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — Retrieval & Generation Layer                 │
│  src/retriever.py  ·  src/llm.py                        │
│  Responsibility: FAISS similarity search, LLM calls     │
├─────────────────────────────────────────────────────────┤
│  Layer 1 — Ingestion Layer                              │
│  src/ingest.py                                          │
│  Responsibility: PDF parsing, chunking, embedding,      │
│                  FAISS index construction               │
└─────────────────────────────────────────────────────────┘
         ▲ shared read-only config via config.py
```

**Coupling rules:**
- `app.py` calls `chain.py` only — it has no knowledge of FAISS, embeddings, or LLM internals.
- `chain.py` calls `retriever.py` and `llm.py` — it does not parse PDFs or build indexes.
- `ingest.py` is called once (at startup via `@st.cache_resource`) and returns only a FAISS index object.
- `config.py` is a passive constants file — it is imported by any layer but imports nothing itself.

---

## 6. RAG Pipeline: The 6 Pillars

### Pillar 1 — Document Loader (`src/ingest.py`)

| Property | Value |
|----------|-------|
| Library | PyMuPDF (`fitz`) |
| Input | `data/document.pdf` (path from `config.PDF_PATH`) |
| Method | `fitz.open()` → iterate pages → `page.get_text()` |
| Output per page | Raw text string + page number (1-indexed) |
| Metadata attached | `{ "page": int, "source": "data/document.pdf" }` |
| Supported documents | Any text-layer PDF: resumes, technical manuals, reports |

Pages with empty text (e.g. pure image pages) are skipped silently.

---

### Pillar 2 — Text Splitter (`src/ingest.py`)

| Property | Value |
|----------|-------|
| Class | `RecursiveCharacterTextSplitter` (LangChain) |
| Splitting mode | Character-based (not token-based) |
| `chunk_size` | `1000` characters |
| `chunk_overlap` | `200` characters |
| Effective stride | `800` characters |
| Separator hierarchy | `["\n\n", "\n", " ", ""]` — tries largest break first |
| Output | `List[Document]` with inherited page metadata |

Overlap ensures that sentences or terms split across a chunk boundary still appear in at least
one complete chunk, maintaining semantic coherence for retrieval.

---

### Pillar 3 — Embeddings (`src/ingest.py`)

| Property | Value |
|----------|-------|
| Provider | Google AI |
| Model | `gemini-embedding-001` |
| LangChain class | `GoogleGenerativeAIEmbeddings` |
| Output dimensions | 3072 (float32) |
| Vector size | 3072 × 4 bytes = **12,288 bytes per chunk** |
| API key | `GOOGLE_API_KEY` from environment |
| Batching | Handled internally by LangChain |

---

### Pillar 4 — Vector Store: FAISS (`src/ingest.py`)

| Property | Value |
|----------|-------|
| Library | `faiss-cpu` |
| LangChain class | `FAISS` (from `langchain_community.vectorstores`) |
| Index type | `IndexFlatL2` (exact L2 distance, default for small corpora) |
| Construction | `FAISS.from_documents(chunks, embeddings)` |
| Persistence | None — in-memory only; rebuilt on every server restart |
| Lifecycle | Cached for the session lifetime via `@st.cache_resource` |

---

### Pillar 5 — Retriever (`src/retriever.py` + `src/chain.py`)

The retriever operates in two stages:

**Stage A — Question Rewriting (History-Aware)**

| Property | Value |
|----------|-------|
| LangChain function | `create_history_aware_retriever` |
| Input | Raw user question + `chat_history` (LangChain message list) |
| Process | LLM reformulates ambiguous follow-ups into standalone queries |
| Example | "What are her certifications?" → "What certifications does Jane Doe hold?" |
| Prompt used | Contextualization prompt (see §7.1) |

**Stage B — FAISS Similarity Search**

| Property | Value |
|----------|-------|
| Index queried | In-memory FAISS `IndexFlatL2` |
| Similarity metric | L2 (Euclidean distance on 3072-dim vectors) |
| Top-k | `k=5` chunks returned |
| Output | `List[Document]` with `page_content` + `metadata` |

---

### Pillar 6 — LLM (`src/llm.py`)

| Property | Value |
|----------|-------|
| Provider | Google Gemini |
| Model | `gemini-2.0-flash` |
| LangChain class | `ChatGoogleGenerativeAI` |
| Temperature | `0.0` — fully deterministic; no creative extrapolation |
| Context window | 1,000,000 tokens (far exceeds any expected prompt size) |
| Roles in pipeline | (a) Question rewriter in Stage A; (b) Answer generator in Stage B |
| Fallback behaviour | Exact string returned when context is insufficient (see §7.2) |

---

## 7. Prompt Specifications

Two prompts govern LLM behaviour. Both are defined as static strings in `src/chain.py`.

### 7.1 Contextualization Prompt

**Purpose:** Rewrites ambiguous follow-up questions into standalone queries before FAISS lookup.
**Used by:** `create_history_aware_retriever`

```
╔══════════════════════════════════════════════════════════════╗
║  CONTEXTUALIZATION SYSTEM PROMPT                            ║
╠══════════════════════════════════════════════════════════════╣
║  Given the conversation history and the latest user         ║
║  question, reformulate the question so it is fully          ║
║  self-contained and understandable without the history.     ║
║                                                              ║
║  Rules:                                                      ║
║  - Do NOT answer the question.                               ║
║  - Only rewrite if the question references prior context.    ║
║  - If the question is already standalone, return it as-is.  ║
╚══════════════════════════════════════════════════════════════╝
```

### 7.2 Answer System Prompt (Anti-Hallucination Guardrail)

**Purpose:** Generates the final answer strictly from retrieved context.
**Used by:** `create_retrieval_chain` answer step.

```
╔══════════════════════════════════════════════════════════════╗
║  ANSWER SYSTEM PROMPT                                       ║
╠══════════════════════════════════════════════════════════════╣
║  You are a precise assistant that answers questions         ║
║  strictly from the context provided below.                  ║
║                                                              ║
║  Rules:                                                      ║
║  - Use ONLY the context below. No outside knowledge.        ║
║  - If the answer is not explicitly in the context,          ║
║    respond with exactly:                                     ║
║    "I do not know the answer based on the provided          ║
║     documentation."                                         ║
║  - Do not extrapolate, infer, summarise beyond the text,    ║
║    or guess.                                                 ║
║  - Cite page numbers inline as [Page N].                    ║
║                                                              ║
║  Context:                                                    ║
║  {context}                                                   ║
╚══════════════════════════════════════════════════════════════╝
```

### 7.3 Context Block Format

Each of the 5 retrieved chunks is formatted as:

```
┌─────────────────────────────────┐
│ --- Page {page_number} ---      │
│ {chunk_text}                    │
└─────────────────────────────────┘
```

Blocks are concatenated with a blank line separator before insertion into the answer prompt.

---

## 8. Data Footprint & Memory Estimates

Estimates below assume a **2-page prototype PDF** with dense text (≈ 3,000 characters/page).

### 8.1 Chunk Count

| Variable | Calculation | Result |
|----------|-------------|--------|
| Total characters | 2 pages × 3,000 chars | ~6,000 chars |
| Effective stride | chunk_size − chunk_overlap = 1,000 − 200 | 800 chars |
| Approximate chunk count | ⌈6,000 / 800⌉ | **~8 chunks** |

Actual count varies by page density; image-heavy pages produce fewer chunks.

### 8.2 Embedding & FAISS Memory

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Vector dimensions | gemini-embedding-001 output | 3072 dims |
| Bytes per float | float32 | 4 bytes |
| Bytes per vector | 3072 × 4 | 12,288 bytes (12 KB) |
| Total vector data | 8 × 12,288 | ~96 KB |
| FAISS `IndexFlatL2` overhead | index metadata + L2 norms | ~50 KB |
| **Total FAISS index size** | | **< 1 MB** |

### 8.3 Embedding API Calls at Startup

| Variable | Value |
|----------|-------|
| Chunks to embed | ~8 |
| Google API batch size | up to 100 embeddings per request |
| API calls required | 1 request |
| Estimated startup time | 2–5 seconds (network dependent) |

### 8.4 Per-Query LLM Context Budget

| Component | Token estimate |
|-----------|----------------|
| System prompt (§7.2) | ~120 tokens |
| 5 retrieved chunks (×~300 tokens each) | ~1,500 tokens |
| Conversation history (5 turns) | ~500 tokens |
| User question | ~30 tokens |
| **Total prompt tokens** | **~2,150 tokens** |
| Gemini 2.0 Flash context window | 1,000,000 tokens |
| Headroom | >99.8% |

### 8.5 Total Process Memory

| Component | Estimate |
|-----------|----------|
| Python + Streamlit + LangChain baseline | ~250–350 MB |
| FAISS index | < 1 MB |
| In-flight embedding tensors (startup only) | ~10 MB |
| **Total RAM at runtime** | **~300–400 MB** |

---

## 9. Data Flow

```
App Start
  └─► ingest.py reads PDF → extracts pages → splits into 1000-char chunks (200 overlap)
      → embeds with gemini-embedding-001 (~2 API calls) → builds FAISS index (cached in memory)

User submits question
  └─► chain.py (history-aware retrieval chain):
        1. History-aware retriever (Pillar 5, Stage A):
           - LLM rewrites follow-up question into standalone query using chat_history
        2. Standalone query → FAISS similarity search → top-5 Document objects (Stage B)
        3. Context blocks assembled from retrieved chunks (labeled by page)
        4. Answer prompt built: §7.2 guardrail + context blocks + chat_history + question
        5. Gemini LLM (temperature=0.0) generates answer
           - If context lacks the answer → exact fallback string returned
        6. Sources extracted from Document metadata
        └─► Returns { answer, sources }

Streamlit renders
  └─► Assistant chat bubble with answer text
  └─► Collapsed citation expanders (Page N + excerpt)
  └─► Turn appended to st.session_state.chat_history (LangChain message list)
```

---

## 10. Git Lifecycle Phasing

The project is delivered in four decoupled milestones. Each milestone produces a self-contained,
testable increment that is committed and pushed independently.

| Phase | Milestone | Files Delivered | Testable Outcome |
|-------|-----------|-----------------|------------------|
| 1 | Project scaffold + config | `config.py`, `.env.example`, `requirements.txt`, `.gitignore`, `data/.gitkeep` | `pip install` succeeds; `python config.py` imports cleanly |
| 2 | Ingestion pipeline | `src/__init__.py`, `src/ingest.py` | Script runs standalone; FAISS index builds; chunk count printed to stdout |
| 3 | RAG chain (no UI) | `src/retriever.py`, `src/llm.py`, `src/chain.py` | CLI smoke test: question in → answer + sources out; fallback string verified with off-topic question |
| 4 | Streamlit UI | `src/app.py` | `streamlit run src/app.py` loads; citations render; multi-turn memory verified; "Clear Chat" works |

Each phase is committed on its own branch (`phase/1-scaffold`, `phase/2-ingest`, `phase/3-chain`,
`phase/4-ui`) and merged to `main` via a pull request before the next phase begins.

---

## 11. Scope Boundaries

**In scope:**
- Single fixed PDF document
- Session-scoped FAISS index (no disk persistence)
- Single-user local application

**Out of scope:**
- Multi-PDF support
- User authentication
- Persistent vector store across restarts
- PDF upload UI
- Cloud deployment
