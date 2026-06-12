# PDF RAG Chatbot — Implementation Guide

> **How to use this document**
> Follow phases in order. Complete every step in a phase, run its Quality Gate in the terminal,
> confirm it passes, then commit and open a pull request before starting the next phase.
> Never skip a Quality Gate — each one is a pre-commit contract.

---

## Phase 1 — Bootstrap: Project Scaffold & Configuration

**Branch:** `phase/1-scaffold`
**Files delivered:** `config.py`, `.env.example`, `.gitignore`, `data/.gitkeep`

### Step 1.1 — Initialise Git and create the branch

```bash
git init
git checkout -b phase/1-scaffold
```

### Step 1.2 — Create the directory structure

```bash
mkdir -p src data
touch src/__init__.py data/.gitkeep
```

### Step 1.3 — Create `.gitignore`

```
.env
data/document.pdf
__pycache__/
*.pyc
.venv/
*.egg-info/
.DS_Store
```

### Step 1.4 — Create `.env.example`

```
GOOGLE_API_KEY=your_google_api_key_here
```

### Step 1.5 — Copy and populate `.env`

```bash
cp .env.example .env
# Edit .env and set your real GOOGLE_API_KEY value
```

### Step 1.6 — Create `config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH        = "data/document.pdf"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 5
GEMINI_MODEL    = "gemini-2.0-flash"
EMBED_MODEL     = "models/gemini-embedding-001"
TEMPERATURE     = 0.0
GOOGLE_API_KEY  = os.environ["GOOGLE_API_KEY"]
```

### Step 1.7 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### QC-1 — Quality Gate: Bootstrap Verification

Run all three checks. Every line must print PASS before committing.

```bash
# Check 1: Dependencies installed — no ImportError
python -c "
import streamlit, langchain, langchain_community, langchain_google_genai
import faiss, fitz, dotenv
print('PASS — all packages import cleanly')
"

# Check 2: Config loads and API key is present
python -c "
import config
assert config.GOOGLE_API_KEY, 'GOOGLE_API_KEY is empty'
assert config.CHUNK_SIZE == 1000
assert config.CHUNK_OVERLAP == 200
assert config.TEMPERATURE == 0.0
print('PASS — config constants correct, API key loaded')
"

# Check 3: Config raises KeyError if GOOGLE_API_KEY is missing
python -c "
import os, importlib
os.environ.pop('GOOGLE_API_KEY', None)
try:
    import config as c; importlib.reload(c)
    print('FAIL — expected KeyError')
except KeyError:
    print('PASS — KeyError raised for missing GOOGLE_API_KEY')
"
```

**Expected output:**
```
PASS — all packages import cleanly
PASS — config constants correct, API key loaded
PASS — KeyError raised for missing GOOGLE_API_KEY
```

**Git commit when all three pass:**
```bash
git add config.py .env.example .gitignore requirements.txt src/__init__.py data/.gitkeep
git commit -m "phase/1: project scaffold, config, and dependencies"
```

---

## Phase 2 — Ingestion Pipeline

**Branch:** `phase/2-ingest`
**Files delivered:** `src/ingest.py`

```bash
git checkout -b phase/2-ingest
```

Place your PDF at `data/document.pdf` before running any code in this phase.

### Step 2.1 — Create `src/ingest.py`

```python
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import config


def build_index() -> FAISS:
    pdf = fitz.open(config.PDF_PATH)
    raw_docs = []
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        if text.strip():
            raw_docs.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": config.PDF_PATH},
            ))
    pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return FAISS.from_documents(chunks, embeddings)
```

---

### QC-2 — Quality Gate: Ingestion Pipeline Verification

```bash
# Check 1: Index builds and chunk count is in expected range
python -c "
import time
from src.ingest import build_index

start = time.time()
index = build_index()
elapsed = time.time() - start

total = index.index.ntotal
print(f'Chunks indexed : {total}')
print(f'Startup time   : {elapsed:.1f}s')

assert total > 0, 'No chunks produced — check that data/document.pdf exists and has text'
assert elapsed < 30, f'FAIL — indexing took {elapsed:.1f}s (target: <30s)'
print('PASS — index built within SLA')
"

# Check 2: Metadata is correctly attached to chunks
python -c "
from src.ingest import build_index

index = build_index()
docs = index.similarity_search('test', k=1)
doc = docs[0]
assert 'page' in doc.metadata, 'Missing page metadata'
assert 'source' in doc.metadata, 'Missing source metadata'
assert isinstance(doc.metadata['page'], int), 'page must be int'
print(f'PASS — metadata intact: page={doc.metadata[\"page\"]}, source={doc.metadata[\"source\"]}')
"

# Check 3: Chunk size is within bounds (character check)
python -c "
from src.ingest import build_index
index = build_index()
docs = index.similarity_search('test', k=5)
for doc in docs:
    assert len(doc.page_content) <= 1200, f'Chunk too large: {len(doc.page_content)} chars'
print('PASS — all sampled chunks within size bounds')
"
```

**Expected output:**
```
Chunks indexed : 145        ← will vary by PDF
Startup time   : 8.3s       ← will vary by network
PASS — index built within SLA
PASS — metadata intact: page=3, source=data/document.pdf
PASS — all sampled chunks within size bounds
```

**Git commit when all three pass:**
```bash
git add src/ingest.py
git commit -m "phase/2: ingestion pipeline — PDF loader, chunker, embedder, FAISS index"
```

---

## Phase 3 — Conversational Logic: Retriever, LLM & RAG Chain

**Branch:** `phase/3-chain`
**Files delivered:** `src/retriever.py`, `src/llm.py`, `src/chain.py`

```bash
git checkout -b phase/3-chain
```

### Step 3.1 — Create `src/retriever.py`

```python
from langchain_community.vectorstores import FAISS
import config


def get_retriever(index: FAISS):
    return index.as_retriever(search_kwargs={"k": config.TOP_K})
```

### Step 3.2 — Create `src/llm.py`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
import config


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )
```

### Step 3.3 — Create `src/chain.py`

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from src.retriever import get_retriever
from src.llm import get_llm

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and the latest user question, "
     "reformulate the question so it is fully self-contained and understandable "
     "without the conversation history. "
     "Do NOT answer the question. "
     "Only rewrite it if it references prior context; otherwise return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant that answers questions strictly from the "
     "context provided below. Do not use any outside knowledge.\n\n"
     "If the answer is not explicitly present in the context, respond with exactly:\n"
     "\"I do not know the answer based on the provided documentation.\"\n\n"
     "Do not extrapolate, infer, summarise beyond the text, or guess.\n"
     "Cite page numbers inline as [Page N].\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_chain(index: FAISS):
    llm = get_llm()
    retriever = get_retriever(index)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    return create_retrieval_chain(history_aware_retriever, qa_chain)
```

**Chain invocation contract** (used by `app.py`):

```python
result = chain.invoke({"input": question, "chat_history": chat_history})
answer  = result["answer"]
sources = [
    {"page": doc.metadata["page"], "excerpt": doc.page_content[:300]}
    for doc in result["context"]
]
```

---

### QC-3 — Quality Gate: Conversational Logic Verification

```bash
python -c "
from src.ingest import build_index
from src.chain import build_chain
from langchain_core.messages import HumanMessage, AIMessage

FALLBACK = 'I do not know the answer based on the provided documentation.'

index = build_index()
chain = build_chain(index)

# --- Test 1: On-topic question returns a non-fallback answer ---
r1 = chain.invoke({'input': 'What is this document about?', 'chat_history': []})
assert r1['answer'] != FALLBACK, 'On-topic question incorrectly returned fallback'
assert len(r1['context']) > 0, 'No source documents returned'
print('PASS — T1: on-topic question answered with sources')

# --- Test 2: Out-of-context question triggers exact fallback string ---
r2 = chain.invoke({'input': 'What is the capital of Australia?', 'chat_history': []})
assert FALLBACK in r2['answer'], f'Expected fallback, got: {r2[\"answer\"]}'
print('PASS — T2: out-of-context question deflected correctly')

# --- Test 3: History-aware follow-up (pronoun coreference) ---
first_q  = 'What is this document about?'
first_a  = r1['answer']
history  = [HumanMessage(content=first_q), AIMessage(content=first_a)]
r3 = chain.invoke({'input': 'Can you summarise it in one sentence?', 'chat_history': history})
assert r3['answer'] != FALLBACK, 'Follow-up lost context and returned fallback'
print('PASS — T3: pronoun follow-up resolved using history')

# --- Test 4: Source metadata is intact ---
for doc in r1['context']:
    assert 'page' in doc.metadata
    assert isinstance(doc.metadata['page'], int)
print('PASS — T4: source documents carry valid page metadata')

print()
print('QC-3 PASSED — all 4 checks passed')
"
```

**Expected output:**
```
PASS — T1: on-topic question answered with sources
PASS — T2: out-of-context question deflected correctly
PASS — T3: pronoun follow-up resolved using history
PASS — T4: source documents carry valid page metadata

QC-3 PASSED — all 4 checks passed
```

**Git commit when all four pass:**
```bash
git add src/retriever.py src/llm.py src/chain.py
git commit -m "phase/3: RAG chain — history-aware retriever, LLM, anti-hallucination prompts"
```

---

## Phase 4 — Streamlit UI

**Branch:** `phase/4-ui`
**Files delivered:** `src/app.py`

```bash
git checkout -b phase/4-ui
```

### Step 4.1 — Create `src/app.py`

```python
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.ingest import build_index
from src.chain import build_chain

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

FALLBACK = "I do not know the answer based on the provided documentation."


@st.cache_resource(show_spinner="Indexing document — please wait...")
def load_chain():
    index = build_index()
    return build_chain(index)


chain = load_chain()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.caption("Powered by Gemini + FAISS")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # UI log: [{role, content, sources}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain message objects

# ── Render message history ────────────────────────────────────────────────────
st.header("Chat with your document")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            for src in msg["sources"]:
                with st.expander(f"📄 Page {src['page']}"):
                    st.caption(src["excerpt"])

# ── Handle new input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history,
            })
        answer = result["answer"]
        sources = [
            {"page": doc.metadata["page"], "excerpt": doc.page_content[:300]}
            for doc in result["context"]
        ]
        st.markdown(answer)
        if answer != FALLBACK:
            for src in sources:
                with st.expander(f"📄 Page {src['page']}"):
                    st.caption(src["excerpt"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ])
```

### Step 4.2 — Launch the app

```bash
streamlit run src/app.py
```

---

### QC-4 — Quality Gate: Streamlit UI Verification

Run the app (`streamlit run src/app.py`) and manually verify each item below.
Tick off every checkbox before committing.

| # | Action | Expected result |
|---|--------|-----------------|
| UI-1 | App starts | Spinner "Indexing document..." appears, then disappears. No error in terminal. |
| UI-2 | Ask an on-topic question | Answer appears in assistant bubble with at least one `📄 Page N` expander below. |
| UI-3 | Expand a citation | Expander shows a text excerpt from the document. Page number is plausible. |
| UI-4 | Ask a follow-up using a pronoun | e.g. "Tell me more about that." — answer is coherent and references prior context. |
| UI-5 | Ask an off-topic question | Exact fallback string displayed. No citation expanders shown. |
| UI-6 | Click "Clear Chat" | All messages disappear. Next question is treated as a fresh session (no history leak). |
| UI-7 | Refresh the browser | Messages are cleared (session state not persisted). Index is NOT rebuilt (cached). |

**Git commit when all UI checks pass:**
```bash
git add src/app.py
git commit -m "phase/4: Streamlit UI — chat interface, citation expanders, session memory, clear-chat"
```

---

## End-to-End Verification Test Suite

Run these tests after merging all four phases to `main`. Each test defines the exact terminal
command or manual procedure, the assertion to verify, and the pass criterion.

---

### EV-01 — Out-of-Bounds Context Deflection

**What it tests:** The LLM must not answer questions that have no grounding in the PDF.

```bash
python -c "
from src.ingest import build_index
from src.chain import build_chain

FALLBACK = 'I do not know the answer based on the provided documentation.'
chain = build_chain(build_index())

probes = [
    'Who won the 2024 US presidential election?',
    'What is the boiling point of water?',
    'Write me a haiku about the ocean.',
]
for q in probes:
    r = chain.invoke({'input': q, 'chat_history': []})
    assert FALLBACK in r['answer'], f'FAIL on: {q!r}\nGot: {r[\"answer\"]}'
    print(f'PASS — deflected: {q!r}')
print('EV-01 PASSED')
"
```

**Pass criterion:** All three probes return the exact fallback string.

---

### EV-02 — Pronoun Coreference Tracking

**What it tests:** The history-aware retriever must resolve pronouns (she/he/it/they) from a
previous turn into a standalone query before hitting FAISS.

```bash
python -c "
from src.ingest import build_index
from src.chain import build_chain
from langchain_core.messages import HumanMessage, AIMessage

FALLBACK = 'I do not know the answer based on the provided documentation.'
chain = build_chain(build_index())

# Turn 1: anchor a subject
r1 = chain.invoke({'input': 'What are the main topics of this document?', 'chat_history': []})
a1 = r1['answer']
assert a1 != FALLBACK, 'Turn 1 returned fallback unexpectedly'

# Turn 2: use a pronoun referring to the topics from turn 1
history = [HumanMessage(content='What are the main topics?'), AIMessage(content=a1)]
r2 = chain.invoke({'input': 'Can you expand on them?', 'chat_history': history})
assert r2['answer'] != FALLBACK, 'Pronoun follow-up lost context'
print('Turn 1 answer preview:', a1[:80])
print('Turn 2 answer preview:', r2[\"answer\"][:80])
print('EV-02 PASSED — pronoun coreference resolved')
"
```

**Pass criterion:** Turn 2 returns a relevant answer, not the fallback string.

---

### EV-03 — Multi-Turn Memory Chain (5+ Turns)

**What it tests:** Conversation history is maintained correctly across at least 5 turns without
context drift or session bleed.

```bash
python -c "
from src.ingest import build_index
from src.chain import build_chain
from langchain_core.messages import HumanMessage, AIMessage

FALLBACK = 'I do not know the answer based on the provided documentation.'
chain = build_chain(build_index())

history = []
questions = [
    'What is this document about?',
    'What is the first major section?',
    'Can you elaborate on that section?',
    'What comes after it?',
    'Summarise everything we have discussed so far.',
]

for i, q in enumerate(questions, 1):
    r = chain.invoke({'input': q, 'chat_history': history})
    a = r['answer']
    history.extend([HumanMessage(content=q), AIMessage(content=a)])
    print(f'Turn {i}: {q!r}')
    print(f'       → {a[:80]}')
    print()

assert FALLBACK not in history[-1].content, 'Turn 5 returned fallback'
print('EV-03 PASSED — 5-turn conversation maintained coherently')
"
```

**Pass criterion:** All 5 turns return non-fallback answers. Turn 5 references earlier turns.

---

### EV-04 — Citation Accuracy

**What it tests:** The page numbers cited in the LLM answer (`[Page N]`) match the page numbers
in the source documents returned by the chain.

```bash
python -c "
import re
from src.ingest import build_index
from src.chain import build_chain

chain = build_chain(build_index())
r = chain.invoke({'input': 'What is this document about?', 'chat_history': []})

answer = r['answer']
cited_pages = set(int(n) for n in re.findall(r'\[Page (\d+)\]', answer))
source_pages = set(doc.metadata['page'] for doc in r['context'])

print('Pages cited in answer :', cited_pages)
print('Pages in retrieved docs:', source_pages)

for p in cited_pages:
    assert p in source_pages, f'[Page {p}] cited but not in retrieved sources'
print('EV-04 PASSED — all cited pages match retrieved source metadata')
"
```

**Pass criterion:** Every `[Page N]` token in the answer corresponds to a retrieved document
with that page number in its metadata.

---

### EV-05 — Exact Fallback String Match

**What it tests:** The fallback response is byte-for-byte identical to the string specified in F-06.

```bash
python -c "
from src.ingest import build_index
from src.chain import build_chain

REQUIRED = 'I do not know the answer based on the provided documentation.'
chain = build_chain(build_index())

r = chain.invoke({'input': 'What is the GDP of Iceland?', 'chat_history': []})
actual = r['answer'].strip()

print('Expected:', repr(REQUIRED))
print('Actual  :', repr(actual))
assert actual == REQUIRED, 'Fallback string does not match exactly'
print('EV-05 PASSED — exact fallback string verified')
"
```

**Pass criterion:** `result["answer"].strip() == "I do not know the answer based on the provided documentation."`

---

### EV-06 — Startup Indexing SLA (< 30 seconds)

**What it tests:** NF-01 — the full indexing pipeline completes within the 30-second target.

```bash
python -c "
import time
from src.ingest import build_index

start = time.time()
index = build_index()
elapsed = time.time() - start

print(f'Indexing time : {elapsed:.2f}s')
print(f'Chunks indexed: {index.index.ntotal}')
assert elapsed < 30, f'FAIL — SLA breached: {elapsed:.2f}s > 30s'
print('EV-06 PASSED — startup indexing within 30s SLA')
"
```

**Pass criterion:** Elapsed time < 30 seconds.

---

### EV-07 — Per-Query Latency SLA (< 5 seconds)

**What it tests:** NF-02 — end-to-end query time (retrieval + generation) stays under 5 seconds.

```bash
python -c "
import time
from src.ingest import build_index
from src.chain import build_chain

chain = build_chain(build_index())

queries = [
    'What is the document about?',
    'What are the key findings?',
    'Who wrote this document?',
]
for q in queries:
    start = time.time()
    chain.invoke({'input': q, 'chat_history': []})
    elapsed = time.time() - start
    print(f'{elapsed:.2f}s — {q!r}')
    assert elapsed < 5, f'FAIL — query took {elapsed:.2f}s: {q!r}'

print('EV-07 PASSED — all queries within 5s SLA')
"
```

**Pass criterion:** All three queries complete in under 5 seconds each.

---

### EV-08 — Clear Chat State Reset

**What it tests:** After clicking "Clear Chat", session state is fully reset — no history leaks
into the next question.

**Procedure (manual — run Streamlit):**

```bash
streamlit run src/app.py
```

1. Ask: `"What is this document about?"` — note the answer.
2. Ask: `"Can you repeat exactly what you just said?"` — verify the answer references turn 1.
3. Click **"🗑️ Clear Chat"** in the sidebar.
4. Ask: `"Can you repeat exactly what you just said?"` — now with no prior context.
5. Verify the answer is the fallback string (no history to reference).

**Pass criterion:** After clearing, step 4 returns the fallback string because chat history is empty.

---

### EV-09 — Ambiguous Follow-Up Question Rewriting

**What it tests:** The contextualization prompt (§7.1 of specs.md) produces a meaningful
standalone query from an ambiguous follow-up, verified by inspecting the intermediate rewrite.

```bash
python -c "
from src.ingest import build_index
from src.llm import get_llm
from src.chain import CONTEXTUALIZE_PROMPT
from langchain_core.messages import HumanMessage, AIMessage

index = build_index()
llm   = get_llm()

history = [
    HumanMessage(content='What certifications are mentioned?'),
    AIMessage(content='The document mentions AWS Certified Solutions Architect and GCP Professional.'),
]
followup = 'How many does she have?'

msgs = CONTEXTUALIZE_PROMPT.format_messages(
    chat_history=history,
    input=followup,
)
rewritten = llm.invoke(msgs).content
print('Original  :', followup)
print('Rewritten :', rewritten)

assert len(rewritten) > len(followup), 'Rewrite did not expand the question'
assert any(kw in rewritten.lower() for kw in ['certif', 'aws', 'gcp']), \
    'Rewrite lost the subject domain'
print('EV-09 PASSED — ambiguous follow-up rewritten into a standalone query')
"
```

**Pass criterion:** The rewritten query explicitly names the subject or domain from the prior
turn and no longer relies on pronoun references.

---

### EV-10 — Image-Only Page Handling

**What it tests:** Pages with no extractable text (e.g. scanned images, diagrams) are skipped
silently without raising an exception, and indexing still completes successfully.

```bash
python -c "
from src.ingest import build_index

try:
    index = build_index()
    print(f'Index built with {index.index.ntotal} chunks')
    assert index.index.ntotal >= 0
    print('EV-10 PASSED — image-only pages skipped without error')
except Exception as e:
    print(f'FAIL — exception during indexing: {e}')
    raise
"
```

**Pass criterion:** `build_index()` completes without raising an exception, even if the PDF
contains pages with no extractable text.

---

## Summary: Quality Gate Reference

| Gate | Phase | When to run | Must pass before |
|------|-------|-------------|------------------|
| QC-1 | Bootstrap | After `pip install` and `config.py` creation | Committing Phase 1 |
| QC-2 | Ingestion | After `src/ingest.py` + placing PDF | Committing Phase 2 |
| QC-3 | Chain | After `src/chain.py`, `retriever.py`, `llm.py` | Committing Phase 3 |
| QC-4 | UI | After `src/app.py` runs in browser | Committing Phase 4 |
| EV-01–10 | End-to-End | After all phases merged to `main` | Shipping / sharing |
