# RAG App — Project Workflow

## Overview

This is a **Retrieval-Augmented Generation (RAG)** application that lets users upload PDF documents and have a conversational Q&A session with an AI assistant grounded on the content of those documents. It uses **Ollama** (local LLM), **FAISS** (vector search), and **LangChain** for the RAG pipeline, with a **FastAPI** backend and a **Streamlit** frontend.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User (Browser)                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Streamlit Frontend  (frontend.py)               │
│  - Session management (UUID per browser tab)                 │
│  - PDF upload form (sidebar)                                 │
│  - Chat interface (main area)                                │
└──────────────────────┬───────────────────────────────────────┘
                       │  HTTP (REST)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Backend  (main.py)                      │
│  POST /upload   →  ingest_pdf()                              │
│  POST /ask      →  ask_question()                            │
│  DELETE /session/{id}  →  delete_session_vectorstore()       │
└────────────┬─────────────────────────┬────────────────────────┘
             │                         │
             ▼                         ▼
┌────────────────────────┐   ┌─────────────────────────────────┐
│  Ingestion  (ingest.py)│   │  RAG Pipeline  (rag.py)         │
│  PyPDFLoader           │   │  History-aware retriever        │
│  Text chunking (1000/  │   │  FAISS similarity search (k=4)  │
│    200 overlap)        │   │  Context assembly (≤6000 chars)  │
│  OllamaEmbeddings      │   │  OllamaLLM (llama3.2:1b)        │
│  FAISS vectorstore     │   │  InMemoryChatMessageHistory      │
└────────────┬───────────┘   └─────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  Local Storage                                               │
│  uploads/          — raw uploaded PDF files                  │
│  vectorstore/sessions/<session_id>/   — FAISS index files    │
└──────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  Ollama  (localhost:11434)                                   │
│  Embedding model : all-minilm                                │
│  LLM             : llama3.2:1b                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### 1. Session Initialization (Frontend)

When a user opens the Streamlit app:
- A **unique session ID** (UUID4) is generated and stored in `st.session_state`.
- This session ID ties together the user's uploaded documents, FAISS vectorstore, and in-memory chat history for the duration of the session.

---

### 2. PDF Upload & Ingestion

**Frontend → `POST /upload` → `ingest_pdf()`**

1. User picks a PDF file in the sidebar uploader.
2. Streamlit sends the file as a multipart form-data request to `POST /upload?session_id=<uuid>`.
3. FastAPI saves the raw file to `uploads/`.
4. `ingest_pdf()` is called:
   - **Load**: `PyPDFLoader` reads all pages from the PDF.
   - **Metadata**: Each page document is tagged with `source = filename`.
   - **Split**: `RecursiveCharacterTextSplitter` breaks the text into chunks of **1000 characters** with **200-character overlap** to preserve context across boundaries.
   - **Embed**: Each chunk is converted to a vector using the `all-minilm` embedding model via Ollama.
   - **Store**: Vectors are saved into a **FAISS index** on disk at `vectorstore/sessions/<session_id>/`.
     - If an index for this session already exists, new chunks are **added** to the existing index (supports multiple PDFs per session).
     - If no index exists, a new one is created.
5. A success message is returned to the frontend.

---

### 3. Asking a Question

**Frontend → `POST /ask` → `ask_question()` → RAG pipeline**

1. User types a question in the chat input.
2. Streamlit sends `{ "question": "...", "session_id": "..." }` to `POST /ask`.
3. FastAPI passes the request to `ask_question()`, which invokes the RAG chain.

#### Inside the RAG Pipeline (`rag_pipeline`):

**a) Acknowledgement Filter**
- If the user input is a short social phrase (e.g. "thanks", "ok", "great"), the pipeline short-circuits and returns `"Glad that helped."` without touching the LLM or vectorstore.

**b) Chat History Retrieval**
- The last **4 messages** from `InMemoryChatMessageHistory` for this session are retrieved to provide conversational context.

**c) Question Contextualization (History-Aware Retriever)**
- A `create_history_aware_retriever` step rewrites the user's question using the chat history so it is self-contained.
  - Example: "What does it say about that?" → "What does the document say about the refund policy?"
- This rewritten question is used for the vector search.

**d) Document Retrieval**
- The FAISS vectorstore for this session is loaded from disk.
- The **top-4 most semantically similar chunks** (`k=4`) are retrieved.

**e) Context Assembly**
- Retrieved chunks are concatenated in order, capped at **6000 characters** total, to stay within the LLM's effective context window.

**f) LLM Answer Generation**
- The final prompt is assembled with:
  - System instructions (answer only from context)
  - Formatted chat history
  - Retrieved document context
  - The user's question
- `llama3.2:1b` via Ollama generates the answer.

**g) Response**
- Returned to the frontend as `{ "answer": "...", "sources": [...] }`, where sources include document name, page number, and a 300-character preview of each retrieved chunk.

---

### 4. Displaying the Response

- Streamlit appends both the user message and the AI answer to the chat history displayed in the main area.
- All previous messages in the session are re-rendered on each interaction.

---

### 5. Starting a New Chat

- Clicking **"New Chat"** in the sidebar:
  - Sends `DELETE /session/<session_id>` to the backend, which removes the FAISS index from disk.
  - Clears Streamlit session state (messages, uploaded files).
  - Generates a new session UUID — the user starts fresh.

---

## Key Design Decisions

| Decision | Detail |
|---|---|
| **Session isolation** | Each browser session gets its own FAISS vectorstore, so multiple users don't share document context. |
| **Persistent vectorstore** | FAISS index is saved to disk, allowing multiple PDFs to be added incrementally to the same session. |
| **History-aware retrieval** | Follow-up questions are rewritten before vector search, improving retrieval accuracy in multi-turn conversations. |
| **Context cap** | Retrieved chunks are capped at 6000 chars to prevent prompt overflow with the small 1b model. |
| **Local-only LLM** | Ollama runs entirely on-device — no API keys, no data sent to third parties. |

---

## Project File Map

```
rag-app/
├── app/
│   ├── main.py        # FastAPI app — routes: /upload, /ask, /session
│   ├── ingest.py      # PDF loading, chunking, embedding, FAISS storage
│   ├── rag.py         # RAG pipeline, LLM, chat history, prompt templates
│   ├── frontend.py    # Streamlit UI — chat interface + PDF uploader
│   └── test.py        # Manual test scripts
├── uploads/           # Uploaded PDF files (temporary storage)
├── vectorstore/
│   └── sessions/
│       └── <session_id>/   # Per-session FAISS index (index.faiss + index.pkl)
├── data/              # Optional local data directory
├── requirements.txt   # Python dependencies
└── flow.md            # This file
```

---

## Running the App

**1. Start Ollama (in a separate terminal):**
```bash
ollama serve
ollama pull llama3.2:1b
ollama pull all-minilm
```

**2. Start the FastAPI backend:**
```bash
uvicorn app.main:app --reload
```

**3. Start the Streamlit frontend:**
```bash
streamlit run app/frontend.py
```

Open `http://localhost:8501` in your browser.
