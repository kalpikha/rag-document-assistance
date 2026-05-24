# rag-document-assistance

Conversational RAG app for PDF document Q&A using local AI (Ollama).

Upload one or more PDFs and ask questions about them in natural language. Each session maintains its own FAISS vectorstore and conversation history. Queries are rewritten using prior context before retrieval, so follow-up questions resolve correctly. All inference runs locally via Ollama — no external API calls.

## Stack

- **Frontend** — Streamlit
- **Backend** — FastAPI
- **Embeddings** — `all-minilm` via Ollama
- **LLM** — `llama3.2:1b` via Ollama
- **Vector store** — FAISS (per-session)
- **Retrieval** — History-aware retriever (LangChain)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. Start Ollama
ollama serve

# 2. Pull required models (first time)
ollama pull llama3.2:1b
ollama pull all-minilm

# 3. Start backend
uvicorn app.main:app --reload

# 4. Start frontend (separate terminal)
streamlit run app/frontend.py
```

## How it works

1. Upload a PDF via the sidebar
2. PDF is chunked and indexed into a per-session FAISS vectorstore
3. Each query is rewritten using conversation history before retrieval
4. Relevant chunks are passed to the LLM for a grounded response
5. Conversation history is maintained per session