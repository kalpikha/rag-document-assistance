# rag-document-assistance
RAG implementation for document assistance

# Connect Ollama
- ollama serve

# Run streamlit app
- streamlit run app/frontend.py

# Run server FASTAPI
- uvicorn app.main:app --reload


# Project flow
User Query
    ↓
Session Memory
    ↓
History-Aware Query Rewriting
    ↓
Retriever (FAISS)
    ↓
Relevant Chunks
    ↓
Prompt Construction
    ↓
LLM (Ollama)
    ↓
Grounded Response
    ↓
Store Conversation History

# CORE COMPONENTS

| Component               | Responsibility             |
| ----------------------- | -------------------------- |
| Streamlit               | Chat UI                    |
| FastAPI                 | Backend APIs               |
| Session Memory          | Conversation continuity    |
| History-Aware Retriever | Rewrites follow-up queries |
| FAISS                   | Semantic vector search     |
| Ollama Embeddings       | Embedding generation       |
| Ollama LLM              | Response generation        |
| Prompt Template         | Controlled prompting       |


🚀 YOUR SYSTEM NOW SUPPORTS

✅ Multi-document querying
✅ Conversational memory
✅ History-aware retrieval
✅ Semantic search
✅ Metadata-aware responses
✅ Session isolation
✅ Local AI inference
✅ API-based architecture