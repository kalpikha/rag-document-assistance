import os
# Must be set before importing FAISS/other native libs that may load OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.rag import ask_question

app = FastAPI(
    title="AI Document Assistant",
    description="Conversational RAG API using Ollama + FAISS",
    version="2.0"
)

# -----------------------------------
# CORS CONFIG
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# REQUEST MODEL
# -----------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str


# -----------------------------------
# HEALTH CHECK
# -----------------------------------

@app.get("/")
def health_check():

    return {
        "status": "running",
        "message": "AI Document Assistant API is active"
    }


# -----------------------------------
# ASK ENDPOINT
# -----------------------------------

@app.post("/ask")
def ask(request: QueryRequest):

    response = ask_question(
        query=request.question,
        session_id=request.session_id
    )

    return response