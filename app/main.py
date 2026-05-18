import os
# Must be set before importing FAISS/other native libs that may load OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask_question

app = FastAPI(
    title="RAG API",
    description="AI Document Assistant using Ollama + FAISS",
    version="1.0"
)


# Request schema --> Pydantic for schema validation
class QueryRequest(BaseModel):
    question: str


# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


# Ask endpoint
@app.post("/ask")
def ask(request: QueryRequest):

    response = ask_question(request.question)

    return response