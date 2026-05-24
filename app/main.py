import os
# Must be set before importing FAISS/other native libs that may load OpenMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import shutil
from pathlib import Path

try:
    from .ingest import ingest_pdf, delete_session_vectorstore
    from .rag import ask_question
except ImportError:
    from ingest import ingest_pdf, delete_session_vectorstore
    from rag import ask_question

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

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
# UPLOAD PDF
# -----------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = "default_user"
):

    # Validate PDF
    if not file.filename.endswith(".pdf"):

        return {
            "error": "Only PDF files allowed"
        }

    save_path = UPLOAD_DIR / file.filename

    # Save uploaded file
    with open(save_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Auto ingest into session vectorstore
    ingest_pdf(save_path, session_id)

    return {
        "message": f"{file.filename} uploaded and indexed successfully"
    }

# -----------------------------------
# CLEANUP SESSION
# -----------------------------------

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete the vectorstore for a session."""
    delete_session_vectorstore(session_id)
    return {
        "message": f"Session {session_id} vectorstore deleted"
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