from pathlib import Path
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_BASE = BASE_DIR / "vectorstore" / "sessions"
UPLOAD_DIR = BASE_DIR / "uploads"


embeddings = OllamaEmbeddings(
    model="all-minilm",
    base_url="http://localhost:11434"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def _session_vectorstore_path(session_id):
    return VECTORSTORE_BASE / session_id


def ingest_pdf(file_path, session_id):

    file_path = Path(file_path)
    vs_path = _session_vectorstore_path(session_id)

    print(f"\n📄 Processing: {file_path.name} for session {session_id}")

    loader = PyPDFLoader(str(file_path))

    documents = loader.load()

    # Add metadata
    for doc in documents:

        doc.metadata["source"] = file_path.name

    chunks = splitter.split_documents(documents)

    print(f"✂️ Created {len(chunks)} chunks")

    # Check if session vectorstore exists
    if vs_path.exists():

        print("\n📦 Loading existing session vectorstore...")

        vectorstore = FAISS.load_local(
            str(vs_path),
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Add new chunks
        vectorstore.add_documents(chunks)

    else:

        print("\n🆕 Creating new session vectorstore...")

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    # Save updated vectorstore
    vs_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vs_path))

    print("\n✅ Session vectorstore updated successfully")


def delete_session_vectorstore(session_id):
    """Delete the vectorstore for a given session."""
    vs_path = _session_vectorstore_path(session_id)
    if vs_path.exists():
        shutil.rmtree(vs_path)
        print(f"🗑️ Deleted vectorstore for session {session_id}")


def session_has_vectorstore(session_id):
    """Check if a session has an existing vectorstore."""
    return _session_vectorstore_path(session_id).exists()