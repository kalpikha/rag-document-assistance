import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

embeddings = OllamaEmbeddings(
    model="all-minilm",
    base_url="http://localhost:11434"
)


def ingest_documents():

    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    print(os.listdir)

    # Loop through all PDFs
    for filename in os.listdir(UPLOAD_FOLDER):

        if filename.endswith(".pdf"):

            file_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            print(file_path)

            print(f"\n📄 Loading: {filename}")

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            # Add metadata
            for doc in documents:

                doc.metadata["source"] = filename

            chunks = splitter.split_documents(
                documents
            )

            all_chunks.extend(chunks)

    print(f"\n✂️ Total chunks: {len(all_chunks)}")


    # Create vectorstore
    vectorstore = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    vectorstore.save_local(str(VECTORSTORE_PATH))

    print("\n✅ Multi-document vectorstore created")

# ingest_documents()