from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Load PDF
loader = PyPDFLoader("data/sample_data.pdf")
documents = loader.load()

print(f"📄 Loaded {len(documents)} pages")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"✂️ Created {len(chunks)} chunks")

# Embeddings
embeddings = OllamaEmbeddings(
    model="all-minilm",
    base_url="http://localhost:11434"
)

# Create vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save
vectorstore.save_local("vectorstore/")

print("✅ Vectorstore created and saved")