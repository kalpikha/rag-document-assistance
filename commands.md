# Run Streamlit app
-  streamlit run app/frontend.py

# Run server FASTAPI
- uvicorn app.main:app --reload
- Go to http://127.0.0.1:8000/docs


# 🔧 Free Open Source Embedding Options

You have multiple completely free options for embeddings:

## Option 1: Ollama Embeddings (Recommended)
Free Ollama Embedding Models:
- **all-minilm**: Lightweight, fast, good quality
- **nomic-embed-text**: High quality, optimized for embeddings  
- **mxbai-embed-large**: Large model, best quality

## Option 2: Local TF-IDF Embeddings
- Completely offline
- No model downloads needed
- Implemented in cell 16

## Option 3: Sentence Transformers (HuggingFace)
- Variety of free models
- Good quality embeddings
- May require SSL certificate fixes

**Current Setup**: Using Ollama with all-minilm model

OLLAMA SETUP GUIDE - Run these in terminal:

1. Start Ollama (in one terminal):
   ollama serve

2. Download a free embedding model (in another terminal):
   ollama pull all-minilm
   
   OR for better quality:
   ollama pull nomic-embed-text

3. Verify installation:
   ollama list

4. Test the model:
   ollama show all-minilm



