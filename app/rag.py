from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Embeddings
embeddings = OllamaEmbeddings(
    model="all-minilm",
    base_url="http://localhost:11434"
)

# Load vector DB
vectorstore = FAISS.load_local(
    "vectorstore/",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# LLM
llm = OllamaLLM(model="llama3.2:1b")


def build_prompt(context, question):
    return f"""
You are a factual document assistant.

Use ONLY the provided context to answer.

If the answer is not present in the context,
reply with: "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


def ask_question(query):
    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = build_prompt(context, query)

    # LLM response
    response = llm.invoke(prompt)

    return {
        "question": query,
        "answer": response,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:20]
            }
            for doc in docs
        ]
    }