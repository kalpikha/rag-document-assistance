from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory

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

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a factual AI assistant.

Answer ONLY from the provided context.

Rules:
- Do not hallucinate.
- If answer is missing, say:
  "I don't know based on the provided document."
"""
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

MAX_CONTEXT_CHARS = 2000


def ask_question(query):

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    context = context[:MAX_CONTEXT_CHARS]

    prompt = prompt_template.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)

    return {
        "question": query,
        "answer": response,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200]
            }
            for doc in docs
        ]
    }