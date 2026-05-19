from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

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

#session-aware memory
def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]



MAX_CONTEXT_CHARS = 2000
ACKNOWLEDGEMENT_INPUTS = {
    "cool",
    "good",
    "good job",
    "great",
    "great!",
    "nice",
    "ok",
    "okay",
    "perfect",
    "sounds good",
    "super",
    "thanks",
    "thanks!",
    "thank you",
    "thank you!"
}


def _coerce_question_text(value):

    if isinstance(value, str):
        return value

    if isinstance(value, BaseMessage):
        return value.content if isinstance(value.content, str) else str(value.content)

    if isinstance(value, (list, tuple)):
        parts = [
            _coerce_question_text(item)
            for item in value
        ]

        return "\n".join(
            part for part in parts
            if part
        )

    return str(value)


def _extract_latest_query(value):

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if isinstance(item, BaseMessage):
                return _coerce_question_text(item)

        if value:
            return _coerce_question_text(value[-1])

    return _coerce_question_text(value)


def _is_acknowledgement(query):

    normalized_query = " ".join(query.lower().strip().split())

    return normalized_query in ACKNOWLEDGEMENT_INPUTS


def rag_pipeline(inputs):

    query = _extract_latest_query(inputs["question"])

    if _is_acknowledgement(query):
        return {
            "answer": "Glad that helped.",
            "sources": []
        }

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
        "answer": response,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:300]
            }
            for doc in docs
        ]
    }


chain = RunnableWithMessageHistory(
    
    RunnableLambda(rag_pipeline),

    get_session_history,

    input_messages_key="question",
    output_messages_key="answer"
)

def ask_question(query, session_id="default_user"):

    response = chain.invoke(

        {"question": query},

        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    return response
