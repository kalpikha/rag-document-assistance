from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from pathlib import Path

# Embeddings
embeddings = OllamaEmbeddings(
    model="all-minilm",
    base_url="http://localhost:11434"
)

# Load vector DB

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_BASE = BASE_DIR / "vectorstore" / "sessions"


def get_retriever(session_id):

    vs_path = VECTORSTORE_BASE / session_id

    if not vs_path.exists():
        return None

    vectorstore = FAISS.load_local(
        str(vs_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever

# LLM
llm = OllamaLLM(model="llama3.2:1b")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the chat history and latest user question,
rewrite the question so it can be understood
without chat history.

Do NOT answer the question.
Only rewrite it if needed.
"""
        ),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")
    ]
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a factual AI assistant.

Answer ONLY from the provided context below.
Use the chat history to understand follow-up questions.

Rules:
- Base your answer strictly on the context.
- If the context does not contain the answer, say:
  "I don't know based on the provided document."
- Be concise but complete.

Chat History:
{chat_history}

Context from documents:
{context}
"""
        ),

        (
            "human",
            "{question}"
        )
    ]
)

store = {}

#session-aware memory
def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]



MAX_CONTEXT_CHARS = 6000
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
    session_id = inputs.get("session_id", "default_user")

    if _is_acknowledgement(query):
        return {
            "answer": "Glad that helped.",
            "sources": []
        }

    history = get_session_history(session_id)

    chat_history = "\n".join(
        [
            f"{msg.type}: {msg.content}"
            for msg in history.messages[-4:]
        ]
    )

    retriever = get_retriever(session_id)

    if retriever is None:
        return {
            "answer": "No documents uploaded yet. Please upload a PDF first.",
            "sources": []
        }

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    docs = history_aware_retriever.invoke(
        {
            "input": query,
            "chat_history": history.messages
        }
    )

    # Build context from retrieved docs, keeping whole chunks
    context_parts = []
    total_len = 0
    for doc in docs:
        if total_len + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(doc.page_content)
        total_len += len(doc.page_content)

    context = "\n\n".join(context_parts)

    print(f"\n🔍 Query: {query}")
    print(f"📄 Retrieved {len(docs)} docs, using {len(context_parts)} in context ({len(context)} chars)")
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', '?')
        pg = doc.metadata.get('page', '?')
        print(f"   Doc {i+1}: {src} p.{pg} — {doc.page_content[:80]}...")

    prompt = prompt_template.format_messages(
        chat_history=chat_history,
        context=context,
        question=query
    )

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources": [
            {
                "document": doc.metadata.get("source"),
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

        {
            "question": query,
            "session_id": session_id
        },

        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    return response
