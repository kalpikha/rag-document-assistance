import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    st.write("Current Session ID:")
    st.code(st.session_state.session_id)

    if st.button("🔄 New Chat"):

        st.session_state.messages = []

        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


# -----------------------------
# MAIN UI
# -----------------------------

st.title("🤖 AI Document Assistant")

st.caption(
    "Conversational RAG system using Ollama + FAISS + LangChain"
)

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources if available
        if message.get("sources"):

            with st.expander("📚 Sources"):

                for i, source in enumerate(message["sources"]):

                    st.markdown(
                        f"### Source {i+1}"
                    )

                    st.write(
                        f"📄 Page: {source['page']}"
                    )

                    st.write(
                        source["preview"]
                    )

                    st.divider()


# -----------------------------
# USER INPUT
# -----------------------------

user_input = st.chat_input(
    "Ask something about your document..."
)

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": user_input,
                        "session_id": st.session_state.session_id
                    }
                )

                data = response.json()

                answer = data.get(
                    "answer",
                    "No response generated."
                )

                sources = data.get(
                    "sources",
                    []
                )

                st.markdown(answer)

                # Display sources
                if sources:

                    with st.expander("📚 Sources"):

                        for i, source in enumerate(sources):

                            st.markdown(
                                f"### Source {i+1}"
                            )

                            st.write(
                                f"📄 Page: {source['page']}"
                            )

                            st.write(
                                source["preview"]
                            )

                            st.divider()

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                error_message = f"❌ Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )