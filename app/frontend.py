import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Document Assistant")

st.write("Ask questions from your uploaded document.")

# User input
question = st.text_input(
    "Enter your question:"
)

# Ask button
if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            response = requests.post(
                API_URL,
                json={"question": question}
            )

            data = response.json()

            # Answer
            st.subheader("🧠 Answer")
            st.write(data["answer"])

            # Sources
            st.subheader("📚 Retrieved Sources")

            for i, source in enumerate(data["sources"]):

                st.markdown(f"### Source {i+1}")

                st.write(
                    source["content"]
                )

                st.divider()