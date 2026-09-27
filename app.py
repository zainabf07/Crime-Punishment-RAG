import streamlit as st
from rag import ask_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crime & Punishment RAG",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Crime & Punishment RAG")
st.write(
    "Ask questions about Dostoevsky's "
    "Crime and Punishment using a Retrieval-Augmented Generation system."
)


# ============================================================
# QUESTION
# ============================================================

question = st.text_input(
    "Enter your question:",
    placeholder="Why did Raskolnikov feel guilty?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the book and generating answer..."):

            try:

                answer, documents = ask_rag(question)

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.subheader("Answer")

                st.write(answer)


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                st.subheader("Retrieved Passages")

                for i, document in enumerate(
                    documents,
                    start=1
                ):

                    with st.expander(
                        f"Passage {i}"
                    ):

                        st.write(document)


            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )