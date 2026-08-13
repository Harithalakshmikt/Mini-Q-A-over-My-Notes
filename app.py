import streamlit as st
import hashlib
from rag_backend.rag_service import RAGService

# --- Page config ---
st.set_page_config(page_title="Mini-RAG Notes Q&A", layout="centered")
st.title("Mini-RAG Notes Q&A")
st.write("Ask questions about your notes. Uses RAG (FAISS + Flan-T5).")


@st.cache_resource
def load_rag(notes_content: str, _content_hash: str):
    rag = RAGService()
    rag.load_knowledge_base(notes_content)
    return rag


# --- Read notes file every rerun (cheap), but only rebuild index if content changed ---
with open("data/notes.txt", "r", encoding="utf-8") as f:
    notes = f.read()

content_hash = hashlib.md5(notes.encode("utf-8")).hexdigest()
rag = load_rag(notes, content_hash)

st.caption(f"Knowledge base loaded ({len(notes)} characters)")

# --- User input ---
question = st.text_input("Ask a question about your notes:")

# --- Top-k slider ---
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)

if st.button("Get Answer") and question:
    with st.spinner("Retrieving & generating answer..."):
        context = rag.store.retrieve(question, k=top_k)

        st.subheader("Retrieved Chunks")
        if context:
            st.write(context)
        else:
            st.warning("No relevant chunks found in the knowledge base.")

        answer = rag.generator.generate(context, question)
        st.subheader("Answer")
        st.write(answer if answer else "Could not generate an answer.")