import streamlit as st
from rag_backend.rag_service import RAGService

# --- Page config ---
st.set_page_config(page_title="Mini-RAG Notes Q&A", layout="centered")
st.title("Mini-RAG Notes Q&A")
st.write("Ask questions about your notes. Uses RAG (FAISS + Flan-T5).")

# --- Load knowledge base ---
@st.cache_resource
def load_rag():
    with open("data/notes.txt", "r", encoding="utf-8") as f:
        notes = f.read()
    rag = RAGService()
    rag.load_knowledge_base(notes)
    return rag

rag = load_rag()

# --- User input ---
question = st.text_input("Ask a question about your notes:")

# --- Top-k slider ---
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)

if st.button("Get Answer") and question:
    with st.spinner("Retrieving & generating answer..."):
        # Retrieve context from top-k chunks
        context = rag.store.retrieve(question, k=top_k)
        st.subheader("Retrieved Chunks")
        st.write(context)

        # Generate answer from context
        answer = rag.generator.generate(context, question)
        st.subheader("Answer")
        st.write(answer)
