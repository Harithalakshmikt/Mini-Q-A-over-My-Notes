# Mini-Q-A-over-My-Notes

A **simple but powerful Retrieval-Augmented Generation (RAG) demo** using **FAISS**, **Flan-T5**, and **LangChain**.  
This project allows users to **ask questions about any text/notes** and get **context-aware answers**, combining a vector database with a language model.

---

## **Demo**

Live Streamlit demo: [https://mini-q-a-over-my-notes-yonjq535fdrp4qgiirsoyo.streamlit.app](https://mini-q-a-over-my-notes-yonjq535fdrp4qgiirsoyo.streamlit.app)

---

## **Features**

- **RAG pipeline**: retrieves relevant chunks from your notes using FAISS and generates answers with Flan-T5.  
- **Interactive UI**: built with Streamlit.  
- **Top-k slider**: control how many chunks are retrieved for each question.  
- **Modular backend**: `RAGService` separates embeddings, retrieval, and generation.

---

## **Folder Structure**

