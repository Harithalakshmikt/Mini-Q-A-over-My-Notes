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
```
Mini-Q-A-over-My-Notes/
├─ mini_rag/               # Backend modules
│  ├─ __init__.py
│  ├─ embeddings_store.py
│  ├─ flan_generator.py
│  └─ rag_service.py
├─ data/
│  └─ notes.txt            # Your knowledge base
├─ app.py                  # Streamlit UI
└─ requirements.txt        # Dependencies
```


---

## **Installation / Running Locally**

1. Clone the repo:

```bash
git clone https://github.com/Harithalakshmikt/Mini-Q-A-over-My-Notes.git
cd Mini-Q-A-over-My-Notes
```

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

python -m streamlit run app.py

## **How It Works**

-  Load Notes → notes.txt is split into chunks and embedded using HuggingFace embeddings.

-  Store in FAISS → all chunks are stored in a vector database for similarity search.

- Ask a Question → user input is converted to embedding and matched with top-k relevant chunks.

- Generate Answer → Flan-T5 uses the retrieved chunks as context to generate an answer.

**Example**

- Notes content: Activist is being taken to Jodhpur. Internet services suspended, curfew continues in Leh.

- Question: “Where is the activist being taken?”

- Retrieved chunks: “Activist is being taken to Jodhpur.”

- Generated answer: “The activist is being taken to Jodhpur.”

## **Dependencies**

- Python ≥ 3.12

- Streamlit

- LangChain

- HuggingFace Transformers

- FAISS

- Flan-T5
  
- PyTorch

## **License**

MIT License


