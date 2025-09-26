from rag_backend.rag_service import RAGService

if __name__ == "__main__":
    with open("data/notes.txt", "r", encoding="utf-8") as f:
        notes = f.read()

    rag = RAGService()
    rag.load_knowledge_base(notes)

    question = input("Ask a question about your notes: ")
    print("\n--- Answer ---\n", rag.answer(question))
