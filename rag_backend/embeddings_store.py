from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingsStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def build_index(self, text: str, chunk_size: int = 500, overlap: int = 100):
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        self.db = FAISS.from_texts(chunks, self.embedder)

    def retrieve(self, query: str, k: int = 3) -> str:
        docs = self.db.similarity_search(query, k=k)
        return "\n".join(d.page_content for d in docs)
