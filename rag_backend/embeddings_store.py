from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class EmbeddingsStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None

    def build_index(self, text: str, chunk_size: int = 300, overlap: int = 100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        self.db = FAISS.from_texts(
            chunks,
            self.embedder,
            distance_strategy=DistanceStrategy.COSINE
        )

    def retrieve(self, query: str, k: int = 3, score_threshold: float = 0.5) -> str:
        if self.db is None:
            return ""

        results = self.db.similarity_search_with_score(query, k=k)

        scored = []
        for doc, l2_distance in results:
            cosine_sim = 1 - (l2_distance ** 2) / 2
            print(f"L2: {l2_distance:.4f} -> COSINE SIM: {cosine_sim:.4f} | {doc.page_content[:60]}")
            scored.append((doc, cosine_sim))

        filtered = [doc for doc, sim in scored if sim >= score_threshold]

        # Fallback: if nothing clears the threshold, still use the single best match
        # rather than returning empty context and forcing an incorrect "I don't know"
        if not filtered and scored:
            best_doc, best_sim = max(scored, key=lambda x: x[1])
            print(f"No chunk passed threshold {score_threshold}, falling back to best match (sim={best_sim:.4f})")
            filtered = [best_doc]

        return "\n".join(d.page_content for d in filtered)