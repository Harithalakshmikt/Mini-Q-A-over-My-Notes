from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import hashlib
import json
import os


class EmbeddingsStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "faiss_index"
    ):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None
        self.index_path = index_path

        # File used to remember which version of notes.txt
        # was used to create the FAISS index
        self.metadata_file = os.path.join(
            self.index_path,
            "metadata.json"
        )

    def _get_text_hash(self, text: str) -> str:
        """Create a hash of the knowledge-base text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _index_exists(self) -> bool:
        """Check whether a saved FAISS index exists."""
        index_file = os.path.join(
            self.index_path,
            "index.faiss"
        )

        metadata_file = os.path.join(
            self.index_path,
            "index.pkl"
        )

        return (
            os.path.exists(index_file)
            and os.path.exists(metadata_file)
            and os.path.exists(self.metadata_file)
        )

    def _is_index_up_to_date(self, text: str) -> bool:
        """Check whether the saved index was created from the current text."""

        if not os.path.exists(self.metadata_file):
            return False

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            current_hash = self._get_text_hash(text)

            return metadata.get("text_hash") == current_hash

        except (json.JSONDecodeError, OSError):
            return False

    def _load_index(self):
        """Load the existing FAISS index from disk."""

        self.db = FAISS.load_local(
            self.index_path,
            self.embedder,
            allow_dangerous_deserialization=True
        )

        print("Loaded existing FAISS index.")

    def _save_index(self, text: str):
        """Save FAISS index and metadata to disk."""

        os.makedirs(self.index_path, exist_ok=True)

        # Save FAISS files
        self.db.save_local(self.index_path)

        # Save hash of the source knowledge base
        metadata = {
            "text_hash": self._get_text_hash(text)
        }

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)

        print("FAISS index saved successfully.")

    def build_index(
        self,
        text: str,
        chunk_size: int = 300,
        overlap: int = 100
    ):
        """
        Load an existing FAISS index if the knowledge base
        has not changed. Otherwise build and save a new index.
        """

        # Reuse existing index if notes haven't changed
        if self._index_exists() and self._is_index_up_to_date(text):
            self._load_index()
            return

        print("Building new FAISS index...")

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

        # Save index for future runs
        self._save_index(text)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.5
    ) -> str:

        if self.db is None:
            return ""

        results = self.db.similarity_search_with_score(
            query,
            k=k
        )

        scored = []

        for doc, l2_distance in results:

            cosine_sim = 1 - (l2_distance ** 2) / 2

            print(
                f"L2: {l2_distance:.4f} "
                f"-> COSINE SIM: {cosine_sim:.4f} "
                f"| {doc.page_content[:60]}"
            )

            scored.append(
                (doc, cosine_sim)
            )

        filtered = [
            doc
            for doc, sim in scored
            if sim >= score_threshold
        ]

        # Fallback to best match
        if not filtered and scored:

            best_doc, best_sim = max(
                scored,
                key=lambda x: x[1]
            )

            print(
                f"No chunk passed threshold "
                f"{score_threshold}, "
                f"falling back to best match "
                f"(sim={best_sim:.4f})"
            )

            filtered = [best_doc]

        return "\n".join(
            doc.page_content
            for doc in filtered
        )