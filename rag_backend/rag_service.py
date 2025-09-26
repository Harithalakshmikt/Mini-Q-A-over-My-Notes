from .embeddings_store import EmbeddingsStore
from .flan_generator import FlanGenerator

class RAGService:
    def __init__(self):
        self.store = EmbeddingsStore()
        self.generator = FlanGenerator()

    def load_knowledge_base(self, text: str):
        self.store.build_index(text)

    def answer(self, question: str) -> str:
        context = self.store.retrieve(question)
        return self.generator.generate(context, question)
