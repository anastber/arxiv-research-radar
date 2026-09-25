"""Load the FAISS index and retrieve the top-k chunks for a question."""
import json

import faiss
from sentence_transformers import SentenceTransformer

import config


class IndexNotBuiltError(RuntimeError):
    pass


class Retriever:
    def __init__(self):
        if not (config.FAISS_PATH.exists() and config.CHUNKS_PATH.exists()):
            raise IndexNotBuiltError("Index not found. Run `python -m rag.index` first.")
        self.index = faiss.read_index(str(config.FAISS_PATH))
        self.chunks = json.loads(config.CHUNKS_PATH.read_text())
        self.model = SentenceTransformer(config.EMBED_MODEL)

    def search(self, question: str, k: int = 5, min_score: float = 0.0) -> list[dict]:
        """Top-k chunks by cosine similarity, each with a `score`. May return [] if
        nothing clears `min_score` - callers should treat that as "no relevant context"."""
        q = self.model.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(q, k)
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1 and s >= min_score
        ]
