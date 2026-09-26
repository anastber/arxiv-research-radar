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

    def search(
        self,
        question: str,
        k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
        max_per_paper: int = config.MAX_CHUNKS_PER_PAPER,
    ) -> list[dict]:
        """Top-k chunks by cosine similarity (best first), each with a `score`.

        At most `max_per_paper` chunks come from any one paper so answers draw on
        several papers. Returns [] if nothing clears `min_score` - callers should
        treat that as "the indexed papers don't cover this"."""
        q = self.model.encode([question], normalize_embeddings=True)
        # Over-fetch so the per-paper cap still leaves k results.
        scores, ids = self.index.search(q, min(self.index.ntotal, k * 5))
        hits, per_paper = [], {}
        for s, i in zip(scores[0], ids[0]):
            if i == -1 or s < min_score:
                break  # results are sorted, so everything after is worse
            chunk = self.chunks[i]
            if per_paper.get(chunk["arxiv_id"], 0) >= max_per_paper:
                continue
            per_paper[chunk["arxiv_id"]] = per_paper.get(chunk["arxiv_id"], 0) + 1
            hits.append({**chunk, "score": float(s)})
            if len(hits) == k:
                break
        return hits
