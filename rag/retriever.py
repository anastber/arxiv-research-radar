"""Load the FAISS index and retrieve the top-k chunks for a question."""
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

import config


class IndexNotBuiltError(RuntimeError):
    pass


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        faiss_path, chunks_path = index_dir / config.FAISS_FILE, index_dir / config.CHUNKS_FILE
        if not (faiss_path.exists() and chunks_path.exists()):
            raise IndexNotBuiltError(f"Index not found in {index_dir}. Build it first (see README).")
        self.index = faiss.read_index(str(faiss_path))
        self.chunks = json.loads(chunks_path.read_text())
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
        for s, i in zip(scores[0], ids[0], strict=True):
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
