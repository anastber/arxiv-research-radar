"""Load the FAISS index and retrieve the top-k chunks for a question."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from rag.bm25 import BM25, reciprocal_rank_fusion
from rag.index import embedding_text


class IndexNotBuiltError(RuntimeError):
    pass


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR, mode: str = config.RETRIEVAL_MODE):
        faiss_path, chunks_path = index_dir / config.FAISS_FILE, index_dir / config.CHUNKS_FILE
        if not (faiss_path.exists() and chunks_path.exists()):
            raise IndexNotBuiltError(f"Index not found in {index_dir}. Build it first (see README).")
        self.index = faiss.read_index(str(faiss_path))
        self.chunks = json.loads(chunks_path.read_text())
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.mode = mode
        self.bm25 = BM25([embedding_text(c) for c in self.chunks]) if mode == "hybrid" else None
        self.reranker = CrossEncoder(config.RERANK_MODEL) if config.RERANK_MODEL else None

    def search(
        self,
        question: str,
        k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
        max_per_paper: int = config.MAX_CHUNKS_PER_PAPER,
    ) -> list[dict]:
        """Top-k chunks, best first, each with its cosine similarity as `score`.

        dense:  ranked by cosine similarity; chunks below `min_score` are dropped.
        hybrid: dense and BM25 rankings merged with Reciprocal Rank Fusion. `min_score` gates
                the question as a whole (on its best cosine), since a keyword match can
                legitimately have a low cosine.
        With RERANK_MODEL set, a cross-encoder then reorders the first RERANK_CANDIDATES.

        At most `max_per_paper` chunks come from any one paper so answers draw on
        several papers. Returns [] if nothing clears `min_score` - callers should
        treat that as "the indexed papers don't cover this"."""
        q = self.model.encode([config.EMBED_QUERY_PREFIX + question], normalize_embeddings=True)
        # Over-fetch so the per-paper cap still leaves k results.
        n = min(self.index.ntotal, k * 5)
        scores, ids = self.index.search(q, n)
        cosine = {int(i): float(s) for s, i in zip(scores[0], ids[0], strict=True) if i != -1}
        if not cosine or max(cosine.values()) < min_score:
            return []
        if self.mode == "hybrid":
            ranked = reciprocal_rank_fusion([list(cosine), self.bm25.top(question, n)])
            for i in ranked:
                if i not in cosine:  # BM25-only candidate: score it like the others
                    cosine[i] = float(np.dot(q[0], self.index.reconstruct(i)))
        else:
            ranked = [i for i in cosine if cosine[i] >= min_score]
        if self.reranker:
            head = ranked[:config.RERANK_CANDIDATES]
            relevance = self.reranker.predict([(question, embedding_text(self.chunks[i])) for i in head])
            ranked = [head[j] for j in np.argsort(-relevance, kind="stable")] + ranked[len(head):]

        hits, per_paper = [], {}
        for i in ranked:
            chunk = self.chunks[i]
            if per_paper.get(chunk["arxiv_id"], 0) >= max_per_paper:
                continue
            per_paper[chunk["arxiv_id"]] = per_paper.get(chunk["arxiv_id"], 0) + 1
            hits.append({**chunk, "score": cosine[i]})
            if len(hits) == k:
                break
        return hits
