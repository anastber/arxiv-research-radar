"""Okapi BM25 over the chunk texts, plus Reciprocal Rank Fusion to merge it with dense search.

Dense embeddings match meaning but can miss exact terms (method names, acronyms, numbers);
BM25 matches terms exactly. Small enough to build in memory when the retriever loads.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        docs = [Counter(tokenize(t)) for t in texts]
        lengths = np.array([sum(d.values()) for d in docs], dtype=float)
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1.0))
        postings: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                ids, weights = postings[term]
                ids.append(i)
                weights.append(tf * (k1 + 1) / (tf + norm[i]))  # saturating, length-normalized tf
        self.n = len(docs)
        self.postings = {t: (np.array(ids), np.array(w)) for t, (ids, w) in postings.items()}
        self.idf = {t: math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
                    for t, (ids, _) in self.postings.items()}

    def top(self, query: str, n: int) -> list[int]:
        """Indices of the n best-scoring texts (best first); texts sharing no term are left out."""
        scores = np.zeros(self.n)
        for term in set(tokenize(query)):
            if term in self.postings:
                ids, weights = self.postings[term]
                scores[ids] += self.idf[term] * weights
        best = np.argsort(-scores, kind="stable")[:n]
        return [int(i) for i in best if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge rankings by summing 1 / (k + rank): items ranked well by several lists come first."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])
