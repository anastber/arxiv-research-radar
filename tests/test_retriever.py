import faiss
import numpy as np
import pytest

import config
from rag.retriever import IndexNotBuiltError, Retriever

# Unit vectors at decreasing angles from the query direction [1, 0], so the cosine
# similarity of each chunk with the query is known in advance (best first).
SIMILARITIES = [0.95, 0.9, 0.85, 0.8, 0.5, 0.2]
PAPERS = ["A", "A", "A", "B", "C", "D"]


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype="float32")


@pytest.fixture
def retriever() -> Retriever:
    vectors = np.array([[s, np.sqrt(1 - s * s)] for s in SIMILARITIES], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    r = Retriever.__new__(Retriever)  # skip loading the real index and embedding model
    r.index = index
    r.chunks = [{"arxiv_id": p, "text": f"chunk {i}"} for i, p in enumerate(PAPERS)]
    r.model = FakeModel()
    return r


def test_results_are_best_first_with_scores(retriever):
    hits = retriever.search("q", k=6, min_score=0.0, max_per_paper=6)
    assert [h["text"] for h in hits] == [f"chunk {i}" for i in range(6)]
    assert hits[0]["score"] == pytest.approx(0.95, abs=1e-5)


def test_chunks_below_min_score_are_dropped(retriever):
    hits = retriever.search("q", k=6, min_score=0.6, max_per_paper=6)
    assert [h["text"] for h in hits] == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]


def test_nothing_above_min_score_returns_empty(retriever):
    assert retriever.search("q", k=6, min_score=0.99) == []


def test_per_paper_cap_spreads_results_across_papers(retriever):
    hits = retriever.search("q", k=4, min_score=0.0, max_per_paper=2)
    assert [h["arxiv_id"] for h in hits] == ["A", "A", "B", "C"]


def test_k_limits_the_number_of_results(retriever):
    assert len(retriever.search("q", k=2, min_score=0.0, max_per_paper=6)) == 2


def test_missing_index_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FAISS_PATH", tmp_path / "faiss.index")
    with pytest.raises(IndexNotBuiltError):
        Retriever()
