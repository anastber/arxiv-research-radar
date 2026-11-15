import faiss
import numpy as np
import pytest

from rag.bm25 import BM25, reciprocal_rank_fusion
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
    r.mode, r.bm25, r.reranker = "dense", None, None
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


def test_missing_index_raises(tmp_path):
    with pytest.raises(IndexNotBuiltError):
        Retriever(tmp_path)


def test_hybrid_surfaces_a_keyword_match_that_dense_ranks_low(retriever):
    retriever.chunks[5]["text"] = "ColBERT late interaction"  # cosine 0.2: last for dense search
    retriever.mode, retriever.bm25 = "hybrid", BM25([c["text"] for c in retriever.chunks])
    hits = retriever.search("How does ColBERT work?", k=3, min_score=0.5, max_per_paper=1)
    assert [h["arxiv_id"] for h in hits] == ["D", "A", "B"]  # BM25 rank 1 + dense rank 6 beats dense alone
    assert hits[0]["score"] == pytest.approx(0.2, abs=1e-5)  # scores stay cosine similarities


def test_hybrid_still_rejects_off_topic_questions(retriever):
    retriever.mode, retriever.bm25 = "hybrid", BM25([c["text"] for c in retriever.chunks])
    assert retriever.search("chunk", min_score=0.99) == []


def test_bm25_prefers_rare_terms_and_rrf_rewards_agreement():
    bm25 = BM25(["the model", "the model retrieval", "the rare acronym"])
    assert bm25.top("rare model", 3)[0] == 2
    assert bm25.top("unknown", 3) == []
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 1]]) == [1, 3, 2]


def test_reranker_reorders_candidates(retriever, monkeypatch):
    class FakeReranker:
        def predict(self, pairs):  # prefers the chunk the dense search ranked last
            return np.array([1.0 if text.endswith("chunk 4") else 0.0 for _, text in pairs])

    monkeypatch.setattr("rag.retriever.embedding_text", lambda c: c["text"])
    retriever.reranker = FakeReranker()
    hits = retriever.search("q", k=2, min_score=0.0, max_per_paper=6)
    assert [h["text"] for h in hits] == ["chunk 4", "chunk 0"]
