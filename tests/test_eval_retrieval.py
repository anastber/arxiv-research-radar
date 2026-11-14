import pytest

from eval import retrieval
from eval.retrieval import evaluate, ndcg_at, recall_at, reciprocal_rank


@pytest.mark.parametrize("ranked, relevant, k, expected", [
    (["A", "B", "C"], {"A"}, 1, 1.0),
    (["B", "A", "C"], {"A"}, 1, 0.0),
    (["B", "A", "C"], {"A", "C"}, 2, 0.5),
    (["B", "A", "C"], {"A", "C"}, 3, 1.0),
])
def test_recall_at(ranked, relevant, k, expected):
    assert recall_at(ranked, relevant, k) == expected


@pytest.mark.parametrize("ranked, relevant, expected", [
    (["A", "B"], {"A"}, 1.0),
    (["B", "C", "A"], {"A", "C"}, 0.5),   # first relevant hit counts
    (["B", "C"], {"A"}, 0.0),
])
def test_reciprocal_rank(ranked, relevant, expected):
    assert reciprocal_rank(ranked, relevant) == expected


def test_ndcg():
    assert ndcg_at(["A", "B"], {"A", "B"}, 10) == pytest.approx(1.0)
    assert ndcg_at(["X", "A"], {"A"}, 10) == pytest.approx(1 / 1.58496, abs=1e-4)  # 1 / log2(3)
    assert ndcg_at(["X", "A"], {"A"}, 1) == 0.0


class FakeRetriever:
    """Returns fixed (arxiv_id, score) hits per question, applying k and min_score like the real one."""

    def __init__(self, hits: dict[str, list[tuple[str, float]]], min_score: float = 0.3, top_k: int = 2):
        self.hits, self.min_score, self.top_k = hits, min_score, top_k

    def search(self, question, k=None, min_score=None, max_per_paper=None):
        k = self.top_k if k is None else k
        min_score = self.min_score if min_score is None else min_score
        return [{"arxiv_id": p, "score": s} for p, s in self.hits[question] if s >= min_score][:k]


def q(qid: str, kind: str, arxiv_ids: list[str]) -> dict:
    return {"id": qid, "kind": kind, "question": qid, "arxiv_ids": arxiv_ids, "status": "accepted"}


def test_evaluate_scores_ranking_and_production_views(monkeypatch):
    monkeypatch.setattr(retrieval, "KS", (1, 5))
    retriever = FakeRetriever({
        "q1": [("A", 0.8), ("B", 0.6)],                 # hit at rank 1
        "q2": [("X", 0.7), ("Y", 0.5), ("B", 0.4)],     # multi: B at rank 3, outside the top-2 context
        "q3": [("X", 0.2)],                             # in scope, but below the threshold
        "q4": [("X", 0.1)],                             # off-topic, correctly rejected
        "q5": [("A", 0.9)],                             # rejected by review: ignored
    })
    questions = [q("q1", "single", ["A"]), q("q2", "multi", ["B", "C"]), q("q3", "single", ["X"]),
                 q("q4", "off_topic", []), {**q("q5", "single", ["A"]), "status": "rejected"}]

    report = evaluate(retriever, questions)
    by_id = {r["id"]: r for r in report["questions"]}
    assert set(by_id) == {"q1", "q2", "q3", "q4"}
    assert by_id["q2"]["recall@1"] == 0.0 and by_id["q2"]["recall@5"] == 0.5
    assert by_id["q2"]["mrr"] == pytest.approx(1 / 3)
    assert by_id["q2"]["context_recall"] == 0.0
    assert by_id["q3"]["recall@1"] == 1.0 and by_id["q3"]["rejected"] and by_id["q3"]["context_recall"] == 0.0

    summary = report["summary"]
    assert summary["single"]["n"] == 2 and summary["single"]["recall@1"] == 1.0
    assert summary["single"]["rejected"] == 0.5
    assert summary["in_scope"]["n"] == 3
    assert summary["off_topic"] == {"n": 1, "rejected": 1.0, "median_top_score": pytest.approx(0.1)}
    assert "single" in retrieval.format_table(summary)
