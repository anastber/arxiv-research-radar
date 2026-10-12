import pytest

from rag import answer
from rag.answer import (
    EMPTY_ANSWER,
    NO_CONTEXT_ANSWER,
    answer_question,
    build_sources,
    build_user_message,
    extract_citations,
)
from rag.llm import Completion


def chunk(arxiv_id: str, text: str, score: float = 0.5) -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "section": "1 Introduction", "text": text, "score": score}


CHUNKS = [chunk("A", "a1", 0.9), chunk("B", "b1", 0.8), chunk("A", "a2", 0.7), chunk("C", "c1", 0.6)]


class FakeRetriever:
    def __init__(self, hits: list[dict]):
        self.hits = hits

    def search(self, question: str) -> list[dict]:
        return self.hits


def test_build_sources_groups_by_paper_in_best_first_order():
    sources = build_sources(CHUNKS)
    assert [(s["id"], s["arxiv_id"]) for s in sources] == [(1, "A"), (2, "B"), (3, "C")]
    assert [c["text"] for c in sources[0]["chunks"]] == ["a1", "a2"]


def test_user_message_wraps_sources_and_question():
    msg = build_user_message("What is X?", build_sources(CHUNKS))
    assert '<source id="1" title="Paper A">' in msg
    assert "[1 Introduction] a1" in msg
    assert msg.endswith("<question>What is X?</question>")


@pytest.mark.parametrize("text, expected_ids", [
    ("Claim [2]. Other claim [1][2].", [2, 1]),        # first-cited order, no duplicates
    ("Grouped citation [1, 3].", [1, 3]),
    ("Unknown source [7] is ignored [3].", [3]),
    ("No citations at all.", []),
])
def test_extract_citations(text, expected_ids):
    cited = extract_citations(text, build_sources(CHUNKS))
    assert [c["id"] for c in cited] == expected_ids


def test_extract_citations_resolves_paper_metadata():
    [cited] = extract_citations("Claim [2].", build_sources(CHUNKS))
    assert cited == {"id": 2, "arxiv_id": "B", "title": "Paper B", "url": "https://arxiv.org/abs/B"}


def test_no_relevant_chunks_skips_the_llm(monkeypatch):
    def fail(*args):
        raise AssertionError("LLM must not be called when nothing was retrieved")

    monkeypatch.setattr(answer.llm, "complete", fail)
    result = answer_question("off-topic?", FakeRetriever([]))
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["cited_papers"] == []
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_answer_returns_cited_papers_and_usage(monkeypatch):
    monkeypatch.setattr(answer.llm, "complete", lambda system, user: Completion("X does Y [2].", 120, 30, False))
    result = answer_question("What does X do?", FakeRetriever(CHUNKS))
    assert result["answer"] == "X does Y [2]."
    assert [p["arxiv_id"] for p in result["cited_papers"]] == ["B"]
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 30}
    assert result["truncated"] is False


def test_empty_llm_output_gets_a_fallback_answer(monkeypatch):
    monkeypatch.setattr(answer.llm, "complete", lambda system, user: Completion("", 120, 1000, True))
    result = answer_question("Broad question?", FakeRetriever(CHUNKS))
    assert result["answer"] == EMPTY_ANSWER
    assert result["truncated"] is True
