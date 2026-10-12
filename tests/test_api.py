import json

import pytest
from fastapi.testclient import TestClient

import config
from api import main, usage
from api.ratelimit import limiter
from rag.llm import AnswerError

PAPER = {"arxiv_id": "2601.00001", "title": "A Paper", "authors": ["Jane Doe"], "published": "2026-01-01",
         "category": "cs.CL", "abs_url": "https://arxiv.org/abs/2601.00001",
         "pdf_url": "https://arxiv.org/pdf/2601.00001", "full_text": "..."}

ANSWER = {"answer": "X does Y [1].",
          "cited_papers": [{"id": 1, "arxiv_id": "2601.00001", "title": "A Paper",
                            "url": "https://arxiv.org/abs/2601.00001"}],
          "usage": {"input_tokens": 100, "output_tokens": 20}, "truncated": False}


@pytest.fixture
def client(monkeypatch, tmp_path):
    papers_path = tmp_path / "papers.json"
    papers_path.write_text(json.dumps([PAPER]))
    monkeypatch.setattr(config, "PAPERS_PATH", papers_path)
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setattr(main, "Retriever", lambda: object())  # no index or embedding model needed
    monkeypatch.setattr(main, "answer_question", lambda question, retriever: ANSWER)
    limiter.reset()
    with TestClient(main.app) as c:
        yield c


def test_health_and_papers(client):
    assert client.get("/health").json() == {"status": "ok", "papers": 1}
    [paper] = client.get("/papers").json()
    assert paper["has_full_text"] is True
    assert "full_text" not in paper


def test_ask_returns_answer(client):
    resp = client.post("/ask", json={"question": "What does X do?"})
    assert resp.status_code == 200
    assert resp.json() == {k: ANSWER[k] for k in ("answer", "cited_papers", "truncated")}


@pytest.mark.parametrize("question", ["", "hi", "x" * (config.MAX_QUESTION_CHARS + 1)])
def test_ask_rejects_invalid_question_length(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


@pytest.mark.parametrize("kind", main.ANSWER_ERRORS)
def test_llm_errors_map_to_safe_messages_and_release_the_slot(client, monkeypatch, kind):
    def fail(question, retriever):
        raise AnswerError("internal detail", kind)

    monkeypatch.setattr(main, "answer_question", fail)
    monkeypatch.setattr(config, "DAILY_CAP", 1)
    resp = client.post("/ask", json={"question": "What does X do?"})
    status, message = main.ANSWER_ERRORS[kind]
    assert resp.status_code == status
    assert resp.json() == {"detail": message}
    assert usage.reserve_daily_slot(cap=1) is not None  # the failed request gave its slot back


def test_daily_cap_returns_429(client, monkeypatch):
    monkeypatch.setattr(config, "DAILY_CAP", 1)
    assert client.post("/ask", json={"question": "First question?"}).status_code == 200
    resp = client.post("/ask", json={"question": "Second question?"})
    assert resp.status_code == 429
    assert resp.json()["error"] == "daily_cap"


def test_per_ip_rate_limit_returns_429(client, monkeypatch):
    limit = int(config.ASK_RATE_LIMIT.split("/")[0])
    monkeypatch.setattr(config, "DAILY_CAP", limit + 10)
    for _ in range(limit):
        assert client.post("/ask", json={"question": "Same question?"}).status_code == 200
    resp = client.post("/ask", json={"question": "Same question?"})
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limit"
    assert "Retry-After" in resp.headers
