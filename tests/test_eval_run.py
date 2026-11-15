import json

import pytest

import config
from eval import answers, calls, run
from rag.llm import AnswerError, Completion


class FakeRetriever:
    """Paper A is the only match for every in-scope question; off-topic questions match nothing."""

    def search(self, question, k=None, min_score=None, max_per_paper=None):
        if "bees" in question:
            return []
        return [{"arxiv_id": "A", "title": "Paper A", "abs_url": "u", "section": "1 Intro", "text": "a fact",
                 "score": 0.8}]


@pytest.fixture
def eval_env(monkeypatch, tmp_path):
    for name, path in [("EVAL_DIR", tmp_path), ("EVAL_ANSWERS_PATH", tmp_path / "answers.jsonl"),
                       ("EVAL_QUESTIONS_PATH", tmp_path / "questions.jsonl"),
                       ("EVAL_RESULTS_DIR", tmp_path / "results")]:
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(calls, "CALL_INTERVAL", 0)
    monkeypatch.setattr(run, "Retriever", lambda _: FakeRetriever())
    llm_calls = []

    def fake_complete(system, user, model=None, max_tokens=None):
        llm_calls.append(system)
        if system == answers.JUDGE_PROMPT:
            return Completion(json.dumps({"claims": [{"claim": "c", "cited": [1], "supported_by": [1]}],
                                          "declined": False, "relevance": 5}), 0, 0, False)
        return Completion("A fact [1].", 10, 5, False)

    monkeypatch.setattr("rag.llm.complete", fake_complete)
    qs = [{"id": "q001", "kind": "single", "question": "What fact?", "arxiv_ids": ["A"], "status": "accepted"},
          {"id": "q002", "kind": "off_topic", "question": "How do bees fly?", "arxiv_ids": [],
           "status": "accepted"}]
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    return llm_calls


def test_retrieval_only_run_makes_no_llm_calls(eval_env, tmp_path):
    assert run.main(["--retrieval-only", "--out", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert eval_env == []
    assert report["answers"] is None and report["questions"]["accepted"] == 2
    assert report["retrieval"]["summary"]["single"]["recall@1"] == 1.0
    assert report["retrieval"]["summary"]["off_topic"]["rejected"] == 1.0
    assert set(report["git"]) == {"commit", "dirty"} and len(report["questions"]["sha256"]) == 64


def test_full_run_writes_a_timestamped_report_with_both_sections(eval_env):
    assert run.main([]) == 0
    [path] = list(config.EVAL_RESULTS_DIR.glob("*.json"))
    report = json.loads(path.read_text())
    assert report["config"]["judge_model"] == config.EVAL_JUDGE_MODEL
    assert report["answers"]["summary"]["single"]["faithfulness"] == 1.0
    assert report["answers"]["summary"]["off_topic"]["declined"] == 1.0
    rows = {r["id"]: r for r in report["answers"]["questions"]}
    assert rows["q001"]["citation_precision"] == 1.0 and "answer" not in rows["q001"]
    assert rows["q002"]["retrieval_rejected"]


def test_llm_failure_reports_an_error(eval_env, monkeypatch):
    def failing(*args, **kwargs):
        raise AnswerError("down", "unavailable")

    monkeypatch.setattr("rag.llm.complete", failing)
    assert run.main([]) == 1
    assert not config.EVAL_RESULTS_DIR.exists()
