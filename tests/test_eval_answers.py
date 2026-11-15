import json

import pytest

import config
from eval import answers, calls
from eval.answers import answer_metrics, summarize, valid_judgment
from rag.answer import NO_CONTEXT_ANSWER
from rag.llm import Completion


def judgment(claims, declined=False, relevance=5) -> dict:
    return {"claims": [{"claim": "c", "cited": cited, "supported_by": sup} for cited, sup in claims],
            "declined": declined, "relevance": relevance}


def test_answer_metrics():
    m = answer_metrics(judgment([([1], [1]), ([1, 2], [2]), ([], [])], relevance=4))
    assert m["faithfulness"] == pytest.approx(2 / 3)       # last claim has no support
    assert m["citation_precision"] == pytest.approx(2 / 3)  # [1] for the 2nd claim is wrong
    assert m["relevance"] == 0.75 and m["declined"] is False


def test_answer_metrics_undefined_without_claims_or_citations():
    m = answer_metrics(judgment([], declined=True, relevance=1))
    assert m == {"faithfulness": None, "citation_precision": None, "relevance": 0.0, "declined": True}
    assert answer_metrics(judgment([([], [1])]))["citation_precision"] is None


@pytest.mark.parametrize("j, ok", [
    (judgment([([1], [1])]), True),
    (judgment([]), True),
    ({"claims": [], "declined": "no", "relevance": 5}, False),
    ({"claims": [], "declined": False, "relevance": 7}, False),
    ({"claims": [{"claim": "c", "cited": 1, "supported_by": []}], "declined": False, "relevance": 5}, False),
    (None, False),
])
def test_valid_judgment(j, ok):
    assert valid_judgment(j) is ok


def test_summary_skips_undefined_metrics_and_counts_judge_failures():
    records = [
        {"kind": "single", "judgment": judgment([([1], [1])])},
        {"kind": "single", "judgment": judgment([], declined=True, relevance=1)},
        {"kind": "multi", "judgment": None},
        {"kind": "off_topic", "judgment": answers.REJECTED_JUDGMENT},
    ]
    s = summarize(records)
    assert s["single"]["faithfulness"] == 1.0 and s["single"]["declined"] == 0.5
    assert s["multi"] == {"n": 1, "faithfulness": None, "citation_precision": None, "relevance": None,
                          "declined": None, "judge_failures": 1}
    assert s["in_scope"]["n"] == 3 and s["off_topic"]["declined"] == 1.0
    assert "judge_failures" in answers.format_table(s)


class FakeRetriever:
    def search(self, question):
        if "bees" in question:
            return []  # below the threshold
        return [{"arxiv_id": "A", "title": "Paper A", "abs_url": "u", "section": "1 Intro", "text": "a fact"}]


@pytest.fixture
def eval_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_ANSWERS_PATH", tmp_path / "answers.jsonl")
    monkeypatch.setattr(config, "EVAL_GRADES_PATH", tmp_path / "answer_grades.jsonl")
    monkeypatch.setattr(config, "EVAL_QUESTIONS_PATH", tmp_path / "questions.jsonl")
    monkeypatch.setattr(calls, "CALL_INTERVAL", 0)
    monkeypatch.setattr(answers, "Retriever", lambda _: FakeRetriever())
    llm_calls = []

    def fake_complete(system, user, model=None, max_tokens=None):
        llm_calls.append(model)
        if system == answers.JUDGE_PROMPT:
            return Completion(json.dumps(judgment([([1], [1])], relevance=4)), 0, 0, False)
        return Completion("A fact [1].", 10, 5, False)

    monkeypatch.setattr(calls.llm, "complete", fake_complete)
    monkeypatch.setattr("rag.llm.complete", fake_complete)
    qs = [{"id": "q001", "kind": "single", "question": "What fact?", "status": "accepted"},
          {"id": "q002", "kind": "off_topic", "question": "How do bees fly?", "status": "accepted"},
          {"id": "q003", "kind": "single", "question": "Rejected?", "status": "rejected"}]
    (tmp_path / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    return llm_calls


def test_run_answers_judges_and_resumes(eval_env):
    assert answers.run() == 0
    records = answers.load_jsonl(config.EVAL_ANSWERS_PATH)
    assert [r["id"] for r in records] == ["q001", "q002"]
    assert records[0]["answer"] == "A fact [1]." and records[0]["sources"][0]["chunks"][0]["text"] == "a fact"
    assert answer_metrics(records[0]["judgment"])["faithfulness"] == 1.0
    assert records[1]["answer"] == NO_CONTEXT_ANSWER and records[1]["judgment"] == answers.REJECTED_JUDGMENT
    assert eval_env == [None, config.EVAL_JUDGE_MODEL]  # answer + judge for q001; nothing for rejected q002

    eval_env.clear()
    assert answers.run() == 0
    assert eval_env == []  # same settings: nothing redone


def test_run_redoes_answers_when_settings_change(eval_env, monkeypatch):
    assert answers.run() == 0
    monkeypatch.setattr(config, "TOP_K", config.TOP_K + 1)
    eval_env.clear()
    assert answers.run() == 0
    assert len(eval_env) == 2 and len(answers.load_jsonl(config.EVAL_ANSWERS_PATH)) == 2


def test_grade_then_calibrate(eval_env, monkeypatch, capsys):
    assert answers.run() == 0
    replies = iter(["maybe", "n", "y", "4"])  # "maybe" is invalid and re-prompted
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert answers.grade() == 0
    grades = answers.load_jsonl(config.EVAL_GRADES_PATH)
    assert [g["id"] for g in grades] == ["q001"]  # off-topic and rejected answers are never sampled
    assert grades[0]["grader"] == "human"
    assert grades[0]["grade"] == {"faithful": False, "citations_correct": True, "relevance": 4}
    assert grades[0]["answer"] == "A fact [1]."

    assert answers.calibrate() == 0
    out = capsys.readouterr().out
    assert "(graders: human: 1)" in out
    assert "faithful (all claims supported): 0/1 agree (0%), judge more lenient on 1" in out
    assert "citations all correct:           1/1 agree" in out
    assert "mean abs. difference 0.00" in out
