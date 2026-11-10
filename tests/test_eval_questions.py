import json
import random

import faiss
import numpy as np
import pytest

import config
from eval import questions
from rag.llm import AnswerError, Completion

PROSE = " ".join(["retrieval"] * 90)


def chunk(arxiv_id: str, text: str = PROSE, section: str = "2 Method") -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "section": section, "text": text}


@pytest.fixture
def eval_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "EVAL_QUESTIONS_PATH", tmp_path / "questions.jsonl")
    monkeypatch.setattr(questions, "CALL_INTERVAL", 0)
    return tmp_path


def write_index(directory, chunks: list[dict], vectors: np.ndarray) -> None:
    directory.mkdir()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    faiss.write_index(index, str(directory / config.FAISS_FILE))
    (directory / config.CHUNKS_FILE).write_text(json.dumps(chunks))


@pytest.mark.parametrize("text, expected", [
    (PROSE, True),
    (" ".join(["word"] * 40), False),                     # too short
    (" ".join(["0.81", "0.77", "word"] * 30), False),     # table extracted as text
])
def test_usable(text, expected):
    assert questions.usable(chunk("A", text)) is expected


def test_single_candidates_are_one_usable_chunk_per_paper_and_seeded():
    chunks = [chunk("A"), chunk("A"), chunk("B"), chunk("C", "too short")]
    first = list(questions.single_candidates(chunks, random.Random(0)))
    assert sorted(c["arxiv_id"] for c in first) == ["A", "B"]
    assert first == list(questions.single_candidates(chunks, random.Random(0)))


def test_pair_candidates_pick_related_chunks_from_different_papers():
    chunks = [chunk("A"), chunk("A"), chunk("B"), chunk("C"), chunk("D")]
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],   # same paper as row 0: never paired with it
        [0.8, 0.6, 0.0],   # related to A (sim 0.8)
        [0.0, 0.0, 1.0],   # unrelated to everything
        [0.0, 0.0, 1.0],   # identical to C (sim 1.0): too similar
    ])
    pairs = list(questions.pair_candidates(chunks, vectors, random.Random(0)))
    assert {frozenset((a["arxiv_id"], b["arxiv_id"])) for a, b in pairs} == {frozenset("AB")}


@pytest.mark.parametrize("text, expected", [
    ('{"question": "Q?", "answer": "A."}', {"question": "Q?", "answer": "A."}),
    ('```json\n{"skip": "table"}\n```', {"skip": "table"}),
    ('Here you go: ["a?", "b?"]', ["a?", "b?"]),
    ("not json", None),
    ('{"question": ', None),
])
def test_parse_json(text, expected):
    assert questions.parse_json(text) == expected


def test_questions_round_trip(eval_dir):
    assert questions.load_questions() == []
    qs = [{"id": "q001", "question": "Qu'est-ce que é?"}, {"id": "q002", "question": "B"}]
    questions.save_questions(qs)
    assert questions.load_questions() == qs


def test_ask_waits_out_rate_limits(monkeypatch):
    replies = iter([AnswerError("quota", "rate_limit"), Completion('{"question": "Q?"}', 0, 0, False)])

    def fake_complete(system, user):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(questions.llm, "complete", fake_complete)
    monkeypatch.setattr(questions, "RATE_LIMIT_WAIT", 0)
    assert questions.ask("user") == {"question": "Q?"}


def fake_llm(monkeypatch, skip_paper: str | None = None):
    def fake_complete(system, user):
        if "JSON list" in user:
            return Completion('["How do bees fly?", "How do compilers inline?"]', 0, 0, False)
        if skip_paper and f"Paper {skip_paper}" in user:
            return Completion('{"skip": "no content"}', 0, 0, False)
        return Completion(json.dumps({"question": f"Q{len(user)}?", "answer": "A."}), 0, 0, False)

    monkeypatch.setattr(questions.llm, "complete", fake_complete)


def test_generate_drafts_all_kinds_then_tops_up_rejected(eval_dir, monkeypatch):
    chunks = [chunk("A"), chunk("B"), chunk("C")]
    write_index(config.EVAL_INDEX_DIR, chunks, np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]))
    monkeypatch.setattr(questions, "TARGETS", {"single": 2, "multi": 1, "off_topic": 2})
    fake_llm(monkeypatch)

    assert questions.generate() == 0
    qs = questions.load_questions()
    assert [q["kind"] for q in qs] == ["single", "single", "multi", "off_topic", "off_topic"]
    assert [q["id"] for q in qs] == ["q001", "q002", "q003", "q004", "q005"]
    assert all(q["status"] == "pending" for q in qs)
    assert len(qs[2]["arxiv_ids"]) == 2 and qs[2]["evidence"][0]["text"] == PROSE

    # Rejecting a single-paper question frees its slot; the rerun fills it from an unused paper.
    used_paper = qs[0]["arxiv_ids"][0]
    qs[0]["status"] = "rejected"
    questions.save_questions(qs)
    assert questions.generate() == 0
    qs = questions.load_questions()
    assert len(qs) == 6 and qs[-1]["id"] == "q006" and qs[-1]["kind"] == "single"
    assert qs[-1]["arxiv_ids"] != [used_paper]


def test_generate_skips_passages_the_llm_rejects(eval_dir, monkeypatch):
    chunks = [chunk("A"), chunk("B")]
    write_index(config.EVAL_INDEX_DIR, chunks, np.array([[1.0, 0.0], [0.0, 1.0]]))
    monkeypatch.setattr(questions, "TARGETS", {"single": 2, "multi": 0, "off_topic": 0})
    fake_llm(monkeypatch, skip_paper="A")

    assert questions.generate() == 0
    assert [q["arxiv_ids"] for q in questions.load_questions()] == [["B"]]


def test_generate_needs_the_benchmark_index(eval_dir):
    assert questions.generate() == 1


def test_review_records_decisions(eval_dir, monkeypatch):
    qs = [{"id": f"q00{i}", "kind": "single", "question": f"Q{i}", "answer": "A", "arxiv_ids": ["A"],
           "evidence": [], "status": "pending"} for i in (1, 2, 3)]
    questions.save_questions(qs)
    answers = iter(["x", "a", "e", "Better question?", "", "r"])  # "x" is invalid and re-prompted
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert questions.review() == 0
    qs = questions.load_questions()
    assert [q["status"] for q in qs] == ["accepted", "accepted", "rejected"]
    assert qs[1]["question"] == "Better question?" and qs[1]["answer"] == "A"


def test_review_keeps_questions_generated_while_it_runs(eval_dir, monkeypatch):
    pending = {"kind": "single", "question": "Q", "answer": "A", "arxiv_ids": ["A"], "evidence": [],
               "status": "pending"}
    questions.save_questions([{"id": "q001", **pending}])
    generated = []

    def accept_while_generate_appends(_):
        questions.add(generated or questions.load_questions(), dict(pending, question="New"))
        return "a"

    monkeypatch.setattr("builtins.input", accept_while_generate_appends)
    assert questions.review() == 0
    qs = questions.load_questions()
    assert [(q["id"], q["status"]) for q in qs] == [("q001", "accepted"), ("q002", "pending")]
