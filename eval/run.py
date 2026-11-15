"""Full benchmark run: retrieval and answer metrics on the golden question set, printed as tables
and saved as one JSON report.

    python -m eval.run                   # retrieval + answers (needs GEMINI_API_KEY; reuses stored answers)
    python -m eval.run --retrieval-only  # no LLM calls, no API key needed
    python -m eval.run --out PATH        # report path (default: data/eval/results/<UTC timestamp>.json)

The report records what produced it (git commit, settings, question set) so that numbers quoted
in the README can be traced back to a run.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import config
from eval import answers, retrieval
from eval.questions import load_questions
from rag.llm import AnswerError
from rag.retriever import IndexNotBuiltError, Retriever


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}
    except OSError:  # git not installed
        return {"commit": None, "dirty": None}


def answer_rows(records: list[dict]) -> list[dict]:
    """Per-question answer metrics, without the answer and source texts."""
    return [{"id": r["id"], "kind": r["kind"], "retrieval_rejected": r["retrieval_rejected"],
             "truncated": r["truncated"], "judged": r["judgment"] is not None,
             **(answers.answer_metrics(r["judgment"]) if r["judgment"] else {})}
            for r in records]


def build_report(retriever: Retriever, questions: list[dict], with_answers: bool) -> dict:
    """Raises AnswerError if an LLM call fails (answers produced so far are kept for the next run)."""
    accepted = [q for q in questions if q["status"] == "accepted"]
    retrieval_report = retrieval.evaluate(retriever, questions)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "questions": {"sha256": hashlib.sha256(config.EVAL_QUESTIONS_PATH.read_bytes()).hexdigest(),
                      "accepted": len(accepted)},
        "config": answers.run_config() if with_answers else retrieval_report["config"],
        "retrieval": {k: retrieval_report[k] for k in ("summary", "questions")},
        "answers": None,
    }
    if with_answers:
        records = answers.answer_all(retriever, questions)
        report["answers"] = {"summary": answers.summarize(records), "questions": answer_rows(records)}
    return report


def default_out() -> Path:
    return config.EVAL_RESULTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer metrics (no LLM calls)")
    parser.add_argument("--out", type=Path, help="report path")
    args = parser.parse_args(argv)

    questions = load_questions()
    if not questions:
        print("ERROR: no questions. Run `python -m eval.questions generate` first.", file=sys.stderr)
        return 1
    try:
        retriever = Retriever(config.EVAL_INDEX_DIR)
    except IndexNotBuiltError:
        print("ERROR: no benchmark index. Run `python -m eval.corpus index` first.", file=sys.stderr)
        return 1
    try:
        report = build_report(retriever, questions, with_answers=not args.retrieval_only)
    except AnswerError as e:
        print(f"ERROR: {e} Answers produced so far are saved; rerun to continue.", file=sys.stderr)
        return 1

    out = args.out or default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(f"\n{report['config']}\n\nRetrieval\n{retrieval.format_table(report['retrieval']['summary'])}")
    if report["answers"]:
        print(f"\nAnswers\n{answers.format_table(report['answers']['summary'])}")
    print(f"\nReport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
