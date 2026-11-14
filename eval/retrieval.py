"""Retrieval metrics on the golden question set, scored at the paper level.

    python -m eval.retrieval   # needs the benchmark index (python -m eval.corpus index)

Two views of the same retriever:
- Ranking: papers ordered by their best chunk, with no score threshold. Recall@k, MRR and
  nDCG measure how well relevant papers are ranked, independently of MIN_SCORE.
- Production: `Retriever.search()` with the live settings (TOP_K, MAX_CHUNKS_PER_PAPER,
  MIN_SCORE), i.e. exactly the context the LLM would see. Context recall is the share of
  relevant papers in that context; rejection rates show what the threshold costs and buys.
"""
import math
import statistics
import sys

import config
from eval.questions import load_questions
from rag.retriever import IndexNotBuiltError, Retriever

KS = (1, 5, 10)
RANK_DEPTH = max(KS)
# Papers requested for the ranking view. search() over-fetches k * 5 chunks before applying the
# per-paper cap, so asking for many more papers than RANK_DEPTH keeps the top of the list full.
RANK_FETCH = 100


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1 / (i + 1) for i, p in enumerate(ranked) if p in relevant), 0.0)


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def score_question(retriever: Retriever, q: dict) -> dict:
    ranking = retriever.search(q["question"], k=RANK_FETCH, min_score=-1.0, max_per_paper=1)
    context = retriever.search(q["question"])
    result = {
        "id": q["id"],
        "kind": q["kind"],
        "top_score": ranking[0]["score"] if ranking else None,
        "rejected": not context,
    }
    if q["kind"] == "off_topic":
        return result
    ranked, relevant = [h["arxiv_id"] for h in ranking[:RANK_DEPTH]], set(q["arxiv_ids"])
    return {
        **result,
        **{f"recall@{k}": recall_at(ranked, relevant, k) for k in KS},
        "mrr": reciprocal_rank(ranked, relevant),
        f"ndcg@{RANK_DEPTH}": ndcg_at(ranked, relevant, RANK_DEPTH),
        "context_recall": recall_at([h["arxiv_id"] for h in context], relevant, len(context)) if context else 0.0,
    }


def summarize(results: list[dict]) -> dict:
    """Mean of every metric per kind, plus all in-scope questions together."""
    groups = {kind: [r for r in results if r["kind"] == kind] for kind in ("single", "multi", "off_topic")}
    groups["in_scope"] = groups["single"] + groups["multi"]
    summary = {}
    for name, rows in groups.items():
        if not rows:
            continue
        metrics = [k for k in rows[0] if k not in ("id", "kind", "top_score", "rejected")]
        summary[name] = {
            "n": len(rows),
            **{m: statistics.fmean(r[m] for r in rows) for m in metrics},
            "rejected": statistics.fmean(r["rejected"] for r in rows),
            "median_top_score": statistics.median(r["top_score"] for r in rows if r["top_score"] is not None),
        }
    return summary


def evaluate(retriever: Retriever, questions: list[dict]) -> dict:
    results = [score_question(retriever, q) for q in questions if q["status"] == "accepted"]
    return {
        "config": {"embed_model": config.EMBED_MODEL, "top_k": config.TOP_K,
                   "max_chunks_per_paper": config.MAX_CHUNKS_PER_PAPER, "min_score": config.MIN_SCORE},
        "summary": summarize(results),
        "questions": results,
    }


def format_table(summary: dict) -> str:
    cols = [f"recall@{k}" for k in KS] + ["mrr", f"ndcg@{RANK_DEPTH}", "context_recall", "rejected",
                                         "median_top_score"]
    lines = [f"{'':10} {'n':>4}  " + "  ".join(f"{c:>16}" for c in cols)]
    for name, row in summary.items():
        cells = [f"{row[c]:16.3f}" if c in row else f"{'-':>16}" for c in cols]
        lines.append(f"{name:10} {row['n']:>4}  " + "  ".join(cells))
    return "\n".join(lines)


def main() -> int:
    questions = load_questions()
    if not questions:
        print("ERROR: no questions. Run `python -m eval.questions generate` first.", file=sys.stderr)
        return 1
    try:
        retriever = Retriever(config.EVAL_INDEX_DIR)
    except IndexNotBuiltError:
        print("ERROR: no benchmark index. Run `python -m eval.corpus index` first.", file=sys.stderr)
        return 1
    report = evaluate(retriever, questions)
    print(f"{report['config']}\n\n{format_table(report['summary'])}")
    print("\n`rejected`: share of questions for which nothing cleared MIN_SCORE "
          "(should be 1 for off_topic, 0 otherwise).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
