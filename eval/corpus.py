"""The frozen benchmark corpus: every paper matching QUERY submitted in a closed date
window, so the set never changes as new papers come out (unlike data/papers.json).

    python -m eval.corpus fetch   # arXiv -> data/eval/corpus.json.gz (slow, one-time; resumable,
                                  # failed downloads are retried on the next run)
    python -m eval.corpus index   # corpus -> data/eval/index/ (needed before running the eval)
"""
import gzip
import json
import sys
import time

import config
from ingest.arxiv_client import ArxivError, fetch_recent
from ingest.pdf_parser import extract_text
from rag.index import build_index

QUERY = 'all:"retrieval augmented generation" AND submittedDate:[202608010000 TO 202609302359]'
PAGE_SIZE = 100
SAVE_EVERY = 20  # PDFs between checkpoints, so an interrupted fetch resumes where it stopped


def load_corpus() -> list[dict]:
    if not config.EVAL_CORPUS_PATH.exists():
        return []
    with gzip.open(config.EVAL_CORPUS_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_corpus(papers: list[dict]) -> None:
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    papers = sorted(papers, key=lambda p: p["arxiv_id"])  # stable order -> readable diffs
    # Write then rename, so a crash mid-write never destroys the previous checkpoint.
    tmp = config.EVAL_CORPUS_PATH.with_suffix(".tmp")
    # mtime=0 keeps the archive byte-identical when the content hasn't changed.
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(papers, indent=1, ensure_ascii=False).encode())
    tmp.replace(config.EVAL_CORPUS_PATH)


def list_papers() -> list[dict]:
    papers, start = [], 0
    while True:
        page = fetch_recent(QUERY, PAGE_SIZE, start=start)
        papers += page
        print(f"  listed {len(papers)} papers")
        if len(page) < PAGE_SIZE:
            return papers
        start += PAGE_SIZE
        time.sleep(3)  # arXiv API etiquette


def fetch() -> int:
    try:
        listed = list_papers()
    except ArxivError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    done = {p["arxiv_id"]: p for p in load_corpus()}
    # Papers saved without text (failed download) are retried on the next run.
    todo = [p for p in listed if not done.get(p["arxiv_id"], {}).get("full_text")]
    print(f"{len(listed)} papers in the window, {len(listed) - len(todo)} already parsed, {len(todo)} to download")

    for i, paper in enumerate(todo, 1):
        if i > 1:
            time.sleep(3)
        paper["full_text"] = extract_text(paper["pdf_url"])
        done[paper["arxiv_id"]] = paper
        print(f"[{i}/{len(todo)}] {paper['arxiv_id']}  {'parsed' if paper['full_text'] else 'abstract only':13}"
              f"  {paper['title'][:60]}")
        if i % SAVE_EVERY == 0:
            save_corpus(list(done.values()))

    save_corpus(list(done.values()))
    with_text = sum(1 for p in done.values() if p["full_text"])
    print(f"Saved {len(done)} papers ({with_text} with full text) -> {config.EVAL_CORPUS_PATH}")
    return 0


def index() -> int:
    papers = load_corpus()
    if not papers:
        print("ERROR: no corpus. Run `python -m eval.corpus fetch` first.", file=sys.stderr)
        return 1
    build_index(papers, config.EVAL_INDEX_DIR)
    return 0


if __name__ == "__main__":
    commands = {"fetch": fetch, "index": index}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: python -m eval.corpus {fetch|index}")
    sys.exit(commands[sys.argv[1]]())
