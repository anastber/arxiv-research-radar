"""Pull the most recent arXiv papers for the configured query into data/papers.json.

Usage:  python -m ingest.fetch

Re-running is cheap: papers already in papers.json keep their parsed text, so only
new papers trigger a PDF download. The corpus is always "the N most recent matches";
papers that fall out of the top N are dropped.
"""
import json
import sys
import time

import config
from ingest.arxiv_client import ArxivError, fetch_recent
from ingest.pdf_parser import extract_text


def load_existing() -> dict[str, dict]:
    if not config.PAPERS_PATH.exists():
        return {}
    try:
        return {p["arxiv_id"]: p for p in json.loads(config.PAPERS_PATH.read_text())}
    except (json.JSONDecodeError, KeyError):
        return {}


def main() -> int:
    print(f"Query: {config.ARXIV_QUERY}  (latest {config.ARXIV_MAX_PAPERS})")
    try:
        fresh = fetch_recent(config.ARXIV_QUERY, config.ARXIV_MAX_PAPERS)
    except ArxivError as e:
        # Leave the existing papers.json untouched so a flaky run can't wipe the corpus.
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not fresh:
        print("ERROR: query returned no papers; keeping existing data.", file=sys.stderr)
        return 1

    existing = load_existing()
    papers, downloaded = [], 0
    for i, paper in enumerate(fresh, 1):
        cached = existing.get(paper["arxiv_id"])
        if cached and cached.get("full_text"):
            paper["full_text"] = cached["full_text"]
            status = "cached"
        else:
            if downloaded:
                time.sleep(3)  # be polite to arxiv.org between PDF downloads
            paper["full_text"] = extract_text(paper["pdf_url"])
            downloaded += 1
            status = "parsed" if paper["full_text"] else "abstract only"
        print(f"[{i}/{len(fresh)}] {paper['arxiv_id']}  {status:13}  {paper['title'][:60]}")
        papers.append(paper)

    config.DATA_DIR.mkdir(exist_ok=True)
    config.PAPERS_PATH.write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    with_text = sum(1 for p in papers if p["full_text"])
    print(f"Saved {len(papers)} papers ({with_text} with full text) -> {config.PAPERS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
