"""Daily corpus refresh inside the running API.

The corpus is "the ARXIV_MAX_PAPERS most recent papers for ARXIV_QUERY". Every REFRESH_HOURS it
is re-fetched from arXiv, re-indexed in a staging directory and swapped in while the API keeps
serving. Since the corpus is rebuilt from scratch each time, nothing has to persist across
restarts: a fresh container builds its index on startup.
"""
import json
import logging
import os
import shutil
import threading
import time

from fastapi import FastAPI

import config
from ingest import fetch
from rag.index import build_index
from rag.retriever import Retriever

logger = logging.getLogger("api.refresh")
CHECK_EVERY = 3600  # seconds between staleness checks


def paper_list(papers: list[dict]) -> list[dict]:
    """What /papers exposes: metadata only, no full text."""
    keys = ("arxiv_id", "title", "authors", "published", "category", "abs_url", "pdf_url")
    return [{**{k: p[k] for k in keys}, "has_full_text": bool(p.get("full_text"))} for p in papers]


def index_age_hours() -> float:
    path = config.INDEX_DIR / config.CHUNKS_FILE
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


def refresh(app: FastAPI) -> bool:
    """Fetch the latest papers, rebuild the index and swap it in. On failure (e.g. arXiv down)
    the current corpus keeps being served."""
    if fetch.main() != 0:
        return False
    papers = json.loads(config.PAPERS_PATH.read_text())
    staging = config.INDEX_DIR.with_name(config.INDEX_DIR.name + ".new")
    build_index(papers, staging)
    retriever = Retriever(staging)  # loads into memory, so the files can move afterwards
    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
    os.replace(staging, config.INDEX_DIR)
    # Plain attribute swaps: requests in flight keep the objects they already hold.
    app.state.retriever, app.state.papers = retriever, paper_list(papers)
    logger.info("corpus refreshed: %d papers", len(papers))
    return True


def loop(app: FastAPI, stop: threading.Event) -> None:
    while not stop.is_set():
        if app.state.retriever is None or index_age_hours() >= config.REFRESH_HOURS:
            try:
                refresh(app)
            except Exception:
                logger.exception("corpus refresh failed; keeping the current corpus")
        stop.wait(CHECK_EVERY)
