"""Fetch paper metadata from the arXiv API (Atom feed)."""
import re
import time
import xml.etree.ElementTree as ET

import requests

API_URL = "https://export.arxiv.org/api/query"
# arXiv asks API clients to identify themselves and to stay under ~1 request / 3s.
HEADERS = {"User-Agent": "arxiv-research-radar/0.1 (demo RAG app)"}
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivError(RuntimeError):
    """The arXiv API could not be reached or returned something unusable."""


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_entry(entry: ET.Element) -> dict:
    abs_url = _clean(entry.findtext("atom:id", namespaces=NS))
    # http://arxiv.org/abs/2401.12345v2 -> 2401.12345 (version-less, so updates dedupe)
    arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", pdf_url).replace("http://", "https://")

    primary = entry.find("arxiv:primary_category", NS)
    return {
        "arxiv_id": arxiv_id,
        "title": _clean(entry.findtext("atom:title", namespaces=NS)),
        "authors": [_clean(a.findtext("atom:name", namespaces=NS)) for a in entry.findall("atom:author", NS)],
        "abstract": _clean(entry.findtext("atom:summary", namespaces=NS)),
        "published": _clean(entry.findtext("atom:published", namespaces=NS)),
        "category": primary.get("term") if primary is not None else None,
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdf_url,
    }


def fetch_recent(query: str, max_results: int = 20, retries: int = 3) -> list[dict]:
    """Return the `max_results` most recently submitted papers matching `query`.

    Raises ArxivError if the API keeps failing, so callers never mistake an
    outage for "no papers found".
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            return [_parse_entry(e) for e in root.findall("atom:entry", NS)]
        except (requests.RequestException, ET.ParseError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(3 * attempt)  # also keeps us polite to the API
    raise ArxivError(f"arXiv API failed after {retries} attempts: {last_err}")
