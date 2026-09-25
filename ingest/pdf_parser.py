"""Download a paper PDF and extract its text with pypdf."""
import io

import requests
from pypdf import PdfReader

from ingest.arxiv_client import HEADERS

MAX_PDF_BYTES = 30 * 1024 * 1024  # skip pathological files


def extract_text(pdf_url: str) -> str | None:
    """Return the PDF's text, or None if it can't be fetched/parsed.

    None is an expected outcome (arXiv hiccups, scanned or malformed PDFs);
    callers fall back to the abstract for that paper.
    """
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        if len(resp.content) > MAX_PDF_BYTES:
            return None
        reader = PdfReader(io.BytesIO(resp.content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:  # pypdf raises a wide variety of errors on bad PDFs
        print(f"  ! PDF failed ({pdf_url}): {type(e).__name__}: {e}")
        return None

    text = "\n".join(pages).replace("\x00", "").strip()
    return text or None
