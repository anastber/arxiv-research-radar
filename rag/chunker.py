"""Split a paper into retrieval-sized chunks, respecting section boundaries.

pypdf gives us flat text, but numbered headings ("2.1 Related Work") survive on
their own lines, so we split on those, then pack sentences into ~CHUNK_WORDS chunks
that never cross a section boundary.
"""
import re

import config

# "3 Methodology", "2.1 Related Work", "4. Experiments" - short, no trailing period.
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,70}?)$", re.M)
REFERENCES_RE = re.compile(r"^\s*(?:references|bibliography)\s*$", re.I | re.M)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
MIN_SECTION_WORDS = 80  # smaller sections get merged into the next one
MIN_HEADINGS = 3        # fewer numbered headings than this = probably not a numbered paper
MIN_COVERAGE = 0.6      # ...or if the sections cover less than this share of the text
RUNNING_HEADER_RE = re.compile(r"under review|preprint|published as|conference paper|proceedings", re.I)


def _looks_like_heading(match: re.Match) -> bool:
    title = match.group(2)
    return (
        len(title.split()) <= 10
        and not title.endswith((".", ",", ";"))
        and not RUNNING_HEADER_RE.search(title)  # page headers like "Under review as a ..."
    )


def _normalize(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line-wrapped words
    text = re.sub(r"\s*\n\s*", " ", text)     # PDF line breaks -> spaces
    return re.sub(r"\s+", " ", text).strip()


def _split_sections(full_text: str) -> list[tuple[str, str]]:
    """Return [(section_title, raw_text)], dropping front matter and references."""
    # Cut the bibliography (last "References" line in the back half of the paper).
    refs = [m.start() for m in REFERENCES_RE.finditer(full_text) if m.start() > len(full_text) * 0.4]
    if refs:
        full_text = full_text[: refs[-1]]

    heads = [m for m in HEADING_RE.finditer(full_text) if _looks_like_heading(m)]
    covered = len(full_text[heads[0].start():].split()) if heads else 0
    # Papers with unnumbered headings can produce a few false hits (e.g. a numbered
    # algorithm list); in that case don't trust the split and chunk the whole text.
    if len(heads) < MIN_HEADINGS or covered < MIN_COVERAGE * len(full_text.split()):
        return [("Body", full_text)]

    # Front matter (title/authors/abstract) is dropped: the abstract is indexed separately.
    sections = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(full_text)
        sections.append((f"{m.group(1)} {m.group(2).strip()}", full_text[m.end():end]))

    # Merge tiny sections (e.g. a bare "2 Related Work" followed by "2.1 ...") forward.
    merged, carry_title, carry_text = [], None, ""
    for title, text in sections:
        if carry_title is None:
            carry_title, carry_text = title, text
        else:
            carry_text += " " + title + ". " + text
        if len(carry_text.split()) >= MIN_SECTION_WORDS:
            merged.append((carry_title, carry_text))
            carry_title, carry_text = None, ""
    if carry_title is not None:
        if merged:  # trailing scrap: attach to the previous section
            prev_title, prev_text = merged[-1]
            merged[-1] = (prev_title, prev_text + " " + carry_title + ". " + carry_text)
        else:
            merged.append((carry_title, carry_text))
    return merged


def _pack_sentences(text: str) -> list[str]:
    sentences = []
    for sent in SENTENCE_RE.split(_normalize(text)):
        # Tables/equations extract as long runs with no sentence breaks: hard-split them.
        w = sent.split()
        sentences += [" ".join(w[i:i + config.CHUNK_WORDS]) for i in range(0, len(w), config.CHUNK_WORDS)]
    chunks, current, words = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if current and words + n > config.CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = current[-config.CHUNK_OVERLAP:] if config.CHUNK_OVERLAP else []
            words = sum(len(s.split()) for s in current)
        current.append(sent)
        words += n
    if current:
        tail = " ".join(current)
        # Avoid a tiny orphan chunk that is mostly overlap: fold it into the previous one.
        if chunks and words < config.CHUNK_WORDS * 0.3:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks


def chunk_paper(paper: dict) -> list[dict]:
    """Chunks for one paper. Falls back to abstract-only when there's no full text."""
    base = {"arxiv_id": paper["arxiv_id"], "title": paper["title"], "abs_url": paper["abs_url"]}
    chunks = [{**base, "section": "Abstract", "text": paper["abstract"]}]
    if paper.get("full_text"):
        for section, text in _split_sections(paper["full_text"]):
            for piece in _pack_sentences(text):
                chunks.append({**base, "section": section, "text": piece})
    return chunks
