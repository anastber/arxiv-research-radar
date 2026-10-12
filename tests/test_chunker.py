import config
from rag.chunker import _pack_sentences, chunk_paper


def sentences(topic: str, n: int) -> str:
    """n six-word sentences, each one identifiable by its topic and number."""
    return " ".join(f"The {topic} result number {i} holds." for i in range(n))


PAPER_TEXT = "\n".join([
    "A Paper Title",
    "Jane Doe, John Smith",
    "1 Introduction",
    sentences("intro", 20),
    "2 Method",
    sentences("method", 3),  # too short on its own: merged into 2.1
    "2.1 Encoder Design",
    sentences("encoder", 20),
    "3 Experiments",
    sentences("experiment", 20),
    "12 Under review as a conference paper at ICLR 2026",  # running page header, not a heading
    sentences("experiment-continued", 20),
    "References",
    "[1] Some Author. A cited paper title. 2024.",
])


def make_paper(full_text: str | None = PAPER_TEXT) -> dict:
    return {
        "arxiv_id": "2601.00001",
        "title": "A Paper Title",
        "abs_url": "https://arxiv.org/abs/2601.00001",
        "abstract": "We study things.",
        "full_text": full_text,
    }


def test_abstract_is_always_the_first_chunk():
    chunks = chunk_paper(make_paper())
    assert chunks[0]["section"] == "Abstract"
    assert chunks[0]["text"] == "We study things."


def test_abstract_only_when_no_full_text():
    chunks = chunk_paper(make_paper(full_text=None))
    assert [c["section"] for c in chunks] == ["Abstract"]


def test_chunks_carry_paper_metadata():
    for c in chunk_paper(make_paper()):
        assert c["arxiv_id"] == "2601.00001"
        assert c["abs_url"] == "https://arxiv.org/abs/2601.00001"


def test_splits_on_numbered_headings_and_merges_short_sections():
    sections = {c["section"] for c in chunk_paper(make_paper())} - {"Abstract"}
    assert sections == {"1 Introduction", "2 Method", "3 Experiments"}

    method = " ".join(c["text"] for c in chunk_paper(make_paper()) if c["section"] == "2 Method")
    assert "method result" in method and "encoder result" in method


def test_running_headers_are_not_treated_as_headings():
    assert not any("Under review" in c["section"] for c in chunk_paper(make_paper()))


def test_front_matter_and_bibliography_are_dropped():
    text = " ".join(c["text"] for c in chunk_paper(make_paper()))
    assert "Jane Doe" not in text
    assert "cited paper title" not in text


def test_chunks_never_cross_section_boundaries():
    for c in chunk_paper(make_paper()):
        if c["section"] == "1 Introduction":
            assert "method result" not in c["text"]
        if c["section"] == "3 Experiments":
            assert "encoder result" not in c["text"]


def test_unnumbered_paper_falls_back_to_a_single_body_section():
    sections = {c["section"] for c in chunk_paper(make_paper(sentences("body", 60)))}
    assert sections == {"Abstract", "Body"}


def test_adjacent_chunks_overlap_by_one_sentence():
    chunks = _pack_sentences(sentences("long", 100))
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.rsplit(" The ", 1)[-1]
        assert nxt.startswith("The " + last_sentence.removeprefix("The "))


def test_text_without_sentence_breaks_is_hard_split():
    table_like = " ".join(f"cell{i}" for i in range(500))
    chunks = _pack_sentences(table_like)
    assert len(chunks) > 1
    # the last piece may absorb a small orphan tail, hence the margin
    assert all(len(c.split()) <= config.CHUNK_WORDS * 1.3 for c in chunks)
