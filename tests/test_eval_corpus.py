import pytest

import config
from eval import corpus

PAPERS = [{"arxiv_id": "2608.00002", "title": "B", "full_text": "Ünïcode text"},
          {"arxiv_id": "2608.00001", "title": "A", "full_text": None}]


def test_corpus_round_trip_is_sorted_and_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_CORPUS_PATH", tmp_path / "corpus.json.gz")
    assert corpus.load_corpus() == []

    corpus.save_corpus(PAPERS)
    first = config.EVAL_CORPUS_PATH.read_bytes()
    assert [p["arxiv_id"] for p in corpus.load_corpus()] == ["2608.00001", "2608.00002"]
    assert corpus.load_corpus()[1]["full_text"] == "Ünïcode text"

    corpus.save_corpus(list(reversed(PAPERS)))
    assert config.EVAL_CORPUS_PATH.read_bytes() == first  # no spurious diffs on re-save


def test_failed_save_keeps_the_previous_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "EVAL_CORPUS_PATH", tmp_path / "corpus.json.gz")
    corpus.save_corpus(PAPERS)
    unencodable = [{"arxiv_id": "2608.00003", "full_text": "lone surrogate \ud835"}]
    with pytest.raises(UnicodeEncodeError):
        corpus.save_corpus(unencodable)
    assert len(corpus.load_corpus()) == 2
