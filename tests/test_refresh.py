import json
from types import SimpleNamespace

import config
from api import refresh

PAPER = {"arxiv_id": "2611.00001", "title": "New", "authors": ["A"], "published": "2026-11-15",
         "category": "cs.CL", "abs_url": "a", "pdf_url": "p", "full_text": "text"}


def setup(monkeypatch, tmp_path, fetch_ok=True):
    monkeypatch.setattr(config, "PAPERS_PATH", tmp_path / "papers.json")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "old.txt").write_text("old")

    def fake_fetch():
        if fetch_ok:
            config.PAPERS_PATH.write_text(json.dumps([PAPER]))
        return 0 if fetch_ok else 1

    monkeypatch.setattr(refresh.fetch, "main", fake_fetch)

    def fake_build(papers, out):
        out.mkdir()
        (out / "new.txt").write_text("")

    monkeypatch.setattr(refresh, "build_index", fake_build)
    monkeypatch.setattr(refresh, "Retriever", lambda index_dir: "new retriever")
    return SimpleNamespace(state=SimpleNamespace(retriever="old retriever", papers=["old"]))


def test_refresh_swaps_in_the_new_corpus(monkeypatch, tmp_path):
    app = setup(monkeypatch, tmp_path)
    assert refresh.refresh(app) is True
    assert app.state.retriever == "new retriever"
    assert [p["arxiv_id"] for p in app.state.papers] == ["2611.00001"] and "full_text" not in app.state.papers[0]
    assert [f.name for f in config.INDEX_DIR.iterdir()] == ["new.txt"]
    assert not (tmp_path / "index.new").exists()


def test_failed_fetch_keeps_serving_the_current_corpus(monkeypatch, tmp_path):
    app = setup(monkeypatch, tmp_path, fetch_ok=False)
    assert refresh.refresh(app) is False
    assert app.state.retriever == "old retriever" and app.state.papers == ["old"]
    assert (config.INDEX_DIR / "old.txt").exists()
