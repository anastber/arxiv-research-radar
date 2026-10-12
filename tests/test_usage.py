import sqlite3

import pytest

import config
from api import usage


@pytest.fixture(autouse=True)
def usage_db(monkeypatch, tmp_path):
    path = tmp_path / "usage.db"
    monkeypatch.setattr(config, "USAGE_DB_PATH", path)
    return path


def test_daily_cap_is_enforced():
    assert all(usage.reserve_daily_slot(cap=3) for _ in range(3))
    assert usage.reserve_daily_slot(cap=3) is None


def test_released_slot_can_be_reused():
    days = [usage.reserve_daily_slot(cap=2) for _ in range(2)]
    assert usage.reserve_daily_slot(cap=2) is None
    usage.release_daily_slot(days[0])
    assert usage.reserve_daily_slot(cap=2) is not None


def test_query_log_stores_hashed_ip_only(usage_db):
    usage.log_query("203.0.113.7", "What is RAG?", "ok", {"input_tokens": 100, "output_tokens": 20}, cited=2)
    with sqlite3.connect(usage_db) as conn:
        row = conn.execute("SELECT ip_hash, question, status, input_tokens, output_tokens, cited_papers"
                           " FROM queries").fetchone()
    assert row == (usage.hash_ip("203.0.113.7"), "What is RAG?", "ok", 100, 20, 2)
    assert "203.0.113.7" not in row
