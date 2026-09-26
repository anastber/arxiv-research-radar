"""SQLite-backed daily cap + query log. One file, no server.

Tables:
  daily_counts(day, count)   - /ask calls per UTC day, enforced by reserve_daily_slot()
  queries(...)               - one row per /ask attempt: question, token usage, outcome

Monitor with:  python -m api.usage
"""
import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_counts (day TEXT PRIMARY KEY, count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,          -- ok | no_context | daily_cap | error:<kind>
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cited_papers INTEGER NOT NULL DEFAULT 0
);
"""


def _connect() -> sqlite3.Connection:
    config.USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.USAGE_DB_PATH, timeout=10)
    conn.executescript(SCHEMA)
    return conn


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def hash_ip(ip: str) -> str:
    """Short stable id so abuse is visible in the log without storing raw IPs."""
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def reserve_daily_slot(cap: int) -> str | None:
    """Atomically take one of today's `cap` slots. Returns the day (needed for
    release_daily_slot) or None if the cap is reached. The single UPDATE ... WHERE
    count < cap makes this race-free even with concurrent requests."""
    day = _today()
    with closing(_connect()) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO daily_counts (day, count) VALUES (?, 0)", (day,))
        cur = conn.execute("UPDATE daily_counts SET count = count + 1 WHERE day = ? AND count < ?", (day, cap))
        return day if cur.rowcount == 1 else None


def release_daily_slot(day: str) -> None:
    """Give a slot back when the request failed before producing an answer."""
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE daily_counts SET count = MAX(count - 1, 0) WHERE day = ?", (day,))


def log_query(ip: str, question: str, status: str, usage: dict | None = None, cited: int = 0) -> None:
    usage = usage or {}
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO queries (ts, ip_hash, question, status, input_tokens, output_tokens, cited_papers)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), hash_ip(ip), question, status,
             usage.get("input_tokens", 0), usage.get("output_tokens", 0), cited),
        )


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * config.PRICE_INPUT_PER_MTOK + output_tokens * config.PRICE_OUTPUT_PER_MTOK) / 1e6


def report() -> None:
    with closing(_connect()) as conn:
        day = _today()
        used = conn.execute("SELECT count FROM daily_counts WHERE day = ?", (day,)).fetchone()
        print(f"{day} (UTC): {used[0] if used else 0} / {config.DAILY_CAP} daily calls used")

        for label, where in (("today", "substr(ts, 1, 10) = ?"), ("all time", "? IS NOT NULL")):
            arg = day if label == "today" else "x"
            n, tin, tout = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0)"
                f" FROM queries WHERE {where}", (arg,)).fetchone()
            print(f"{label:9}: {n} requests, {tin} in / {tout} out tokens, ~${_cost(tin, tout):.4f}")

        print("\nstatus breakdown (all time):")
        for status, n in conn.execute("SELECT status, COUNT(*) FROM queries GROUP BY status ORDER BY 2 DESC"):
            print(f"  {status:16} {n}")

        print("\ntop clients today (ip hash):")
        for h, n in conn.execute(
                "SELECT ip_hash, COUNT(*) FROM queries WHERE substr(ts, 1, 10) = ? GROUP BY ip_hash"
                " ORDER BY 2 DESC LIMIT 5", (day,)):
            print(f"  {h}  {n} requests")

        print("\nlast 5 questions:")
        for ts, status, q in conn.execute("SELECT ts, status, question FROM queries ORDER BY id DESC LIMIT 5"):
            print(f"  {ts}  {status:10} {q[:80]}")


if __name__ == "__main__":
    report()
