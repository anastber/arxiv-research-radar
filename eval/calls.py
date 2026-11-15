"""LLM calls for offline evaluation jobs: paced for free-tier rate limits, with JSON parsing."""
import json
import re
import time

from rag import llm
from rag.llm import AnswerError

CALL_INTERVAL = 4       # seconds between LLM calls, to stay under free-tier rate limits
RATE_LIMIT_WAIT = 60

JSON_RE = re.compile(r"[\[{].*[\]}]", re.S)  # outermost JSON value, ignoring code fences or prose


def parse_json(text: str):
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def patiently(fn, *args, **kwargs):
    """Call `fn` (anything that calls the LLM), waiting out rate limits and pacing calls."""
    while True:
        try:
            result = fn(*args, **kwargs)
            time.sleep(CALL_INTERVAL)
            return result
        except AnswerError as e:
            if e.kind != "rate_limit":
                raise
            print(f"  rate limited, waiting {RATE_LIMIT_WAIT}s")
            time.sleep(RATE_LIMIT_WAIT)


def ask_json(system: str, user: str, **kwargs):
    """One LLM call parsed as JSON (None if unparseable)."""
    return parse_json(patiently(llm.complete, system, user, **kwargs).text)
