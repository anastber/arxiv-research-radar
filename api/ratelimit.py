"""Per-IP rate limiting with slowapi (in-memory counters, single process)."""
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

import config


def client_ip(request: Request) -> str:
    """The real client IP.

    Behind N trusted proxies the socket peer is the proxy, so every user would look
    identical. X-Forwarded-For is "client, proxy1, ..." but a client can prepend fake
    entries; only the entries our own proxies appended (the rightmost ones) are
    trustworthy. So with N hops we take the Nth from the right.
    """
    hops = config.TRUSTED_PROXY_HOPS
    forwarded = request.headers.get("x-forwarded-for")
    if hops > 0 and forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=client_ip)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Friendly, structured 429 the Streamlit UI can show verbatim."""
    count, _, period = config.ASK_RATE_LIMIT.partition("/")  # "5/hour" -> "5", "hour"
    body = {
        "error": "rate_limit",
        "detail": f"You've reached the limit of {count} questions per {period} for this demo. "
                  "Please try again later.",
    }
    headers = {}
    try:  # seconds until this IP's window resets
        limit, args = request.state.view_rate_limit
        reset_at, _ = limiter.limiter.get_window_stats(limit, *args)
        retry_after = max(1, int(reset_at - time.time()))
        body["retry_after_seconds"] = retry_after
        headers["Retry-After"] = str(retry_after)
    except Exception:  # never let a cosmetic detail turn a 429 into a 500
        pass
    return JSONResponse(body, status_code=429, headers=headers)
