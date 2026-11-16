"""Per-visitor rate limiting with slowapi (in-memory counters, single process)."""
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import config


def client_ip(request: Request) -> str:
    """The visitor's IP. Requests reach the API through the Streamlit frontend, so the socket
    address is the frontend's; the frontend forwards the visitor's address in X-Visitor-IP.
    The header is trusted because the API is only reachable from the frontend (see Dockerfile)."""
    return request.headers.get("X-Visitor-IP") or get_remote_address(request)


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
