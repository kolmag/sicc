"""scripts/rate_limit.py — lightweight in-memory rate limiting for the API.

A dependency-free sliding-window limiter keyed by client IP, intended to guard
the LLM-backed endpoints (/chat, /chat/stream, /chat/portfolio) which each cost
a paid model call. Suitable for the single-instance demo deployment; a
multi-instance setup would want a shared store (Redis) instead.

Configure via env:
  SICC_RATE_LIMIT           max requests per window per client (default 20; 0 disables)
  SICC_RATE_WINDOW_SECONDS  window length in seconds (default 60)
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_RATE_LIMIT = int(os.environ.get("SICC_RATE_LIMIT", "20"))
_RATE_WINDOW = float(os.environ.get("SICC_RATE_WINDOW_SECONDS", "60"))

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def _client_key(request: Request) -> str:
    # Requests arrive via the Next.js proxy / Docker network, so honour the
    # first X-Forwarded-For hop when present; fall back to the socket peer.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 when a client exceeds the window."""
    if _RATE_LIMIT <= 0:
        return  # limiting disabled

    key = _client_key(request)
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW

    with _lock:
        bucket = _hits[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= _RATE_LIMIT:
            retry_after = int(bucket[0] + _RATE_WINDOW - now) + 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down and retry shortly.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        bucket.append(now)
