"""Access control for the HTTP surface.

The API can write API keys, spend The Odds API quota and rewrite bankroll
bookkeeping, so on a public VPS it cannot stay open. Every request is checked
against a single shared token (`API_TOKEN`) unless its path is public.

Design notes:

* The token comes from the environment only (never from `system_settings`), so a
  client holding the token cannot rotate it through the API.
* Comparison uses `hmac.compare_digest` — a plain `==` on a secret leaks its
  length and prefix through timing.
* With no `API_TOKEN` set the guard is *open* and says so loudly at startup.
  That keeps local development and the test suite frictionless; production is
  expected to set the variable (the deploy docs make it step one).
* Only the dashboard shell is public. `/docs` and `/openapi.json` are not: they
  describe every write endpoint.
"""

from __future__ import annotations

import hmac
import logging
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ai-bettor.security")

TOKEN_HEADER = "X-API-Token"

# Paths served without a token: the dashboard shell plus liveness. The shell is
# inert on its own — every panel it renders is filled by a guarded API call.
PUBLIC_PATHS = frozenset({
    "/",
    "/index.html",
    "/favicon.ico",
    "/health",
    "/auth/check",
})

# Static assets the shell pulls in. Kept as prefixes because the frontend mount
# may grow files; none of them expose data.
PUBLIC_PREFIXES = ("/static/", "/assets/")


def _extract_token(request: Request) -> str:
    """Read the token from `X-API-Token` or `Authorization: Bearer …`."""
    header = request.headers.get(TOKEN_HEADER)
    if header:
        return header.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def token_matches(candidate: str, expected: str) -> bool:
    """Constant-time token comparison. An empty expectation never matches."""
    if not expected:
        return False
    return hmac.compare_digest(str(candidate or ""), str(expected))


def is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that do not carry the shared token.

    `token` is read once at construction: the value is environment-only and a
    rotation means a restart, so re-reading it per request would only add cost.
    """

    def __init__(self, app, token: str, public_paths: Iterable[str] | None = None):
        super().__init__(app)
        self.token = (token or "").strip()
        self.extra_public = frozenset(public_paths or ())

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        # Browsers preflight cross-origin calls without credentials; the CORS
        # middleware answers those, so letting OPTIONS through is required.
        if request.method == "OPTIONS" or is_public_path(path) or path in self.extra_public:
            return await call_next(request)

        if token_matches(_extract_token(request), self.token):
            return await call_next(request)

        logger.warning("Rejected unauthenticated %s %s", request.method, path)
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Missing or invalid API token.",
                "hint": f"Send it as the {TOKEN_HEADER} header or as Authorization: Bearer <token>.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


def parse_origins(raw: str) -> list[str]:
    """Turn the `ALLOWED_ORIGINS` string into a CORS origin list.

    Comma separated, whitespace tolerated. `*` (or an empty value) means "no
    restriction" and is returned as `["*"]` so the caller can log the risk.
    """
    if not raw or raw.strip() == "*":
        return ["*"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def describe_auth(token: str) -> dict:
    """Auth state for `/health` — never the token itself."""
    token = (token or "").strip()
    return {
        "auth_required": bool(token),
        "token_header": TOKEN_HEADER,
        "token_length": len(token),
    }
