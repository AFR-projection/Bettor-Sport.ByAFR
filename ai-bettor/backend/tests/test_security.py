"""Tests for the deployment hardening: API token guard and CORS.

The API can write API keys, spend The Odds API quota and rewrite bankroll
bookkeeping, so on a public host the guard is the only thing between the
internet and all of that. These pin both directions: with a token set nothing
unauthenticated gets through except the dashboard shell, and with no token set
the app stays wide open on purpose (localhost, tests) rather than half-broken.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.security import (
    PUBLIC_PATHS, TOKEN_HEADER, TokenAuthMiddleware, describe_auth,
    is_public_path, parse_origins, token_matches,
)

TOKEN = "s3cret-token-value"


def an_app(token: str, origins: str = "https://bettor.example.com") -> FastAPI:
    """A miniature app wired exactly like `backend.main` does it."""
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware, token=token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_origins(origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", TOKEN_HEADER],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/predictions")
    async def predictions():
        return []

    @app.put("/settings")
    async def settings():
        return {"updated": True}

    return app


@pytest.fixture
def guarded():
    with TestClient(an_app(TOKEN)) as client:
        yield client


@pytest.fixture
def unguarded():
    with TestClient(an_app("")) as client:
        yield client


class TestTokenComparison:
    def test_the_right_token_matches(self):
        assert token_matches(TOKEN, TOKEN) is True

    def test_a_wrong_token_does_not(self):
        assert token_matches("nope", TOKEN) is False
        assert token_matches(TOKEN + "x", TOKEN) is False

    def test_an_unset_expectation_never_matches(self):
        """Otherwise an empty API_TOKEN would accept an empty header as valid."""
        assert token_matches("", "") is False
        assert token_matches("anything", "") is False

    def test_none_is_handled_like_an_empty_string(self):
        assert token_matches(None, TOKEN) is False


class TestGuardedApi:
    def test_a_request_without_a_token_is_rejected(self, guarded):
        response = guarded.get("/predictions")
        assert response.status_code == 401
        assert "token" in response.json()["detail"].lower()

    def test_the_header_form_is_accepted(self, guarded):
        assert guarded.get("/predictions", headers={TOKEN_HEADER: TOKEN}).status_code == 200

    def test_the_bearer_form_is_accepted(self, guarded):
        assert guarded.get(
            "/predictions", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200

    def test_bearer_is_case_insensitive(self, guarded):
        assert guarded.get(
            "/predictions", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200

    def test_a_wrong_token_is_rejected(self, guarded):
        assert guarded.get(
            "/predictions", headers={TOKEN_HEADER: "wrong"}).status_code == 401

    def test_writes_are_guarded_too(self, guarded):
        """The dangerous half: /settings stores API keys."""
        assert guarded.put("/settings").status_code == 401
        assert guarded.put("/settings", headers={TOKEN_HEADER: TOKEN}).status_code == 200

    def test_health_stays_public(self, guarded):
        """The container healthcheck and uptime monitors have no token."""
        assert guarded.get("/health").status_code == 200

    def test_a_rejection_still_carries_cors_headers(self, guarded):
        """Without this the dashboard sees an opaque network error, not a 401."""
        response = guarded.get(
            "/predictions", headers={"Origin": "https://bettor.example.com"})
        assert response.status_code == 401
        assert response.headers["access-control-allow-origin"] == "https://bettor.example.com"

    def test_preflight_is_not_blocked(self, guarded):
        """Browsers send OPTIONS without the token header."""
        response = guarded.options("/predictions", headers={
            "Origin": "https://bettor.example.com",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code == 200


class TestOpenApi:
    def test_with_no_token_everything_is_open(self, unguarded):
        assert unguarded.get("/predictions").status_code == 200
        assert unguarded.put("/settings").status_code == 200

    def test_the_middleware_reports_whether_it_is_on(self):
        assert TokenAuthMiddleware(an_app(TOKEN), token=TOKEN).enabled is True
        assert TokenAuthMiddleware(an_app(""), token="  ").enabled is False


class TestPublicPaths:
    def test_the_dashboard_shell_is_public(self):
        assert is_public_path("/") and is_public_path("/index.html")

    def test_static_assets_are_public(self):
        assert is_public_path("/static/app.js")

    def test_data_endpoints_are_not(self):
        for path in ("/predictions", "/settings", "/bets", "/automation/trigger",
                     "/docs", "/openapi.json"):
            assert not is_public_path(path), f"{path} must not be public"

    def test_the_allowlist_stays_small(self):
        """A path added here bypasses auth, so the set is worth pinning."""
        assert PUBLIC_PATHS == {"/", "/index.html", "/favicon.ico", "/health", "/auth/check"}


class TestCorsOrigins:
    def test_a_single_origin(self):
        assert parse_origins("https://bettor.example.com") == ["https://bettor.example.com"]

    def test_several_origins_with_whitespace(self):
        assert parse_origins("https://a.com , https://b.com") == [
            "https://a.com", "https://b.com"]

    def test_a_trailing_slash_is_dropped(self):
        """CORS compares origins exactly; "https://a.com/" would never match."""
        assert parse_origins("https://a.com/") == ["https://a.com"]

    def test_empty_means_wildcard(self):
        assert parse_origins("") == ["*"]
        assert parse_origins("*") == ["*"]


class TestAuthDescription:
    def test_it_reports_the_state_without_the_secret(self):
        described = describe_auth(TOKEN)
        assert described["auth_required"] is True
        assert described["token_header"] == TOKEN_HEADER
        assert TOKEN not in str(described)

    def test_no_token_means_not_required(self):
        assert describe_auth("")["auth_required"] is False
