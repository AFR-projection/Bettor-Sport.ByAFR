"""Tests for the database URL handling that Neon depends on.

Neon hands out URLs in several shapes and the app has to open all of them over
TLS. Getting this wrong fails at import time on the server, so the rules are
pinned here rather than discovered during a deploy:

* `postgres://` and `postgresql://` both become `postgresql+psycopg2://`
  (SQLAlchemy 2 refuses the bare `postgres://` scheme outright).
* a remote host gets `sslmode=require` appended; localhost does not.
* the password never appears in anything the app reports.
* DDL prefers the direct endpoint: Neon's pooled host is PgBouncer in
  transaction mode, and migrations over it fail without naming pooling.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest

from backend.database.session import (
    SQLITE_FALLBACK_URL, database_info, describe_database_url, engine,
    is_pooled_url, migration_database_url, normalise_database_url,
)

NEON = "postgresql://bettor:pw123@ep-cool-fog-12345-pooler.eu-central-1.aws.neon.tech/ai_bettor"
NEON_DIRECT = NEON.replace("-pooler", "")


class TestDriverNormalisation:
    def test_bare_postgres_scheme_gains_the_driver(self):
        """SQLAlchemy 2 raises on `postgres://`, and Neon still hands it out."""
        assert normalise_database_url(
            "postgres://u:p@ep-x-pooler.aws.neon.tech/db").startswith(
                "postgresql+psycopg2://")

    def test_postgresql_scheme_gains_the_driver(self):
        assert normalise_database_url(NEON).startswith("postgresql+psycopg2://")

    def test_an_explicit_driver_is_left_alone(self):
        url = normalise_database_url(
            "postgresql+psycopg2://u:p@ep-x-pooler.aws.neon.tech/db")
        assert url.count("+psycopg2") == 1

    def test_sqlite_is_untouched(self):
        assert normalise_database_url("sqlite:///./x.db") == "sqlite:///./x.db"

    def test_an_empty_url_stays_empty(self):
        assert normalise_database_url("") == ""
        assert normalise_database_url(None) == ""

    def test_surrounding_whitespace_is_stripped(self):
        assert normalise_database_url(f"  {NEON}  ").startswith("postgresql+psycopg2://")


class TestTls:
    def test_a_remote_host_gets_sslmode_require(self):
        assert "sslmode=require" in normalise_database_url(NEON)

    def test_an_existing_sslmode_is_respected(self):
        url = normalise_database_url(NEON + "?sslmode=verify-full")
        assert "sslmode=verify-full" in url
        assert "sslmode=require" not in url

    def test_sslmode_is_added_only_once(self):
        assert normalise_database_url(NEON + "?sslmode=require").count("sslmode") == 1

    def test_neons_other_query_args_survive(self):
        url = normalise_database_url(NEON + "?channel_binding=require")
        assert "channel_binding=require" in url and "sslmode=require" in url

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "db"])
    def test_a_local_host_is_not_forced_onto_tls(self, host):
        """A container-local Postgres has no certificate; requiring one breaks it."""
        assert "sslmode" not in normalise_database_url(
            f"postgresql://u:p@{host}:5432/ai_bettor")


class TestDescription:
    def test_a_neon_url_is_recognised(self):
        described = describe_database_url(normalise_database_url(NEON))
        assert described["backend"] == "postgres"
        assert described["provider"] == "neon"
        assert described["database"] == "ai_bettor"
        assert described["ssl"] is True

    def test_the_password_is_never_included(self):
        assert "pw123" not in str(describe_database_url(normalise_database_url(NEON)))

    def test_a_plain_postgres_host_is_not_labelled_neon(self):
        assert describe_database_url(
            "postgresql+psycopg2://u:p@10.0.0.5/db")["provider"] == "postgres"

    def test_sqlite_is_described_as_a_local_file(self):
        described = describe_database_url(SQLITE_FALLBACK_URL)
        assert described["backend"] == "sqlite"
        assert described["ssl"] is False

    def test_a_malformed_url_does_not_raise(self):
        """/health calls this; a bad URL must not take the endpoint down."""
        assert describe_database_url("::::")["backend"] in ("unknown", "postgres", "sqlite")


class TestLiveEngine:
    def test_the_test_suite_runs_on_sqlite(self):
        assert engine.dialect.name == "sqlite"

    def test_health_info_carries_the_dialect_and_no_password(self):
        info = database_info()
        assert info["dialect"] == "sqlite"
        assert "password" not in str(info).lower()


class TestPooledDetection:
    def test_a_pooler_host_is_recognised(self):
        assert is_pooled_url(NEON) is True

    def test_the_direct_endpoint_is_not(self):
        assert is_pooled_url(NEON_DIRECT) is False

    def test_sqlite_is_not_pooled(self):
        assert is_pooled_url(SQLITE_FALLBACK_URL) is False


class TestMigrationUrl:
    """Neon publishes two URLs; alembic must take the direct one when it exists."""

    def test_the_unpooled_url_wins_over_database_url(self):
        url = migration_database_url(
            {"DATABASE_URL": NEON, "DATABASE_URL_UNPOOLED": NEON_DIRECT})
        assert "-pooler" not in url
        assert url.startswith("postgresql+psycopg2://")

    def test_direct_database_url_is_also_accepted(self):
        assert "-pooler" not in migration_database_url(
            {"DATABASE_URL": NEON, "DIRECT_DATABASE_URL": NEON_DIRECT})

    def test_the_unpooled_name_takes_precedence(self):
        """`neon env pull` writes DATABASE_URL_UNPOOLED, so it is the first choice."""
        assert "ai_bettor" in migration_database_url({
            "DATABASE_URL_UNPOOLED": NEON_DIRECT,
            "DIRECT_DATABASE_URL": "postgresql://u:p@other.example.com/other",
        })

    def test_an_empty_unpooled_value_is_ignored(self):
        assert "-pooler" in migration_database_url(
            {"DATABASE_URL": NEON, "DATABASE_URL_UNPOOLED": "   "})

    def test_it_falls_back_to_database_url(self):
        assert migration_database_url({"DATABASE_URL": NEON_DIRECT}).startswith(
            "postgresql+psycopg2://")

    def test_a_pooled_fallback_is_used_but_warned_about(self, caplog):
        """Better a warning than a migration that fails without naming pooling."""
        with caplog.at_level("WARNING", logger="ai-bettor.database"):
            url = migration_database_url({"DATABASE_URL": NEON})
        assert "-pooler" in url
        assert "DATABASE_URL_UNPOOLED" in caplog.text

    def test_sqlite_needs_no_second_url(self):
        assert migration_database_url(
            {"DATABASE_URL": SQLITE_FALLBACK_URL}) == SQLITE_FALLBACK_URL

    def test_the_password_is_not_logged(self, caplog):
        with caplog.at_level("WARNING", logger="ai-bettor.database"):
            migration_database_url({"DATABASE_URL": NEON})
        assert "pw123" not in caplog.text
