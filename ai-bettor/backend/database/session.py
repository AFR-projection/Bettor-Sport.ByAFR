"""Database session management for AI Bettor.

Production runs on Neon (serverless PostgreSQL); local development and the test
suite run on SQLite. `DATABASE_URL` decides which, and the URL is normalised
here so every shape Neon hands out in its dashboard works unchanged:

    postgres://user:pw@ep-x-pooler.eu-central-1.aws.neon.tech/db
    postgresql://user:pw@ep-x-pooler.eu-central-1.aws.neon.tech/db?sslmode=require
    postgresql+psycopg2://…

Two things matter for Neon specifically:

* **TLS is mandatory.** `sslmode=require` is appended when it is missing, so a
  copy-pasted URL cannot silently fail to connect.
* **Connections are cheap to make but get closed by the proxy.** The pool is
  kept small and recycled well inside Neon's idle window, with `pool_pre_ping`
  so a stale connection is replaced instead of raising mid-request.

The old behaviour of quietly falling back to SQLite when PostgreSQL was
unreachable is gone: in production that turns a connectivity problem into a
second, empty database that looks like data loss. It now raises unless
`ALLOW_SQLITE_FALLBACK=true` is set explicitly.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings
from backend.database.models import Base

logger = logging.getLogger("ai-bettor.database")

SQLITE_FALLBACK_URL = "sqlite:///./ai_bettor_dev.db"

# Hosts for which TLS is not forced — a local or in-container Postgres usually
# has no certificate, and requiring one would break `docker compose up`.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "db", "postgres", ""}

# Where Neon's *direct* (unpooled) endpoint shows up. `neon env pull` writes
# `DATABASE_URL_UNPOOLED`; `DIRECT_DATABASE_URL` is the name other tooling uses.
MIGRATION_URL_ENV_VARS = ("DATABASE_URL_UNPOOLED", "DIRECT_DATABASE_URL")


def _is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql") or url.startswith("postgres")


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")


def normalise_database_url(url: str) -> str:
    """Return a URL SQLAlchemy + psycopg2 can open, with Neon's TLS applied.

    * `postgres://` and `postgresql://` gain the explicit `+psycopg2` driver
      (`postgres://` alone is rejected outright by SQLAlchemy 2).
    * A remote Postgres host gains `sslmode=require` unless the URL already
      states an `sslmode`.
    * `channel_binding` is left exactly as Neon wrote it.
    """
    url = (url or "").strip()
    if not url or not _is_postgres_url(url):
        return url

    scheme, _, rest = url.partition("://")
    if "+" not in scheme:
        url = f"postgresql+psycopg2://{rest}"

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    host = (parts.hostname or "").lower()
    if "sslmode" not in query and host not in _LOCAL_HOSTS:
        query["sslmode"] = "require"
        parts = parts._replace(query=urlencode(query))
        url = urlunsplit(parts)
    return url


def is_pooled_url(url: str) -> bool:
    """True for Neon's PgBouncer endpoint — its host carries a `-pooler` suffix."""
    host = (urlsplit(normalise_database_url(url)).hostname or "").lower()
    return "-pooler." in host or host.endswith("-pooler")


def migration_database_url(env: Optional[Mapping[str, str]] = None) -> str:
    """The URL that DDL (alembic) should run over.

    Neon's pooled endpoint is PgBouncer in transaction mode: session state does
    not survive between statements, and migrations run over it fail in ways
    that never mention pooling — `prepared statement "s0" already exists`, a
    `SET search_path` that is gone by the next statement, a write landing in an
    inherited read-only transaction. Neon publishes the direct endpoint next to
    the pooled one, so use it when it is set and fall back to `DATABASE_URL`
    (which is correct as-is for SQLite and for a non-pooled Postgres host).
    """
    env = os.environ if env is None else env
    for name in MIGRATION_URL_ENV_VARS:
        direct = (env.get(name) or "").strip()
        if direct:
            return normalise_database_url(direct)

    fallback = normalise_database_url(
        (env.get("DATABASE_URL") or "").strip() or get_settings().DATABASE_URL)
    if is_pooled_url(fallback):
        logger.warning(
            "Migrations are pointed at Neon's pooled endpoint (%s). Set "
            "DATABASE_URL_UNPOOLED to the direct endpoint (same URL without "
            "'-pooler' in the host) if a migration fails for no obvious reason.",
            urlsplit(fallback).hostname)
    return fallback


def describe_database_url(url: str) -> dict:
    """A safe summary for `/health`: backend, host, database — never the password."""
    url = (url or "").strip()
    if _is_sqlite_url(url):
        return {"backend": "sqlite", "host": None, "database": url.split("///")[-1],
                "ssl": False, "provider": "local file"}
    try:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        host = parts.hostname or ""
        return {
            "backend": "postgres",
            "host": host,
            "database": (parts.path or "/").lstrip("/") or None,
            "ssl": query.get("sslmode") in {"require", "verify-ca", "verify-full"},
            "provider": "neon" if "neon.tech" in host else "postgres",
        }
    except Exception:  # pragma: no cover - a malformed URL should not break /health
        return {"backend": "unknown", "host": None, "database": None,
                "ssl": False, "provider": "unknown"}


def _engine_kwargs(url: str) -> dict:
    """Pool settings tuned per backend."""
    settings = get_settings()
    if _is_sqlite_url(url):
        # The scheduler thread and request handlers share one SQLite file.
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_pre_ping": True,
        "pool_recycle": settings.DB_POOL_RECYCLE_SECONDS,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_timeout": 30,
        "connect_args": {
            "connect_timeout": settings.DB_CONNECT_TIMEOUT_SECONDS,
            "application_name": "ai-bettor",
        },
    }


def _make_engine(database_url: str | None = None):
    settings = get_settings()
    url = normalise_database_url(database_url or settings.DATABASE_URL)
    if not url:
        url = SQLITE_FALLBACK_URL

    try:
        engine = create_engine(url, **_engine_kwargs(url))
        with engine.connect():
            pass
        info = describe_database_url(url)
        logger.info(
            "Connected to %s database (provider=%s host=%s ssl=%s)",
            info["backend"], info["provider"], info["host"] or "local", info["ssl"],
        )
        return engine
    except Exception as e:
        if not _is_postgres_url(url):
            raise
        if not settings.ALLOW_SQLITE_FALLBACK:
            # Loud failure on purpose: a hidden fallback would come up with an
            # empty database and read as "all my picks are gone".
            logger.error(
                "Cannot reach PostgreSQL at %s: %s. Check DATABASE_URL (Neon needs "
                "the pooled endpoint and sslmode=require). Set "
                "ALLOW_SQLITE_FALLBACK=true only for offline development.",
                describe_database_url(url)["host"], e,
            )
            raise RuntimeError(
                f"Database unreachable: {e}. Fix DATABASE_URL or set "
                "ALLOW_SQLITE_FALLBACK=true for local development."
            ) from e
        logger.warning(
            "PostgreSQL unavailable (%s). ALLOW_SQLITE_FALLBACK is on — using %s.",
            e, SQLITE_FALLBACK_URL,
        )
        fallback = SQLITE_FALLBACK_URL
        return create_engine(fallback, **_engine_kwargs(fallback))


engine = _make_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def database_info() -> dict:
    """What `/health` reports about the live connection."""
    info = describe_database_url(str(engine.url.render_as_string(hide_password=True)))
    info["dialect"] = engine.dialect.name
    return info


def init_db() -> None:
    """Create all tables if they don't exist, then add any missing columns."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


# Columns added to the schema after a database may already have been created.
# There is no migration framework here, and `create_all` never alters an
# existing table, so a plain ALTER keeps older SQLite/Postgres files usable
# instead of failing every query with "no such column".
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "predictions": {
        "pick_score": "FLOAT",
        "score_label": "VARCHAR(30)",
    },
}


def _add_missing_columns() -> None:
    try:
        inspector = inspect(engine)
    except Exception as e:  # pragma: no cover - inspection is best effort
        logger.warning("Schema inspection failed, skipping column sync: %s", e)
        return

    for table, columns in _ADDED_COLUMNS.items():
        try:
            if not inspector.has_table(table):
                continue
            existing = {col["name"] for col in inspector.get_columns(table)}
        except Exception as e:  # pragma: no cover
            logger.warning("Could not inspect table %s: %s", table, e)
            continue
        for name, ddl in columns.items():
            if name in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                logger.info("Schema updated: added %s.%s", table, name)
            except Exception as e:
                logger.warning("Could not add column %s.%s: %s", table, name, e)


def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for database sessions with commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency."""
    yield from get_session()