"""Database session management for AI Bettor.

Uses PostgreSQL by default (from DATABASE_URL).
Falls back to SQLite for local dev/tests when PostgreSQL is unavailable.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings
from backend.database.models import Base

logger = logging.getLogger("ai-bettor.database")


def _is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql") or url.startswith("postgres+")

def _make_engine(database_url: str | None = None):
    settings = get_settings()
    url = database_url or settings.DATABASE_URL
    if _is_postgres_url(url) and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if url == "postgresql://localhost:5432/ai_bettor":
        url = "postgresql+psycopg2://localhost:5432/ai_bettor"
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect():
            pass
        logger.info("Connected to database: %s", "postgres" if _is_postgres_url(url) else "sqlite")
        return engine
    except Exception as e:
        if _is_postgres_url(url):
            logger.warning(
                "PostgreSQL unavailable (%s). Falling back to SQLite for local dev. "
                "Configure DATABASE_URL for production.", e
            )
            url = "sqlite:///./ai_bettor_dev.db"
        engine = create_engine(url)
        logger.info("Using SQLite fallback database")
        return engine


engine = _make_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


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