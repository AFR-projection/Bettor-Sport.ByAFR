# """Database engine and session management."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
import logging
from backend.config.settings import settings

logger = logging.getLogger("db")
engine = None
async_session_factory = None

def get_database_url() -> str:
    return settings.database_url

def init_db(url: str | None = None) -> None:
    global engine, async_session_factory
    db_url = url or get_database_url()
    engine = create_async_engine(
        db_url, echo=False, pool_pre_ping=True,
        pool_size=5, max_overflow=10
    )
    async_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Database engine initialized")

async def get_session() -> AsyncSession:
    global async_session_factory
    if async_session_factory is None:
        init_db()
    async with async_session_factory() as session:
        yield session

class Base(DeclarativeBase):
    pass

async def close_engine() -> None:
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None
