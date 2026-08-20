"""Alembic environment for AI Bettor.

The connection URL comes from the app's own configuration (`DATABASE_URL`, with
Neon's driver + `sslmode` normalisation already applied), so `alembic upgrade
head` on the server always targets the same database the app uses — there is no
second URL in alembic.ini to keep in sync.

`target_metadata` is the real model metadata, which is what makes
`alembic revision --autogenerate` work.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import get_settings  # noqa: E402
from backend.database.models import Base  # noqa: E402
from backend.database.session import normalise_database_url  # noqa: E402

config = context.config

# alembic.ini deliberately carries no logging config; configure it only if a
# real one is present so this does not raise on a bare ini file.
if config.config_file_name is not None and config.get_section("loggers"):
    fileConfig(config.config_file_name)


def _database_url() -> str:
    """CLI override (`-x url=…`) > environment > app settings."""
    override = context.get_x_argument(as_dictionary=True).get("url")
    if override:
        return normalise_database_url(override)
    return normalise_database_url(os.getenv("DATABASE_URL") or get_settings().DATABASE_URL)


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of touching a database (`--sql`)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most columns; batch mode rewrites the table.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
