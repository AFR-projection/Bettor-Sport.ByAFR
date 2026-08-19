from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
config = context.config
if config.config_file_name is not None: fileConfig(config.config_file_name)
target_metadata = None
def run_migrations_online():
    cfg = config.get_main_option("sqlalchemy.url")
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}) or {}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as conn:
        context.configure(conn=conn, target_metadata=target_metadata)
        with context.begin_transaction(): context.run_migrations()
if context.is_offline_mode():
    context.configure(url=cfg, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction(): context.run_migrations()
else:
    run_migrations_online()

