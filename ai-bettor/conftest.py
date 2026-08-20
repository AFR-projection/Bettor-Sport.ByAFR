"""Pytest bootstrap: pin the test database BEFORE any backend module imports.

Backend engine is a module-level singleton. Without this, the first test file
imported (alphabetical order) creates it with the DATABASE_URL from .env
(Neon) and every SQLite-backed test then fails. pytest imports this conftest
before collecting any test module, so the env var is already set.

.env is also cleared of values tests depend on being unset (API_TOKEN,
OPENROUTER_API_KEY) or default (TELEGRAM_SEND_NO_BET), otherwise the real
production .env silently changes test expectations.
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_ai_bettor.db"
os.environ["API_TOKEN"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["TELEGRAM_SEND_NO_BET"] = "true"