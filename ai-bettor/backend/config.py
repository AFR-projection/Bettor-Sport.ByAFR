"""Configuration module for AI Bettor.

Environment variables are the *bootstrap* layer: they provide the defaults the
process starts with. Anything a user can change from the dashboard is stored in
the database and served by `backend.services.settings_service`, which treats the
values here as fallbacks.

Secrets must come from `.env` or the environment, never hardcoded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, NamedTuple

logger = logging.getLogger("ai-bettor.config")

ENV_PATH = Path(__file__).parent.parent / ".env"

TRUE_VALUES = ("1", "true", "yes", "on")


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in TRUE_VALUES


def _as_int(raw: str | None, default: int, minimum: int | None = None) -> int:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def _as_float(raw: str | None, default: float, minimum: float | None = None) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def load_env_file(path: Path = ENV_PATH, override: bool = False) -> Dict[str, str]:
    """Load a simple KEY=VALUE .env file into os.environ.

    Existing environment variables win unless `override` is True, so a shell
    export or a test monkeypatch is never silently replaced by the file.
    """
    loaded: Dict[str, str] = {}
    if not path.exists():
        logger.info("No .env file at %s — using process environment only", path)
        return loaded
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
        logger.info("Loaded %s value(s) from %s", len(loaded), path)
    except OSError as e:
        logger.warning("Failed to read .env file %s: %s", path, e)
    return loaded


class Settings(NamedTuple):
    """Immutable snapshot of the bootstrap configuration."""

    # --- secrets / integrations ---
    THE_ODDS_API_KEY: str
    THE_ODDS_API_KEYS: str
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # --- infrastructure ---
    DATABASE_URL: str
    TIMEZONE: str

    # --- deployment / access control ---
    # These stay environment-only on purpose: they are not dashboard-writable,
    # so a session that already holds the token cannot rotate it or widen CORS.
    API_TOKEN: str
    ALLOWED_ORIGINS: str
    ALLOW_SQLITE_FALLBACK: bool
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int
    DB_POOL_RECYCLE_SECONDS: int
    DB_CONNECT_TIMEOUT_SECONDS: int

    # --- scanning ---
    DEFAULT_SPORT: str
    DEFAULT_REGIONS: str
    DEFAULT_MARKETS: str
    MAX_LEAGUES_PER_SCAN: int
    ODDS_POLL_INTERVAL_SECONDS: int
    AGENT_SCAN_INTERVAL_SECONDS: int
    SCAN_AUTOMATION_ENABLED: bool
    EARLY_MORNING_ONLY: bool
    EARLY_MORNING_END_HOUR: int
    EARLY_MORNING_DAYS: int

    # --- modelling ---
    MONTE_CARLO_SIMULATIONS: int
    SIMULATION_BATCHES: int
    RANDOM_SEED: int
    MODEL_BLEND_WEIGHT: float
    DEFAULT_TOTAL_GOALS: float

    # --- strategy thresholds ---
    MIN_ODDS: float
    MAX_ODDS: float
    MIN_EDGE: float
    MIN_EV: float
    MIN_CONFIDENCE: int
    MIN_BOOKMAKERS: int
    MIN_DATA_QUALITY: int
    SCORE_BET_THRESHOLD: int
    MAX_UNCERTAINTY: float

    # --- staking / bankroll ---
    BETTING_MODE: str
    INITIAL_BANKROLL: float
    KELLY_FRACTION: float
    MAX_STAKE_PERCENT: float
    MAX_EXPOSURE_PERCENT: float

    # --- notifications / LLM ---
    TELEGRAM_MIN_SCORE: int
    TELEGRAM_MAX_PICKS: int
    TELEGRAM_SEND_NO_BET: bool
    LLM_REVIEW_ENABLED: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a settings snapshot from the current environment."""
        load_env_file()
        env = os.getenv
        return cls(
            THE_ODDS_API_KEY=env("THE_ODDS_API_KEY", "").strip(),
            THE_ODDS_API_KEYS=env("THE_ODDS_API_KEYS", "").strip(),
            OPENROUTER_API_KEY=env("OPENROUTER_API_KEY", "").strip(),
            OPENROUTER_MODEL=env("OPENROUTER_MODEL", "openrouter/auto").strip(),
            TELEGRAM_BOT_TOKEN=env("TELEGRAM_BOT_TOKEN", "").strip(),
            TELEGRAM_CHAT_ID=env("TELEGRAM_CHAT_ID", "").strip(),
            DATABASE_URL=env("DATABASE_URL", "sqlite:///./ai_bettor_dev.db").strip(),
            TIMEZONE=env("TIMEZONE", "Asia/Jakarta").strip(),
            API_TOKEN=env("API_TOKEN", "").strip(),
            ALLOWED_ORIGINS=env("ALLOWED_ORIGINS", "").strip(),
            ALLOW_SQLITE_FALLBACK=_as_bool(env("ALLOW_SQLITE_FALLBACK"), False),
            DB_POOL_SIZE=_as_int(env("DB_POOL_SIZE"), 5, 1),
            DB_MAX_OVERFLOW=_as_int(env("DB_MAX_OVERFLOW"), 5, 0),
            DB_POOL_RECYCLE_SECONDS=_as_int(env("DB_POOL_RECYCLE_SECONDS"), 280, 30),
            DB_CONNECT_TIMEOUT_SECONDS=_as_int(env("DB_CONNECT_TIMEOUT_SECONDS"), 10, 1),
            DEFAULT_SPORT=env("DEFAULT_SPORT", "soccer").strip(),
            DEFAULT_REGIONS=env("DEFAULT_REGIONS", "eu,uk").strip(),
            DEFAULT_MARKETS=env("DEFAULT_MARKETS", "h2h,spreads,totals").strip(),
            MAX_LEAGUES_PER_SCAN=_as_int(env("MAX_LEAGUES_PER_SCAN"), 6, 1),
            ODDS_POLL_INTERVAL_SECONDS=_as_int(env("ODDS_POLL_INTERVAL_SECONDS"), 300, 30),
            AGENT_SCAN_INTERVAL_SECONDS=_as_int(env("AGENT_SCAN_INTERVAL_SECONDS"), 900, 30),
            SCAN_AUTOMATION_ENABLED=_as_bool(env("SCAN_AUTOMATION_ENABLED"), True),
            EARLY_MORNING_ONLY=_as_bool(env("EARLY_MORNING_ONLY"), True),
            EARLY_MORNING_END_HOUR=_as_int(env("EARLY_MORNING_END_HOUR"), 6, 0),
            EARLY_MORNING_DAYS=_as_int(env("EARLY_MORNING_DAYS"), 2, 1),
            MONTE_CARLO_SIMULATIONS=_as_int(env("MONTE_CARLO_SIMULATIONS"), 20000, 100),
            SIMULATION_BATCHES=_as_int(env("SIMULATION_BATCHES"), 3, 1),
            RANDOM_SEED=_as_int(env("RANDOM_SEED"), 42),
            MODEL_BLEND_WEIGHT=_as_float(env("MODEL_BLEND_WEIGHT"), 0.35, 0.0),
            DEFAULT_TOTAL_GOALS=_as_float(env("DEFAULT_TOTAL_GOALS"), 2.7, 0.5),
            MIN_ODDS=_as_float(env("MIN_ODDS"), 1.75, 1.01),
            MAX_ODDS=_as_float(env("MAX_ODDS"), 6.0, 1.02),
            MIN_EDGE=_as_float(env("MIN_EDGE"), 0.02, 0.0),
            MIN_EV=_as_float(env("MIN_EV"), 0.02, 0.0),
            MIN_CONFIDENCE=_as_int(env("MIN_CONFIDENCE"), 60, 0),
            MIN_BOOKMAKERS=_as_int(env("MIN_BOOKMAKERS"), 3, 1),
            MIN_DATA_QUALITY=_as_int(env("MIN_DATA_QUALITY"), 50, 0),
            SCORE_BET_THRESHOLD=_as_int(env("SCORE_BET_THRESHOLD"), 80, 0),
            MAX_UNCERTAINTY=_as_float(env("MAX_UNCERTAINTY"), 0.5, 0.0),
            BETTING_MODE=env("BETTING_MODE", "PAPER").strip().upper(),
            INITIAL_BANKROLL=_as_float(env("INITIAL_BANKROLL"), 1000.0, 1.0),
            KELLY_FRACTION=_as_float(env("KELLY_FRACTION"), 0.25, 0.01),
            MAX_STAKE_PERCENT=_as_float(env("MAX_STAKE_PERCENT"), 2.0, 0.1),
            MAX_EXPOSURE_PERCENT=_as_float(env("MAX_EXPOSURE_PERCENT"), 20.0, 1.0),
            TELEGRAM_MIN_SCORE=_as_int(env("TELEGRAM_MIN_SCORE"), 85, 0),
            TELEGRAM_MAX_PICKS=_as_int(env("TELEGRAM_MAX_PICKS"), 5, 1),
            TELEGRAM_SEND_NO_BET=_as_bool(env("TELEGRAM_SEND_NO_BET"), True),
            LLM_REVIEW_ENABLED=_as_bool(env("LLM_REVIEW_ENABLED"), True),
        )

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide bootstrap settings (cached)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests and after editing .env."""
    global _settings
    _settings = Settings.from_env()
    return _settings


# Backwards-compatible module attribute used by older code paths.
settings_obj = get_settings()
