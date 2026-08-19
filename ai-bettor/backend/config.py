"""Configuration module for AI Bettor.

Loads settings from environment variables.
All API keys and sensitive values must come from .env file,
NEVER hardcoded in source code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, NamedTuple


class Settings(NamedTuple):
    """Settings object with attribute access."""
    THE_ODDS_API_KEY: str
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    DATABASE_URL: str
    TIMEZONE: str
    MONTE_CARLO_SIMULATIONS: int
    RANDOM_SEED: int
    ODDS_POLL_INTERVAL_SECONDS: int
    MIN_EDGE: float
    MIN_EV: float
    MIN_CONFIDENCE: int
    BETTING_MODE: str
    AGENT_SCAN_INTERVAL_SECONDS: int
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings from environment variables."""
        # Determine .env file path
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Load .env file if it exists (simple key=value parsing)
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ.setdefault(key, value)
                logger.info(f"Loaded configuration from {env_path}")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
        else:
            logger.warning(f".env file not found at {env_path}, using system environment")
        
        # Extract settings with defaults
        settings = cls(
            THE_ODDS_API_KEY=os.getenv("THE_ODDS_API_KEY", ""),
            OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY", ""),
            OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID", ""),
            DATABASE_URL=os.getenv("DATABASE_URL", "postgresql://localhost:5432/ai_bettor"),
            TIMEZONE=os.getenv("TIMEZONE", "Asia/Jakarta"),
            MONTE_CARLO_SIMULATIONS=int(os.getenv("MONTE_CARLO_SIMULATIONS", "20000")),
            RANDOM_SEED=int(os.getenv("RANDOM_SEED", "42")),
            ODDS_POLL_INTERVAL_SECONDS=int(os.getenv("ODDS_POLL_INTERVAL_SECONDS", "300")),
            MIN_EDGE=float(os.getenv("MIN_EDGE", "0.01")),
            MIN_EV=float(os.getenv("MIN_EV", "0.01")),
            MIN_CONFIDENCE=int(os.getenv("MIN_CONFIDENCE", "60")),
            BETTING_MODE=os.getenv("BETTING_MODE", "PAPER"),
            AGENT_SCAN_INTERVAL_SECONDS=int(os.getenv("AGENT_SCAN_INTERVAL_SECONDS", "60")),
        )
        
        return settings


# Logger
import logging
logger = logging.getLogger(__name__)

# Create default settings instance
settings_obj = Settings.from_env()


def get_settings() -> Settings:
    """Get settings object."""
    return settings_obj