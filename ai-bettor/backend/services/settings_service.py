"""Runtime settings service for AI Bettor.

Stores settings in the database (system_settings table) with env-var defaults.
Secrets are persisted but never returned unmasked via the API.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.database.models import SystemSetting
from backend.database.session import session_scope
from backend.integrations.odds_router import OddsApiRouter, get_odds_router

logger = logging.getLogger("ai-bettor.settings")

SENSITIVE_KEYS = {
    "THE_ODDS_API_KEYS",
    "OPENROUTER_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "THE_ODDS_API_KEYS": [],
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_MODEL": "openrouter/auto",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "MIN_ODDS": 1.75,
    "MIN_EDGE": 0.01,
    "MIN_EV": 0.01,
    "MIN_CONFIDENCE": 60,
    "TIMEZONE": "Asia/Jakarta",
    "MONTE_CARLO_SIMULATIONS": 20000,
    "RANDOM_SEED": 42,
    "BETTING_MODE": "PAPER",
    "EARLY_MORNING_ONLY": True,
    "EARLY_MORNING_END_HOUR": 6,
    "EARLY_MORNING_DAYS": 2,
    "TELEGRAM_MIN_SCORE": 85,
    "TELEGRAM_MAX_PICKS": 5,
    "SIMULATION_BATCHES": 3,
    "SCAN_AUTOMATION_ENABLED": True,
}


class SettingsService:
    """Persistent settings with DB storage and env fallback."""

    def __init__(self, router: Optional[OddsApiRouter] = None):
        self._lock = threading.RLock()
        self.router = router or get_odds_router()
        self._cache: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading / persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        env = get_settings()
        defaults = dict(DEFAULT_SETTINGS)
        if env.THE_ODDS_API_KEY:
            defaults["THE_ODDS_API_KEYS"] = [env.THE_ODDS_API_KEY]
        if env.OPENROUTER_API_KEY:
            defaults["OPENROUTER_API_KEY"] = env.OPENROUTER_API_KEY
        if env.TELEGRAM_BOT_TOKEN:
            defaults["TELEGRAM_BOT_TOKEN"] = env.TELEGRAM_BOT_TOKEN
        if env.TELEGRAM_CHAT_ID:
            defaults["TELEGRAM_CHAT_ID"] = env.TELEGRAM_CHAT_ID
        self._cache = defaults

        try:
            with session_scope() as session:
                rows = session.query(SystemSetting).all()
                for row in rows:
                    value = self._deserialize(row.value)
                    self._cache[row.key] = value
        except Exception as e:
            logger.warning("Failed to load settings from DB: %s", e)

        self._sync_router()

    def _sync_router(self) -> None:
        keys = [k for k in (self._cache.get("THE_ODDS_API_KEYS") or []) if k]
        self.router.set_keys(keys)

    def _persist(self, key: str, value: Any) -> None:
        try:
            with session_scope() as session:
                row = session.query(SystemSetting).filter(SystemSetting.key == key).first()
                if row is None:
                    row = SystemSetting(key=key, value=self._serialize(value))
                    session.add(row)
                else:
                    row.value = self._serialize(value)
        except Exception as e:
            logger.warning("Failed to persist setting %s: %s", key, e)

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value)

    @staticmethod
    def _deserialize(raw: str) -> Any:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._cache.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache)

    def get_odds_api_keys(self) -> List[str]:
        return [k for k in (self.get("THE_ODDS_API_KEYS") or []) if k]

    def update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings from a request payload. Returns masked view."""
        with self._lock:
            for key, value in payload.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                value = self._coerce(key, value)
                if key == "THE_ODDS_API_KEYS":
                    value = [k.strip() for k in value if k and k.strip()]
                    self._cache[key] = value
                    self.router.set_keys(value)
                    for k in value:
                        self.router.reset_key(k)
                    self._persist(key, value)
                elif key in SENSITIVE_KEYS and not value:
                    continue
                else:
                    self._cache[key] = value
                    self._persist(key, value)
            return self.masked_view()

    def _coerce(self, key: str, value: Any) -> Any:
        types = {
            "MIN_ODDS": float, "MIN_EDGE": float, "MIN_EV": float,
            "MIN_CONFIDENCE": int, "MONTE_CARLO_SIMULATIONS": int,
            "RANDOM_SEED": int, "EARLY_MORNING_END_HOUR": int,
            "EARLY_MORNING_DAYS": int, "TELEGRAM_MIN_SCORE": int,
            "TELEGRAM_MAX_PICKS": int, "SIMULATION_BATCHES": int,
        }
        bools = {"EARLY_MORNING_ONLY", "SCAN_AUTOMATION_ENABLED"}
        if key in bools:
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("1", "true", "yes", "on")
        if key in types:
            try:
                return types[key](value)
            except (TypeError, ValueError):
                return DEFAULT_SETTINGS[key]
        if key == "THE_ODDS_API_KEYS":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = [value]
            return list(value) if isinstance(value, list) else []
        return str(value) if value is not None else ""

    # ------------------------------------------------------------------
    # Masked view for API responses
    # ------------------------------------------------------------------

    def masked_view(self) -> Dict[str, Any]:
        with self._lock:
            view: Dict[str, Any] = {}
            for key, value in self._cache.items():
                if key == "THE_ODDS_API_KEYS":
                    keys = [k for k in value if k]
                    view["THE_ODDS_API_KEYS"] = [
                        {"label": self._mask(k), "is_set": True} for k in keys
                    ]
                    view["THE_ODDS_API_KEYS_COUNT"] = len(keys)
                elif key in SENSITIVE_KEYS:
                    view[key] = self._mask(value) if value else ""
                    view[key + "_SET"] = bool(value)
                else:
                    view[key] = value
            view["odds_router"] = [
                {**entry, "full_key": self._mask(entry["full_key"])}
                for entry in self.router.status()
            ]
            return view

    @staticmethod
    def _mask(value: str) -> str:
        if not value:
            return ""
        if len(value) <= 8:
            return "***"
        return value[:4] + "..." + value[-4:]


_settings_service: Optional[SettingsService] = None


def get_settings_service() -> SettingsService:
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service