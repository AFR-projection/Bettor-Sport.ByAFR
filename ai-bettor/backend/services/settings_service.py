"""Runtime settings service for AI Bettor.

Settings live in the database (`system_settings` table) and fall back to the
environment snapshot from `backend.config`. This service is the single source of
truth at runtime: agents, integrations and the scheduler all read from it, so a
change made in the dashboard takes effect without restarting the process.

Every successful `update()` bumps `revision`, which consumers use to rebuild
cached objects (see `backend.services.pipeline.get_pipeline`).

Secrets are persisted but only ever returned masked.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

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

# Every key the dashboard may write. Anything not listed here is rejected.
DEFAULT_SETTINGS: Dict[str, Any] = {
    # integrations
    "THE_ODDS_API_KEYS": [],
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_MODEL": "openrouter/auto",
    "LLM_REVIEW_ENABLED": True,
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    # scanning
    "DEFAULT_SPORT": "soccer",
    "DEFAULT_REGIONS": "eu,uk",
    "DEFAULT_MARKETS": "h2h,spreads,totals",
    "MAX_LEAGUES_PER_SCAN": 6,
    "AGENT_SCAN_INTERVAL_SECONDS": 900,
    "SCAN_AUTOMATION_ENABLED": True,
    "EARLY_MORNING_ONLY": True,
    "EARLY_MORNING_END_HOUR": 6,
    "EARLY_MORNING_DAYS": 2,
    # modelling
    "MONTE_CARLO_SIMULATIONS": 20000,
    "SIMULATION_BATCHES": 3,
    "RANDOM_SEED": 42,
    "MODEL_BLEND_WEIGHT": 0.35,
    "DEFAULT_TOTAL_GOALS": 2.7,
    # strategy
    "MIN_ODDS": 1.75,
    "MAX_ODDS": 6.0,
    "MIN_EDGE": 0.02,
    "MIN_EV": 0.02,
    "MIN_CONFIDENCE": 60,
    "MIN_BOOKMAKERS": 3,
    "MIN_DATA_QUALITY": 50,
    "SCORE_BET_THRESHOLD": 80,
    "MAX_UNCERTAINTY": 0.5,
    # bankroll
    "BETTING_MODE": "PAPER",
    "INITIAL_BANKROLL": 1000.0,
    "KELLY_FRACTION": 0.25,
    "MAX_STAKE_PERCENT": 2.0,
    "MAX_EXPOSURE_PERCENT": 20.0,
    # notifications
    "TELEGRAM_MIN_SCORE": 85,
    "TELEGRAM_MAX_PICKS": 5,
    "TELEGRAM_SEND_NO_BET": True,
    # misc
    "TIMEZONE": "Asia/Jakarta",
}

INT_KEYS = {
    "MAX_LEAGUES_PER_SCAN", "AGENT_SCAN_INTERVAL_SECONDS", "EARLY_MORNING_END_HOUR",
    "EARLY_MORNING_DAYS", "MONTE_CARLO_SIMULATIONS", "SIMULATION_BATCHES",
    "RANDOM_SEED", "MIN_CONFIDENCE", "MIN_BOOKMAKERS", "MIN_DATA_QUALITY",
    "SCORE_BET_THRESHOLD", "TELEGRAM_MIN_SCORE", "TELEGRAM_MAX_PICKS",
}

FLOAT_KEYS = {
    "MODEL_BLEND_WEIGHT", "DEFAULT_TOTAL_GOALS", "MIN_ODDS", "MAX_ODDS",
    "MIN_EDGE", "MIN_EV", "MAX_UNCERTAINTY", "INITIAL_BANKROLL",
    "KELLY_FRACTION", "MAX_STAKE_PERCENT", "MAX_EXPOSURE_PERCENT",
}

BOOL_KEYS = {
    "LLM_REVIEW_ENABLED", "SCAN_AUTOMATION_ENABLED", "EARLY_MORNING_ONLY",
    "TELEGRAM_SEND_NO_BET",
}

# Hard guard rails so a typo in the UI cannot produce a nonsensical engine.
BOUNDS: Dict[str, tuple] = {
    "MAX_LEAGUES_PER_SCAN": (1, 40),
    "AGENT_SCAN_INTERVAL_SECONDS": (30, 86400),
    "EARLY_MORNING_END_HOUR": (0, 23),
    "EARLY_MORNING_DAYS": (1, 7),
    "MONTE_CARLO_SIMULATIONS": (100, 500000),
    "SIMULATION_BATCHES": (1, 20),
    "MIN_CONFIDENCE": (0, 100),
    "MIN_BOOKMAKERS": (1, 30),
    "MIN_DATA_QUALITY": (0, 100),
    "SCORE_BET_THRESHOLD": (0, 100),
    "TELEGRAM_MIN_SCORE": (0, 100),
    "TELEGRAM_MAX_PICKS": (1, 50),
    "MODEL_BLEND_WEIGHT": (0.0, 1.0),
    "DEFAULT_TOTAL_GOALS": (0.5, 8.0),
    "MIN_ODDS": (1.01, 100.0),
    "MAX_ODDS": (1.02, 1000.0),
    "MIN_EDGE": (0.0, 1.0),
    "MIN_EV": (-1.0, 5.0),
    "MAX_UNCERTAINTY": (0.0, 1.0),
    "INITIAL_BANKROLL": (1.0, 100000000.0),
    "KELLY_FRACTION": (0.01, 1.0),
    "MAX_STAKE_PERCENT": (0.1, 100.0),
    "MAX_EXPOSURE_PERCENT": (1.0, 100.0),
}

ENUMS: Dict[str, tuple] = {
    "BETTING_MODE": ("PAPER", "LIVE"),
}


class SettingsService:
    """Persistent settings with DB storage, env fallback and change tracking."""

    def __init__(self, router: Optional[OddsApiRouter] = None):
        self._lock = threading.RLock()
        self.router = router or get_odds_router()
        self._cache: Dict[str, Any] = {}
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.revision = 0
        self._load()

    # ------------------------------------------------------------------
    # Loading / persistence
    # ------------------------------------------------------------------

    def _env_defaults(self) -> Dict[str, Any]:
        """DEFAULT_SETTINGS overlaid with anything provided by .env."""
        env = get_settings()
        defaults = dict(DEFAULT_SETTINGS)
        for key in defaults:
            env_value = getattr(env, key, None)
            if env_value is None:
                continue
            if isinstance(env_value, str) and not env_value:
                continue
            defaults[key] = env_value

        keys: List[str] = []
        if env.THE_ODDS_API_KEYS:
            keys.extend(k.strip() for k in env.THE_ODDS_API_KEYS.split(",") if k.strip())
        if env.THE_ODDS_API_KEY and env.THE_ODDS_API_KEY not in keys:
            keys.insert(0, env.THE_ODDS_API_KEY)
        defaults["THE_ODDS_API_KEYS"] = keys
        return defaults

    def _load(self) -> None:
        self._cache = self._env_defaults()
        try:
            with session_scope() as session:
                for row in session.query(SystemSetting).all():
                    if row.key not in DEFAULT_SETTINGS:
                        continue
                    self._cache[row.key] = self._coerce(row.key, self._deserialize(row.value))
        except Exception as e:  # table may not exist yet on first boot
            logger.warning("Could not load settings from DB (%s) — using env defaults", e)
        self._sync_router()

    def reload(self) -> Dict[str, Any]:
        """Re-read settings from the database."""
        with self._lock:
            self._load()
            self.revision += 1
            return self.masked_view()

    def _sync_router(self) -> None:
        self.router.set_keys(self.get_odds_api_keys())

    def _persist(self, key: str, value: Any) -> bool:
        try:
            with session_scope() as session:
                row = session.query(SystemSetting).filter(SystemSetting.key == key).first()
                if row is None:
                    session.add(SystemSetting(key=key, value=self._serialize(value)))
                else:
                    row.value = self._serialize(value)
            return True
        except Exception as e:
            logger.error("Failed to persist setting %s: %s", key, e)
            return False

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value)

    @staticmethod
    def _deserialize(raw: str) -> Any:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    # ------------------------------------------------------------------
    # Change notification
    # ------------------------------------------------------------------

    def on_change(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback fired after settings change."""
        with self._lock:
            self._listeners.append(callback)

    def _notify(self, changed: Dict[str, Any]) -> None:
        for callback in list(self._listeners):
            try:
                callback(changed)
            except Exception as e:
                logger.warning("Settings listener failed: %s", e)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if default is not None:
                return default
            return DEFAULT_SETTINGS.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache)

    def get_odds_api_keys(self) -> List[str]:
        raw = self.get("THE_ODDS_API_KEYS") or []
        if isinstance(raw, str):
            raw = [raw]
        return [k for k in (str(k).strip() for k in raw) if k]

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a settings payload. Unknown keys are reported, not silently lost."""
        applied: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        failed: List[str] = []

        with self._lock:
            for key, value in payload.items():
                if key not in DEFAULT_SETTINGS:
                    rejected[key] = "unknown_setting"
                    continue
                if key == "THE_ODDS_API_KEYS":
                    keys = self._coerce(key, value)
                    self._cache[key] = keys
                    self.router.set_keys(keys)
                    for k in keys:
                        self.router.reset_key(k)
                    if self._persist(key, keys):
                        applied[key] = f"{len(keys)} key(s)"
                    else:
                        failed.append(key)
                    continue
                if key in SENSITIVE_KEYS and (value is None or str(value).strip() == ""):
                    # Empty secret field means "keep what is stored".
                    rejected[key] = "empty_ignored"
                    continue
                coerced = self._coerce(key, value)
                self._cache[key] = coerced
                if self._persist(key, coerced):
                    applied[key] = "***" if key in SENSITIVE_KEYS else coerced
                else:
                    failed.append(key)

            if applied:
                self.revision += 1
            view = self.masked_view()

        if applied:
            self._notify(applied)
        view["applied"] = applied
        view["rejected"] = rejected
        view["failed"] = failed
        view["saved"] = not failed
        view["revision"] = self.revision
        return view

    def _coerce(self, key: str, value: Any) -> Any:
        default = DEFAULT_SETTINGS[key]

        if key == "THE_ODDS_API_KEYS":
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    value = parsed if isinstance(parsed, list) else [value]
                except ValueError:
                    value = [v for v in value.split(",")]
            if not isinstance(value, (list, tuple)):
                return []
            seen: List[str] = []
            for item in value:
                if isinstance(item, dict):  # masked view round-trip guard
                    continue
                text = str(item).strip()
                if text and text not in seen:
                    seen.append(text)
            return seen

        if key in BOOL_KEYS:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on")

        if key in INT_KEYS:
            try:
                result = int(float(value))
            except (TypeError, ValueError):
                return default
            return self._clamp(key, result)

        if key in FLOAT_KEYS:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return default
            return self._clamp(key, result)

        text = "" if value is None else str(value).strip()
        if key in ENUMS:
            upper = text.upper()
            return upper if upper in ENUMS[key] else default
        return text or default if key not in SENSITIVE_KEYS else text

    @staticmethod
    def _clamp(key: str, value: Any) -> Any:
        bounds = BOUNDS.get(key)
        if not bounds:
            return value
        low, high = bounds
        return max(low, min(high, value))

    # ------------------------------------------------------------------
    # Masked view for API responses
    # ------------------------------------------------------------------

    def masked_view(self) -> Dict[str, Any]:
        with self._lock:
            view: Dict[str, Any] = {}
            for key, value in self._cache.items():
                if key == "THE_ODDS_API_KEYS":
                    keys = self.get_odds_api_keys()
                    view["THE_ODDS_API_KEYS"] = [
                        {"index": i, "label": self._mask(k), "is_set": True}
                        for i, k in enumerate(keys)
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
            view["revision"] = self.revision
            return view

    @staticmethod
    def _mask(value: Any) -> str:
        text = "" if value is None else str(value)
        if not text:
            return ""
        if len(text) <= 8:
            return "***"
        return text[:4] + "..." + text[-4:]


_settings_service: Optional[SettingsService] = None
_service_lock = threading.Lock()


def get_settings_service() -> SettingsService:
    global _settings_service
    if _settings_service is None:
        with _service_lock:
            if _settings_service is None:
                _settings_service = SettingsService()
    return _settings_service


def reset_settings_service() -> None:
    """Drop the cached service (used by tests)."""
    global _settings_service
    with _service_lock:
        _settings_service = None


def get_setting(key: str, default: Any = None) -> Any:
    """Read one live setting, falling back to the environment snapshot.

    Consumers (agents, integrations, scheduler) use this instead of reading
    `backend.config` directly, so a value saved from the dashboard takes effect
    without a restart. Never raises: if the database is unreachable the env
    snapshot is used.
    """
    try:
        return get_settings_service().get(key, default)
    except Exception:  # pragma: no cover - DB unavailable
        env_value = getattr(get_settings(), key, None)
        if env_value is None:
            return default if default is not None else DEFAULT_SETTINGS.get(key)
        return env_value


def get_revision() -> int:
    """Current settings revision (0 when the service cannot be built)."""
    try:
        return get_settings_service().revision
    except Exception:  # pragma: no cover - DB unavailable
        return 0
