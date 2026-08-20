"""Multi-key The Odds API router with automatic failover.

Rotates through multiple API keys (A -> B -> C -> ...) automatically:
- 429 (rate limit)  -> key gets cooldown, switch to next healthy key
- 401 (invalid)     -> key disabled until re-added or re-tested
- 5xx / timeout     -> key gets cooldown, switch to next healthy key
- Success           -> key returns to healthy state

All state is thread-safe so the API pipeline and settings UI can share it.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


DEFAULT_COOLDOWN_429_SECONDS = 60
DEFAULT_COOLDOWN_5XX_SECONDS = 30
DEFAULT_COOLDOWN_NETWORK_SECONDS = 30


def mask_key(key: str) -> str:
    """Mask a key for display, e.g. 'abc...WXYZ'."""
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


class OddsApiRouter:
    """Round-robin failover router for The Odds API keys."""

    def __init__(self, keys: Optional[List[str]] = None):
        self._lock = threading.RLock()
        self._keys: List[str] = []
        self._status: Dict[str, Dict[str, Any]] = {}
        self._index = 0
        if keys:
            for key in keys:
                self.add_key(key)

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def add_key(self, key: str) -> None:
        key = (key or "").strip()
        if not key:
            return
        with self._lock:
            if key in self._status:
                return
            self._keys.append(key)
            self._status[key] = {
                "status": "OK",
                "requests": 0,
                "failures": 0,
                "cooldown_until": 0.0,
                "disabled": False,
                "last_error": None,
                "last_used": None,
                "remaining_requests": None,
                "used_requests": None,
            }

    def remove_key(self, key: str) -> None:
        with self._lock:
            if key in self._status:
                del self._status[key]
            if key in self._keys:
                self._keys.remove(key)
            if self._index >= len(self._keys) and self._keys:
                self._index = 0

    def set_keys(self, keys: List[str]) -> None:
        """Replace the key set, preserving health state of keys that stay."""
        cleaned: List[str] = []
        for key in keys:
            key = (key or "").strip()
            if key and key not in cleaned:
                cleaned.append(key)
        with self._lock:
            for key in list(self._keys):
                if key not in cleaned:
                    self.remove_key(key)
            for key in cleaned:
                self.add_key(key)
            # Keep the router order identical to the configured order so key A
            # really is tried first.
            self._keys = cleaned
            if self._index >= len(self._keys):
                self._index = 0

    @property
    def has_keys(self) -> bool:
        return len(self._keys) > 0

    def key_count(self) -> int:
        return len(self._keys)

    def healthy_count(self) -> int:
        """Number of keys that are neither disabled nor cooling down."""
        with self._lock:
            now = time.time()
            return sum(
                1 for key in self._keys
                if not self._status[key]["disabled"] and self._status[key]["cooldown_until"] <= now
            )

    # ------------------------------------------------------------------
    # Selection / rotation
    # ------------------------------------------------------------------

    def get_key(self) -> Optional[str]:
        """Return next healthy key or None if all keys are unavailable."""
        with self._lock:
            if not self._keys:
                return None
            now = time.time()
            for _ in range(len(self._keys)):
                self._index = (self._index + 1) % len(self._keys)
                key = self._keys[self._index]
                st = self._status[key]
                if st["disabled"]:
                    continue
                if st["cooldown_until"] > now:
                    continue
                return key
            return None

    def peek_key(self) -> Optional[str]:
        """Return the next healthy key WITHOUT advancing rotation (for status)."""
        with self._lock:
            if not self._keys:
                return None
            now = time.time()
            for offset in range(1, len(self._keys) + 1):
                key = self._keys[(self._index + offset) % len(self._keys)]
                st = self._status[key]
                if st["disabled"] or st["cooldown_until"] > now:
                    continue
                return key
            return None

    def report_success(self, key: str, remaining: Optional[int] = None, used: Optional[int] = None) -> None:
        with self._lock:
            st = self._status.get(key)
            if not st:
                return
            st["requests"] += 1
            st["status"] = "OK"
            st["cooldown_until"] = 0.0
            st["disabled"] = False
            st["last_error"] = None
            st["last_used"] = time.time()
            if remaining is not None:
                st["remaining_requests"] = remaining
            if used is not None:
                st["used_requests"] = used

    def report_failure(self, key: str, reason: str, http_status: Optional[int] = None) -> None:
        with self._lock:
            st = self._status.get(key)
            if not st:
                return
            st["failures"] += 1
            st["last_error"] = reason
            st["last_used"] = time.time()
            if http_status == 401 or http_status == 403:
                st["disabled"] = True
                st["status"] = "DISABLED"
                st["cooldown_until"] = 0.0
            elif http_status == 429:
                st["status"] = "COOLDOWN"
                st["cooldown_until"] = time.time() + DEFAULT_COOLDOWN_429_SECONDS
            elif http_status and http_status >= 500:
                st["status"] = "COOLDOWN"
                st["cooldown_until"] = time.time() + DEFAULT_COOLDOWN_5XX_SECONDS
            else:
                st["status"] = "COOLDOWN"
                st["cooldown_until"] = time.time() + DEFAULT_COOLDOWN_NETWORK_SECONDS

    def reset_key(self, key: str) -> None:
        """Reset a key to healthy (e.g. after user re-enters it)."""
        with self._lock:
            st = self._status.get(key)
            if st:
                st["status"] = "OK"
                st["disabled"] = False
                st["cooldown_until"] = 0.0
                st["last_error"] = None

    def clear_all(self) -> None:
        with self._lock:
            self._keys = []
            self._status = {}
            self._index = 0

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def status(self) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            result = []
            for key in self._keys:
                st = self._status[key]
                remaining = st["cooldown_until"] - now
                result.append({
                    "label": mask_key(key),
                    "full_key": key,
                    "status": st["status"],
                    "requests": st["requests"],
                    "failures": st["failures"],
                    "cooldown_remaining_seconds": round(max(0, remaining), 1),
                    "disabled": st["disabled"],
                    "last_error": st["last_error"],
                    "last_used": st["last_used"],
                    "remaining_requests": st["remaining_requests"],
                    "used_requests": st.get("used_requests"),
                })
            return result

    def active_key_label(self) -> Optional[str]:
        key = self.peek_key()
        return mask_key(key) if key else None


_router: Optional[OddsApiRouter] = None


def get_odds_router() -> OddsApiRouter:
    """Lazy singleton for the shared router."""
    global _router
    if _router is None:
        _router = OddsApiRouter()
    return _router