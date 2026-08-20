"""Telegram integration for AI Bettor.

Sends betting picks to Telegram via Bot API.
Only sends picks that pass the threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from backend.config import get_settings

logger = logging.getLogger("ai-bettor.telegram")

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramNotifier:
    """Send formatted pick messages to Telegram."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, timeout: int = 15):
        # Live settings first: a token saved from the dashboard must work without
        # restarting the process. The .env snapshot is only the fallback.
        from backend.services.settings_service import get_setting

        settings = get_settings()
        self.bot_token = bot_token or get_setting(
            "TELEGRAM_BOT_TOKEN", settings.TELEGRAM_BOT_TOKEN) or ""
        self.chat_id = chat_id or get_setting(
            "TELEGRAM_CHAT_ID", settings.TELEGRAM_CHAT_ID) or ""
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Send a raw text message to Telegram."""
        if not self.is_configured:
            logger.warning("Telegram not configured, message not sent")
            return False

        url = f"{TELEGRAM_API_BASE}{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id or self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                logger.info("Telegram message sent")
                return True
            else:
                logger.error("Telegram send failed: %s - %s", resp.status_code, resp.text[:200])
                return False
        except requests.exceptions.RequestException as e:
            logger.error("Telegram request error: %s", e)
            return False

    @staticmethod
    def _num(decision: Dict[str, Any], *keys: str) -> float:
        """First usable numeric value among `keys` (0.0 when none exist)."""
        for key in keys:
            value = decision.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def format_pick_message(self, decision: Dict[str, Any]) -> str:
        """Format a BettorBrain decision into the standard pick message."""
        score = decision.get("score")
        header = "PREMIUM PICK" if (score or 0) >= 90 else "HIGH-CONFIDENCE PICK"
        # The brain exposes the model probability as both `model_probability`
        # and `probability`; accept either so a pick never reports 0%.
        model_probability = self._num(decision, "model_probability", "probability")
        pick_label = (decision.get("label")
                      or " ".join(str(p) for p in (decision.get("market"),
                                                   decision.get("selection")) if p)
                      or "UNKNOWN")
        bookmaker = decision.get("bookmaker") or "best available"
        lines = [
            f"\U0001F3C6 AI BETTOR — {header}",
            "",
            "\u26BD MATCH",
            f"{decision.get('home_team', 'Home')} vs {decision.get('away_team', 'Away')}",
            "",
            "\U0001F550 KICKOFF",
            f"{decision.get('kickoff_wib') or decision.get('kickoff') or 'UNKNOWN'} (WIB)",
            "",
            "\U0001F3AF MARKET",
            f"{decision.get('market', 'UNKNOWN')}",
            "",
            "\U0001F4CC PICK",
            f"{pick_label}",
            "",
            "\U0001F4B0 ODDS",
            f"{self._num(decision, 'odds'):.2f} @ {bookmaker}",
            "",
            "\U0001F4CA MODEL PROBABILITY",
            f"{model_probability * 100:.1f}%",
            "",
            "\U0001F4C8 IMPLIED PROBABILITY",
            f"{self._num(decision, 'implied_probability') * 100:.1f}%",
            "",
            "\U0001F48E EDGE",
            f"+{self._num(decision, 'edge') * 100:.1f}%",
            "",
            "\U0001F4B0 EV",
            f"+{self._num(decision, 'ev'):.2f} per unit",
            "",
            "\U0001F4B5 STAKE",
            f"{self._num(decision, 'stake'):.2f} "
            f"({self._num(decision, 'stake_percent'):.2f}% of bankroll)",
            "",
            "\U0001F3B2 SIMULATION",
            f"{int(self._num(decision, 'simulation_count') or 20000):,} runs (repeated batches)",
            "",
            "\U0001F4AF SCORE",
            f"{score if score is not None else decision.get('confidence', 0)}/100",
            "",
            "\U0001F6E1\uFE0F RISK",
            f"{decision.get('risk', 'UNKNOWN')}",
            "",
            "\U0001F9E0 CONFIDENCE",
            f"{decision.get('confidence', 0)}/100",
            "",
            "DECISION:",
            f"\U0001F7E2 {decision.get('decision', 'NO BET')}",
            "",
            "REASON:",
            (decision.get('reasoning')
             or '; '.join(str(r) for r in (decision.get('reasons') or []))
             or 'No reasoning available'),
            "",
            "\u26A0\uFE0F Probabilistic analysis. Tidak ada jaminan hasil.",
        ]
        return "\n".join(lines)

    def send_pick(self, decision: Dict[str, Any]) -> bool:
        """Send a formatted pick message. Only for BET decisions above threshold."""
        if decision.get("decision") != "BET":
            logger.info("Not sending non-BET decision to Telegram")
            return False
        message = self.format_pick_message(decision)
        return self.send_message(message)

    def send_no_bet_summary(self, summary: str) -> bool:
        """Send a NO BET summary (informational, not a pick)."""
        message = f"\u26D4 AI BETTOR — NO BET\n\n{summary}\n\n\u26A0\uFE0F Probabilistic analysis. Tidak ada jaminan hasil."
        return self.send_message(message)


def get_telegram_notifier() -> TelegramNotifier:
    return TelegramNotifier()