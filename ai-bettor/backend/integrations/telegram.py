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
        settings = get_settings()
        self.bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
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

    def format_pick_message(self, decision: Dict[str, Any]) -> str:
        """Format a BettorBrain decision into the standard pick message."""
        score = decision.get("score")
        header = "PREMIUM PICK" if (score or 0) >= 90 else "HIGH-CONFIDENCE PICK"
        lines = [
            f"\U0001F3C6 AI BETTOR — {header}",
            "",
            "\u26BD MATCH",
            f"{decision.get('home_team', 'Home')} vs {decision.get('away_team', 'Away')}",
            "",
            "\U0001F550 KICKOFF",
            f"{decision.get('kickoff', 'UNKNOWN')} (WIB)",
            "",
            "\U0001F3AF MARKET",
            f"{decision.get('market', 'UNKNOWN')}",
            "",
            "\U0001F4CC PICK",
            f"{decision.get('selection', 'UNKNOWN')}",
            "",
            "\U0001F4B0 ODDS",
            f"{decision.get('odds', 'N/A')}",
            "",
            "\U0001F4CA MODEL PROBABILITY",
            f"{decision.get('model_probability', 0) * 100:.1f}%",
            "",
            "\U0001F4C8 IMPLIED PROBABILITY",
            f"{decision.get('implied_probability', 0) * 100:.1f}%",
            "",
            "\U0001F48E EDGE",
            f"+{decision.get('edge', 0) * 100:.1f}%",
            "",
            "\U0001F4B0 EV",
            f"+{decision.get('ev', 0):.2f}",
            "",
            "\U0001F3B2 SIMULATION",
            f"{decision.get('simulation_count', 20000):,} runs (repeated batches)",
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
            decision.get('reasoning', '; '.join(decision.get('reasons', [])) or 'No reasoning available'),
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