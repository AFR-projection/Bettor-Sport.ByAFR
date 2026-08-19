"""OpenRouter LLM integration for AI Bettor.

LLM is used as a reasoning/decision-support layer ONLY.
LLM is NEVER the source of probabilities, simulations, or EV.
If LLM fails, the system continues with quantitative data only.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from backend.config import get_settings

logger = logging.getLogger("ai-bettor.openrouter")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class LLMError(Exception):
    """Raised when LLM call fails."""


class OpenRouterClient:
    """Client for OpenRouter LLM API with retry and graceful failure."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-bettor.local",
            "X-Title": "AI Bettor",
        })

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> str:
        """Send a completion request to OpenRouter with retry/backoff."""
        if not self.is_configured:
            raise LLMError("OpenRouter not configured")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)
                    logger.warning("OpenRouter rate limited, waiting %ss", wait)
                    time.sleep(wait)
                    continue
                else:
                    logger.error("OpenRouter error %s: %s", resp.status_code, resp.text[:300])
            except requests.exceptions.Timeout:
                logger.warning("OpenRouter timeout (attempt %s)", attempt + 1)
            except requests.exceptions.RequestException as e:
                logger.warning("OpenRouter request error: %s", e)
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        raise LLMError("OpenRouter call failed after retries")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Request JSON structured output and parse it safely."""
        content = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise LLMError("LLM returned invalid JSON")

    def analyze_match(self, match_context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM reasoning layer for a match. Returns structured review.

        The LLM does NOT compute probabilities - it reviews the
        quantitative results for sanity, reasoning quality, and
        decision confirmation.
        """
        system_prompt = (
            "You are a disciplined professional sports bettor's review layer. "
            "You NEVER invent data, odds, probabilities, or statistics. "
            "You only review the quantitative analysis provided to you. "
            "You must output JSON with keys: review, concerns (array), "
            "agrees_with_quant (bool). If data is missing or inconsistent, "
            "set agrees_with_quant to false. Never claim guaranteed wins."
        )
        user_prompt = (
            "Review this quantitative analysis for sanity:\n"
            f"{json.dumps(match_context, indent=2, default=str)}\n\n"
            "Return JSON only."
        )
        try:
            return self.complete_json(system_prompt, user_prompt)
        except LLMError as e:
            logger.warning("LLM review failed, continuing without it: %s", e)
            return {"review": "LLM_UNAVAILABLE", "concerns": [], "agrees_with_quant": True}


def get_llm_client() -> OpenRouterClient:
    return OpenRouterClient()