# """The Odds API client."""
from __future__ import annotations; import httpx, time, logging, asyncio
from datetime import datetime, timezone; from typing import Optional
from backend.config.settings import settings

logger = logging.getLogger("odds_api")
ODDS_BASE = "https://api.the-odds-api.com/v4"

class OddsAPIClient:
    def __init__(self):
        self.api_key = settings.the_odds_api_key
        self._c = httpx.AsyncClient(base_url=ODDS_BASE, timeout=30.0, headers={"apikey": self.api_key})
    async def close(self): await self._c.aclose()

    async def fetch_sports(self):
        r = await self._g("/sports"); return r if isinstance(r,list) else []

    async def fetch_events(self, sport: str, regions=None):
        p = {"regions": regions or settings.odds_api_regions, "markets": settings.odds_api_markets, "dateFormat": "iso", "perPage": 100}
        r = await self._g(f"/sports/{sport}/events", p); return r if isinstance(r,list) else []

    async def fetch_history(self, sport: str, event_id: str, market: str):
        p = {"markets": market}; r = await self._g(f"/sports/{sport}/events/{event_id}/odds", p)
        return r if isinstance(r,list) else []

    async def _g(self, path, params=None):
        for att in range(3):
            try:
                resp = await self._c.get(path, params=params)
                if resp.status_code == 200: return resp.json()
                if resp.status_code == 429: await asyncio.sleep(2**att); continue
                logger.error("API err %d: %s", resp.status_code, resp.text[:200]); return []
            except httpx.TimeoutException: await asyncio.sleep(2**att)
            except Exception as e: logger.error("Req fail: %s", e); return []
        return []

_inst = None
def get_client(): global _inst; _inst = _inst or OddsAPIClient(); return _inst
