# """Scanner service: fetches and processes matches."""
from __future__ import annotations; import asyncio, logging
from datetime import datetime, timezone; from typing import Optional
from backend.integrations.odds_api import OddsAPIClient
from backend.agents.data_scout import DataScoutAgent, ScoutResult; logger = logging.getLogger("scanner")

class ScannerService:
    def __init__(self): self.api = OddsAPIClient(); self.scout = DataScoutAgent()
    async def scan_sport(self, sport_key: str) -> list:
        events = await self.api.fetch_events(sport_key); results = []
        for event in events:
            eid = event.get("id","")
            if not eid: continue
            odd_data = event.get("bookmakers",[]); flat = self._flat(eid, odd_data)
            sr = self.scout.scout(event, flat)
            results.append({"match_id":eid,"home":event.get("home_team",""),"away":event.get("away_team",""),
                "commence":event.get("commence_time",""),"sport":sport_key,"odds_count":len(flat),
                "scout":{"data_quality":sr.data_quality,"markets":sr.available_markets,"warnings":sr.warnings,"status":sr.status,"books":sr.bookmakers},"raw":event})
        return results
    def _flat(self, mid, bms):
        flat=[]
        for bm in bms:
            bn=bm.get("title",""); ms=bm.get("markets",[])
            for m in ms:
                k=m.get("key",""); ln=m.get("point")
                for o in m.get("outcomes",[]):
                    flat.append({"match_id":mid,"bookmaker":bn,"market":k,"selection":o.get("name",""),"odds":o.get("price",0),"line":ln,"dq":100})
        return flat
