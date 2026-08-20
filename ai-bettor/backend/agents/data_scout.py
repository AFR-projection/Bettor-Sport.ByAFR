"""Data Scout Agent for AI Bettor.

Talks to The Odds API v4 and turns the response into a canonical internal shape.
It never invents data: if the API returns nothing, the scout returns nothing.

Responsibilities:
- resolve a sport request ("soccer") into real API sport keys ("soccer_epl", ...)
- validate regions and markets against what the API actually accepts
- fetch odds through the multi-key router (automatic failover)
- record the API quota reported in the response headers
- validate + normalise + score data quality (0-100)

Canonical odds shape produced for downstream agents:

    raw_match_data["odds"] = [
        {
            "name": "Pinnacle",            # bookmaker title
            "key": "pinnacle",
            "last_update": "...",
            "markets": [
                {"key": "1X2", "api_key": "h2h", "point": None,
                 "selections": [{"name": "Home", "odd": 1.91, "point": None}, ...]},
            ],
        },
    ]

Market keys are canonicalised: h2h -> 1X2, spreads -> HDP, totals -> OU.
Selection names for 1X2 are canonicalised to Home / Draw / Away.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from backend.config import get_settings
from backend.integrations.odds_router import OddsApiRouter, get_odds_router

logger = logging.getLogger("ai-bettor.data_scout")

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

VALID_REGIONS = ("us", "us2", "uk", "eu", "au")

# The Odds API has no Indonesian/Asian region; those books are covered by eu/uk.
REGION_ALIASES = {
    "id": "eu,uk",
    "idf": "eu,uk",
    "idn": "eu,uk",
    "indonesia": "eu,uk",
    "asia": "eu,uk",
    "asian": "eu,uk",
    "world": "eu,uk,us",
    "all": "us,uk,eu,au",
}

# Canonical market key <-> The Odds API market key.
MARKET_TO_API = {"1X2": "h2h", "HDP": "spreads", "OU": "totals"}
API_TO_MARKET = {
    "h2h": "1X2",
    "h2h_3_way": "1X2",
    "spreads": "HDP",
    "totals": "OU",
    "outrights": "OUTRIGHT",
}
VALID_API_MARKETS = ("h2h", "spreads", "totals")

# Popular soccer leagues first when expanding a sport group, so a limited
# per-scan league budget is spent on liquid markets.
LEAGUE_PRIORITY = (
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_england_efl_cham",
    "soccer_usa_mls",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_indonesia_liga_1",
    "basketball_nba",
    "basketball_euroleague",
    "americanfootball_nfl",
    "baseball_mlb",
    "icehockey_nhl",
    "tennis_atp_aus_open_singles",
)

WIB = datetime.timezone(datetime.timedelta(hours=7))


class DataScoutResult:
    """Structured output from Data Scout agent."""

    def __init__(self):
        self.match_id: str = ""
        self.data_quality: int = 0  # 0-100 scale
        self.available_markets: List[str] = []
        self.bookmakers: List[str] = []
        self.warnings: List[str] = []
        self.raw_match_data: Optional[Dict] = None
        self.normalized_data: Optional[Dict] = None
        self.commence_time: Optional[str] = None  # UTC ISO from API
        self.kickoff_wib: Optional[str] = None    # Local Asia/Jakarta ISO
        self.sport_key: str = ""
        self.league: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "data_quality": self.data_quality,
            "available_markets": self.available_markets,
            "bookmakers": self.bookmakers,
            "warnings": self.warnings,
            "raw_match_data": self.raw_match_data,
            "normalized_data": self.normalized_data,
            "commence_time": self.commence_time,
            "kickoff_wib": self.kickoff_wib,
            "sport_key": self.sport_key,
            "league": self.league,
        }


def normalise_regions(regions: str) -> str:
    """Map user/legacy region input to regions the API accepts."""
    parts: List[str] = []
    for raw in str(regions or "").split(","):
        token = raw.strip().lower()
        if not token:
            continue
        expanded = REGION_ALIASES.get(token, token)
        for region in expanded.split(","):
            region = region.strip()
            if region in VALID_REGIONS and region not in parts:
                parts.append(region)
    return ",".join(parts) if parts else "eu,uk"


def normalise_markets(markets: str) -> str:
    """Map canonical/legacy market names to API market keys."""
    parts: List[str] = []
    for raw in str(markets or "").split(","):
        token = raw.strip()
        if not token:
            continue
        api_key = MARKET_TO_API.get(token.upper(), token.lower())
        # Unsupported legacy names (HT/FT, OverUnder2.5, ...) map to nothing.
        if api_key in ("overunder2.5", "over_under", "ou2.5"):
            api_key = "totals"
        if api_key in ("ht/ft", "htft", "correct_score"):
            continue
        if api_key in VALID_API_MARKETS and api_key not in parts:
            parts.append(api_key)
    return ",".join(parts) if parts else "h2h,spreads,totals"


def canonical_market(api_key: str) -> str:
    return API_TO_MARKET.get(str(api_key).lower(), str(api_key).upper())


def canonical_selection(name: str, market: str, home_team: str, away_team: str) -> str:
    """Turn API outcome names into stable selection labels."""
    text = (name or "").strip()
    if market in ("1X2", "HDP"):
        if text.lower() == "draw":
            return "Draw"
        if home_team and text.lower() == home_team.lower():
            return "Home"
        if away_team and text.lower() == away_team.lower():
            return "Away"
    if market == "OU":
        lowered = text.lower()
        if lowered.startswith("over"):
            return "Over"
        if lowered.startswith("under"):
            return "Under"
    return text or "UNKNOWN"


class DataScout:
    """Fetches and validates match data. Never fabricates odds."""

    def __init__(
        self,
        timeout: int = 20,
        max_retries: int = 3,
        router: Optional[OddsApiRouter] = None,
        settings_service: Optional[Any] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.settings = get_settings()
        self.api_key = self.settings.THE_ODDS_API_KEY
        self.router = router or get_odds_router()
        self._service = settings_service
        if self.api_key and not self.router.has_keys:
            self.router.add_key(self.api_key)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._sports_cache: Optional[List[Dict[str, Any]]] = None
        self._sports_cache_at: float = 0.0
        self.last_request_count: int = 0
        self.last_sport_keys: List[str] = []

    # ------------------------------------------------------------------
    # Settings access (live values, falling back to .env)
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        if self._service is None:
            try:
                from backend.services.settings_service import get_settings_service
                self._service = get_settings_service()
            except Exception:  # pragma: no cover - DB unavailable
                self._service = False
        if self._service:
            return self._service.get(key, default)
        return getattr(self.settings, key, default)

    @property
    def early_morning_only(self) -> bool:
        return bool(self._cfg("EARLY_MORNING_ONLY", self.settings.EARLY_MORNING_ONLY))

    @property
    def early_morning_end_hour(self) -> int:
        return int(self._cfg("EARLY_MORNING_END_HOUR", self.settings.EARLY_MORNING_END_HOUR))

    @property
    def early_morning_days(self) -> int:
        return int(self._cfg("EARLY_MORNING_DAYS", self.settings.EARLY_MORNING_DAYS))

    @property
    def max_leagues(self) -> int:
        return int(self._cfg("MAX_LEAGUES_PER_SCAN", self.settings.MAX_LEAGUES_PER_SCAN))

    @property
    def has_keys(self) -> bool:
        """Whether a scan can even be attempted. The pipeline asks this so an
        unconfigured key is reported as such instead of looking like a scan that
        found nothing."""
        return bool(self.router.has_keys)

    # ------------------------------------------------------------------
    # Sport resolution
    # ------------------------------------------------------------------

    def fetch_sports(self, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch the list of sports the API currently offers (cached 1 hour).

        The `/sports` endpoint is quota-free, so this call is deliberately not
        counted as a request against the key (and does not rotate the router).
        """
        if not force and self._sports_cache and (time.time() - self._sports_cache_at) < 3600:
            return self._sports_cache

        api_key = self.router.peek_key()
        if not api_key:
            return []
        try:
            response = self.session.get(
                f"{THE_ODDS_API_BASE}/sports/",
                params={"apiKey": api_key},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            self.router.report_failure(api_key, f"sports_list: {e}")
            return self._sports_cache or []

        self._record_quota(api_key, response)
        if response.status_code != 200:
            self.router.report_failure(api_key, f"sports_list HTTP {response.status_code}",
                                       response.status_code)
            return self._sports_cache or []

        try:
            data = response.json()
        except ValueError:
            return self._sports_cache or []
        if isinstance(data, list):
            self._sports_cache = data
            self._sports_cache_at = time.time()
            return data
        return []

    def resolve_sport_keys(self, sport: str) -> Tuple[List[str], List[str]]:
        """Resolve a sport request into concrete API sport keys.

        Returns (keys, warnings). "soccer" expands to in-season soccer leagues,
        "upcoming" is passed through, an explicit key is used as-is.
        """
        warnings: List[str] = []
        requested = (sport or "").strip().lower() or "soccer"
        if requested in ("upcoming", "all"):
            return ["upcoming"], warnings

        sports = self.fetch_sports()
        if not sports:
            # No catalogue available: 'upcoming' always works and costs 1 request.
            warnings.append("SPORTS_CATALOGUE_UNAVAILABLE")
            return ["upcoming"], warnings

        by_key = {str(s.get("key", "")): s for s in sports if s.get("key")}
        if requested in by_key:
            return [requested], warnings

        group_matches = [
            key for key, meta in by_key.items()
            if key.startswith(requested + "_") and meta.get("active") and not meta.get("has_outrights")
        ]
        if not group_matches:
            group_matches = [
                key for key, meta in by_key.items()
                if str(meta.get("group", "")).lower().replace(" ", "") == requested and meta.get("active")
            ]
        if not group_matches:
            warnings.append(f"UNKNOWN_SPORT:{requested}")
            return ["upcoming"], warnings

        priority = {key: i for i, key in enumerate(LEAGUE_PRIORITY)}
        group_matches.sort(key=lambda k: (priority.get(k, 500), k))
        limit = max(1, self.max_leagues)
        if len(group_matches) > limit:
            warnings.append(f"LEAGUES_TRUNCATED:{len(group_matches)}->{limit}")
        return group_matches[:limit], warnings

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_matches(
        self,
        sports: Optional[str] = None,
        regions: Optional[str] = None,
        markets: Optional[str] = None,
        odds_format: str = "decimal",
        early_morning_only: Optional[bool] = None,
    ) -> List[DataScoutResult]:
        """Scan matches + odds from The Odds API.

        Returns one DataScoutResult per match. Empty list means "no data" — the
        scout never fabricates fixtures.
        """
        results: List[DataScoutResult] = []
        self.last_request_count = 0

        if not self.router.has_keys:
            logger.warning("No The Odds API key configured — scan skipped")
            return results

        sport_request = sports or self._cfg("DEFAULT_SPORT", self.settings.DEFAULT_SPORT)
        region_param = normalise_regions(regions or self._cfg("DEFAULT_REGIONS", self.settings.DEFAULT_REGIONS))
        market_param = normalise_markets(markets or self._cfg("DEFAULT_MARKETS", self.settings.DEFAULT_MARKETS))
        if early_morning_only is None:
            early_morning_only = self.early_morning_only

        sport_keys, scan_warnings = self.resolve_sport_keys(sport_request)
        self.last_sport_keys = sport_keys
        logger.info(
            "Scanning %s sport key(s) [%s] regions=%s markets=%s",
            len(sport_keys), ", ".join(sport_keys), region_param, market_param,
        )

        seen_ids: set = set()
        for sport_key in sport_keys:
            payload = self._fetch_odds(sport_key, region_param, market_param, odds_format)
            if payload is None:
                continue
            for result in self._process_odds_data(payload):
                if not result.match_id or result.match_id in seen_ids:
                    continue
                seen_ids.add(result.match_id)
                result.warnings.extend(scan_warnings)
                results.append(result)

        if early_morning_only:
            results = self._filter_early_morning(results)
        return results

    def _fetch_odds(
        self,
        sport_key: str,
        regions: str,
        markets: str,
        odds_format: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch odds for one sport key, rotating keys on failure."""
        endpoint = f"{THE_ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        attempts_left = max(1, self.max_retries)
        tried_keys: set = set()

        while attempts_left > 0:
            api_key = self.router.get_key()
            if not api_key:
                logger.warning("All API keys unavailable (cooldown/disabled)")
                return None
            if api_key in tried_keys and len(tried_keys) >= self.router.key_count():
                time.sleep(1)
            tried_keys.add(api_key)
            attempts_left -= 1

            try:
                response = self.session.get(
                    endpoint, params={**params, "apiKey": api_key}, timeout=self.timeout
                )
            except requests.exceptions.Timeout:
                self.router.report_failure(api_key, "TIMEOUT")
                continue
            except requests.exceptions.RequestException as e:
                self.router.report_failure(api_key, f"REQUEST_ERROR: {e}")
                continue

            self.last_request_count += 1
            self._record_quota(api_key, response)
            status = response.status_code

            if status == 200:
                self.router.report_success(api_key, *self._quota(response))
                try:
                    data = response.json()
                except ValueError:
                    logger.error("Invalid JSON from %s", endpoint)
                    return None
                return data if isinstance(data, list) else []

            detail = self._error_detail(response)
            if status in (401, 403):
                self.router.report_failure(api_key, f"HTTP {status}: {detail}", status)
                logger.error("API key rejected (%s): %s", status, detail)
                continue
            if status == 429:
                self.router.report_failure(api_key, f"RATE_LIMITED: {detail}", status)
                logger.warning("Key rate limited, rotating")
                continue
            if status in (404, 422):
                # Bad sport/market/region combination - retrying will not help.
                self.router.report_success(api_key, *self._quota(response))
                logger.error("API rejected request for %s (%s): %s", sport_key, status, detail)
                return None
            if status >= 500:
                self.router.report_failure(api_key, f"HTTP {status}: {detail}", status)
                continue
            logger.error("Unexpected API status %s for %s: %s", status, sport_key, detail)
            return None

        logger.warning("Giving up on %s after %s attempt(s)", sport_key, self.max_retries)
        return None

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                return str(body.get("message") or body.get("error_code") or body)[:200]
        except Exception:
            pass
        try:
            return str(response.text)[:200]
        except Exception:
            return ""

    @staticmethod
    def _quota(response: Any) -> Tuple[Optional[int], Optional[int]]:
        headers = getattr(response, "headers", {}) or {}

        def _int(name: str) -> Optional[int]:
            try:
                return int(str(headers.get(name)).strip())
            except (TypeError, ValueError):
                return None

        return _int("x-requests-remaining"), _int("x-requests-used")

    def _record_quota(self, api_key: str, response: Any) -> None:
        remaining, used = self._quota(response)
        if remaining is not None:
            logger.info("Odds API quota: %s remaining, %s used", remaining, used)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _process_odds_data(self, data: List[Dict]) -> List[DataScoutResult]:
        """Turn the API payload into canonical results."""
        results: List[DataScoutResult] = []
        if not data:
            return results

        for match_data in data:
            if not isinstance(match_data, dict):
                continue
            result = DataScoutResult()
            result.match_id = str(match_data.get("id", "") or "")

            home_team, away_team = self._extract_teams(match_data)
            result.normalized_data = {
                "home_team": home_team,
                "away_team": away_team,
                "league": match_data.get("sport_title", "UNKNOWN"),
                "sport_key": match_data.get("sport_key", ""),
            }
            result.sport_key = str(match_data.get("sport_key", "") or "")
            result.league = str(match_data.get("sport_title", "") or "UNKNOWN")

            commence_time = match_data.get("commence_time")
            if commence_time:
                result.commence_time = str(commence_time)
                local = self._to_wib(result.commence_time)
                if local:
                    result.kickoff_wib = local.isoformat()

            odds = self._normalise_bookmakers(match_data, home_team, away_team)
            canonical = dict(match_data)
            canonical["odds"] = odds
            canonical["home_team"] = home_team
            canonical["away_team"] = away_team
            result.raw_match_data = canonical

            result.available_markets = self._extract_markets(odds)
            result.bookmakers = self._extract_bookmakers(odds)
            result.data_quality = self._calculate_quality(odds, match_data)
            result.warnings = self._collect_warnings(match_data, odds)
            results.append(result)

        return results

    @staticmethod
    def _extract_teams(match_data: Dict[str, Any]) -> Tuple[str, str]:
        """Support v4 (home_team/away_team) and the older teams[] shape."""
        home = str(match_data.get("home_team") or "").strip()
        away = str(match_data.get("away_team") or "").strip()
        if home and away:
            return home, away
        teams = match_data.get("teams") or []
        if isinstance(teams, list) and len(teams) >= 2:
            return str(teams[0]), str(teams[1])
        return home or "UNKNOWN", away or "UNKNOWN"

    def _normalise_bookmakers(
        self,
        match_data: Dict[str, Any],
        home_team: str,
        away_team: str,
    ) -> List[Dict[str, Any]]:
        """Build the canonical odds list from either API or legacy input."""
        books = match_data.get("bookmakers")
        if isinstance(books, list) and books:
            return [
                book for book in (
                    self._normalise_book(raw, home_team, away_team) for raw in books
                ) if book
            ]

        legacy = match_data.get("odds")
        if isinstance(legacy, list) and legacy:
            return [
                book for book in (
                    self._normalise_legacy_book(raw, home_team, away_team) for raw in legacy
                ) if book
            ]
        return []

    def _normalise_book(
        self,
        raw: Dict[str, Any],
        home_team: str,
        away_team: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        markets: List[Dict[str, Any]] = []
        for raw_market in raw.get("markets") or []:
            if not isinstance(raw_market, dict):
                continue
            api_key = str(raw_market.get("key", ""))
            market_key = canonical_market(api_key)
            selections: List[Dict[str, Any]] = []
            for outcome in raw_market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                odd = self._as_odd(outcome.get("price", outcome.get("odd")))
                if odd is None:
                    continue
                point = outcome.get("point")
                selections.append({
                    "name": canonical_selection(outcome.get("name", ""), market_key, home_team, away_team),
                    "raw_name": outcome.get("name", ""),
                    "odd": odd,
                    "point": point,
                })
            if not selections:
                continue
            points = {s.get("point") for s in selections if s.get("point") is not None}
            markets.append({
                "key": market_key,
                "api_key": api_key,
                "point": next(iter(points)) if len(points) == 1 else None,
                "selections": selections,
            })
        if not markets:
            return None
        return {
            "name": str(raw.get("title") or raw.get("name") or raw.get("key") or "UNKNOWN"),
            "key": str(raw.get("key") or raw.get("name") or "unknown"),
            "last_update": raw.get("last_update"),
            "markets": markets,
        }

    def _normalise_legacy_book(
        self,
        raw: Dict[str, Any],
        home_team: str,
        away_team: str,
    ) -> Optional[Dict[str, Any]]:
        """Accept the pre-existing {name, markets:[{key, selections:[{name, odd}]}]} shape."""
        if not isinstance(raw, dict):
            return None
        markets: List[Dict[str, Any]] = []
        for raw_market in raw.get("markets") or []:
            if not isinstance(raw_market, dict):
                continue
            market_key = canonical_market(raw_market.get("key", ""))
            selections = []
            for selection in raw_market.get("selections") or []:
                odd = self._as_odd(selection.get("odd", selection.get("price")))
                if odd is None:
                    continue
                selections.append({
                    "name": canonical_selection(selection.get("name", ""), market_key, home_team, away_team),
                    "raw_name": selection.get("name", ""),
                    "odd": odd,
                    "point": selection.get("point", raw_market.get("point")),
                })
            if selections:
                markets.append({
                    "key": market_key,
                    "api_key": MARKET_TO_API.get(market_key, market_key.lower()),
                    "point": raw_market.get("point"),
                    "selections": selections,
                })
        if not markets:
            return None
        return {
            "name": str(raw.get("name") or raw.get("title") or "UNKNOWN"),
            "key": str(raw.get("key") or raw.get("name") or "unknown"),
            "last_update": raw.get("last_update"),
            "markets": markets,
        }

    @staticmethod
    def _as_odd(value: Any) -> Optional[float]:
        try:
            odd = float(value)
        except (TypeError, ValueError):
            return None
        if odd <= 1.0 or odd > 1000:
            return None
        return odd

    @staticmethod
    def _extract_markets(odds_data: List[Dict]) -> List[str]:
        markets = set()
        for bookmaker in odds_data or []:
            for market in bookmaker.get("markets", []) or []:
                key = market.get("key", "")
                if key:
                    markets.add(key)
        return sorted(markets)

    @staticmethod
    def _extract_bookmakers(odds_data: List[Dict]) -> List[str]:
        books = set()
        for bookmaker in odds_data or []:
            name = bookmaker.get("name", "")
            if name:
                books.add(name)
        return sorted(books)

    def _calculate_quality(self, odds_data: List[Dict], match_data: Dict) -> int:
        """Data quality score 0-100 based on coverage and validity."""
        score = 100
        if not odds_data:
            return 10

        bookmakers = self._extract_bookmakers(odds_data)
        if len(bookmakers) == 0:
            return 10
        if len(bookmakers) < 3:
            score -= 25
        elif len(bookmakers) < 5:
            score -= 10

        markets = set(self._extract_markets(odds_data))
        if "1X2" not in markets:
            score -= 20
        if not markets.intersection({"HDP", "OU"}):
            score -= 10

        total = valid = 0
        for bookmaker in odds_data:
            for market in bookmaker.get("markets", []) or []:
                for selection in market.get("selections", []) or []:
                    total += 1
                    odd = selection.get("odd")
                    if isinstance(odd, (int, float)) and odd > 1:
                        valid += 1
        if total == 0:
            return 10
        if valid / total < 0.9:
            score -= 15

        if not match_data.get("commence_time"):
            score -= 15

        return max(0, min(100, score))

    def _collect_warnings(self, match_data: Dict, odds_data: List[Dict]) -> List[str]:
        warnings: List[str] = []
        if not match_data.get("id"):
            warnings.append("MISSING_MATCH_ID")
        if not odds_data:
            warnings.append("NO_ODDS_DATA")
        books = self._extract_bookmakers(odds_data)
        if len(books) < 3:
            warnings.append("LOW_BOOKMAKER_COVERAGE")
        markets = set(self._extract_markets(odds_data))
        if "1X2" not in markets:
            warnings.append("NO_1X2_MARKET")
        last_updates = [
            b.get("last_update") for b in odds_data or [] if b.get("last_update")
        ]
        if odds_data and not last_updates:
            warnings.append("NO_TIMESTAMP")
        else:
            stale = self._staleness_seconds(last_updates)
            if stale is not None and stale > 3600:
                warnings.append(f"STALE_ODDS:{int(stale)}s")
        if not match_data.get("commence_time"):
            warnings.append("NO_COMMENCE_TIME")
        return warnings

    @staticmethod
    def _staleness_seconds(timestamps: Iterable[Any]) -> Optional[float]:
        newest: Optional[datetime.datetime] = None
        for raw in timestamps:
            parsed = DataScout._parse_iso(raw)
            if parsed and (newest is None or parsed > newest):
                newest = parsed
        if newest is None:
            return None
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - newest).total_seconds()

    @staticmethod
    def _parse_iso(raw: Any) -> Optional[datetime.datetime]:
        if not raw:
            return None
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.datetime.fromisoformat(text)
        except (ValueError, TypeError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed

    # ------------------------------------------------------------------
    # Early-morning window filter (dini hari WIB)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_wib(commence_time: str) -> Optional[datetime.datetime]:
        """Parse an ISO timestamp and convert it to Asia/Jakarta (UTC+7)."""
        parsed = DataScout._parse_iso(commence_time)
        return parsed.astimezone(WIB) if parsed else None

    def _is_early_morning_window(self, commence_time: str) -> bool:
        """True when kickoff (WIB) is before the cut-off hour, today..+N days."""
        local = self._to_wib(commence_time)
        if local is None:
            return False
        now_wib = datetime.datetime.now(WIB)
        day_diff = (local.date() - now_wib.date()).days
        if day_diff < 0 or day_diff >= max(1, self.early_morning_days):
            return False
        return local.hour < self.early_morning_end_hour

    def _filter_early_morning(self, results: List[DataScoutResult]) -> List[DataScoutResult]:
        """Keep only matches kicking off in the early-morning WIB window."""
        filtered: List[DataScoutResult] = []
        for result in results:
            if result.commence_time and self._is_early_morning_window(result.commence_time):
                filtered.append(result)
            elif not result.commence_time:
                result.warnings.append("NO_COMMENCE_TIME_SKIPPED")
        return filtered


def get_data_scout(timeout: int = 20, max_retries: int = 3, **kwargs: Any) -> DataScout:
    """Factory function to create a DataScout instance."""
    return DataScout(timeout=timeout, max_retries=max_retries, **kwargs)


def test_odds_api_key(api_key: str, timeout: int = 15) -> Dict[str, Any]:
    """Validate one The Odds API key with a real (quota-free) API call."""
    result: Dict[str, Any] = {
        "success": False, "status_code": None, "message": "",
        "sport_count": 0, "requests_remaining": None, "requests_used": None,
    }
    if not api_key or not api_key.strip():
        result["message"] = "API key is empty"
        return result
    try:
        response = requests.get(
            f"{THE_ODDS_API_BASE}/sports/",
            params={"apiKey": api_key.strip()},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
    except requests.exceptions.Timeout:
        result["message"] = "Request timed out. Check internet connection."
        return result
    except requests.exceptions.RequestException as e:
        result["message"] = f"Network error: {e}"
        return result

    remaining, used = DataScout._quota(response)
    result["status_code"] = response.status_code
    result["requests_remaining"] = remaining
    result["requests_used"] = used

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            data = []
        result["success"] = True
        result["sport_count"] = len(data) if isinstance(data, list) else 0
        quota = f" Quota left: {remaining}." if remaining is not None else ""
        result["message"] = f"Valid key. {result['sport_count']} sports available.{quota}"
    elif response.status_code == 429:
        result["message"] = "Key is rate limited / quota exhausted (429)."
    elif response.status_code == 401:
        result["message"] = "Invalid key (401 Unauthorized)."
    elif response.status_code == 403:
        result["message"] = "Key forbidden (403). Check plan / access rights."
    else:
        result["message"] = f"Unexpected HTTP {response.status_code}: {DataScout._error_detail(response)}"
    return result
