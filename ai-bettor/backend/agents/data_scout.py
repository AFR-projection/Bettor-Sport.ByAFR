"""Data Scout Agent for AI Bettor.

Responsibilities:
- Fetch fixtures from The Odds API
- Fetch odds for matches
- Validate response data
- Check timestamps
- Check bookmaker coverage
- Check market coverage
- Check for empty data
- Normalize data
- Detect invalid data

Output is always structured JSON.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from backend.config import get_settings


THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "data_quality": self.data_quality,
            "available_markets": self.available_markets,
            "bookmakers": self.bookmakers,
            "warnings": self.warnings,
            "raw_match_data": self.raw_match_data,
            "normalized_data": self.normalized_data,
        }


class DataScout:
    """
    Data Scout Agent - responsible for fetching and validating match data.
    
    Does NOT generate or hallucinate data. Only fetches from official APIs
    and validates/normalizes what's received.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.settings = get_settings()
        self.api_key = self.settings.THE_ODDS_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })
    
    def scan_matches(
        self, 
        sports: str = "soccer",
        regions: str = "idf",
        markets: str = "1X2,HT/FT,OverUnder2.5",
        odds_format: str = "decimal",
    ) -> List[DataScoutResult]:
        """
        Scan matches from The Odds API.
        
        Fetches all available matches for the given parameters.
        Returns list of DataScoutResult for each match.
        """
        results = []
        
        if not self.api_key:
            self._add_warning_all(results, "NO_API_KEY")
            return results
        
        endpoint = f"{THE_ODDS_API_BASE}/sports/{sports}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "apiKey": self.api_key,
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    endpoint, 
                    params=params, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    processed = self._process_odds_data(data)
                    results.extend(processed)
                    break  # Success, no need to retry
                    
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    time.sleep(min(wait_time, 30))
                    continue
                    
                else:
                    self._add_warning_all(results, f"API_ERROR:{response.status_code}")
                    break
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self._add_warning_all(results, "API_TIMEOUT")
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self._add_warning_all(results, f"REQUEST_ERROR:{str(e)}")
        
        return results
    
    def _process_odds_data(self, data: List[Dict]) -> List[DataScoutResult]:
        """Process raw API response into structured results."""
        results = []
        
        if not data:
            self._add_warning_all(results, "EMPTY_RESPONSE")
            return results
        
        for match_data in data:
            result = DataScoutResult()
            
            # Extract match ID
            result.match_id = match_data.get("id", "")
            if not result.match_id:
                self._add_warning(result, "NO_MATCH_ID")
                result.data_quality = 10
                results.append(result)
                continue
            
            # Store raw data
            result.raw_match_data = match_data
            
            # Extract teams
            teams = match_data.get("teams", [])
            if teams:
                result.normalized_data = {
                    "home_team": teams[0] if len(teams) > 0 else "UNKNOWN",
                    "away_team": teams[1] if len(teams) > 1 else "UNKNOWN",
                }
            
            # Extract odds and markets
            odds_data = match_data.get("odds", [])
            result.available_markets = self._extract_markets(odds_data)
            result.bookmakers = self._extract_bookmakers(odds_data)
            
            # Calculate data quality score
            quality_score = self._calculate_quality(odds_data, match_data)
            result.data_quality = quality_score
            
            # Collect warnings
            result.warnings = self._collect_warnings(match_data, odds_data)
            
            results.append(result)
        
        return results
    
    def _extract_markets(self, odds_data: List[Dict]) -> List[str]:
        """Extract market keys from odds data."""
        markets = set()
        for bookmaker in odds_data:
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                if market_key:
                    markets.add(market_key)
        return sorted(list(markets))
    
    def _extract_bookmakers(self, odds_data: List[Dict]) -> List[str]:
        """Extract bookmaker names from odds data."""
        bookies = set()
        for bookmaker in odds_data:
            name = bookmaker.get("name", "")
            if name:
                bookies.add(name)
        return sorted(list(bookies))
    
    def _calculate_quality(self, odds_data: List[Dict], match_data: Dict) -> int:
        """Calculate data quality score 0-100."""
        score = 100
        
        # Penalty: no odds data
        if not odds_data:
            score -= 40
        
        # Penalty: no bookmakers
        bookmakers = self._extract_bookmakers(odds_data)
        if not bookmakers:
            score -= 30
        
        # Penalty: limited markets
        markets = self._extract_markets(odds_data)
        required = {"1X2", "HDP", "OU"}
        if not required.intersection(markets):
            score -= 20
        
        # Check for valid odds values
        valid_odds_count = 0
        total_odds_check = 0
        for bookmaker in odds_data:
            for market in bookmaker.get("markets", []):
                for selection in market.get("selections", []):
                    odd = selection.get("odd", 0)
                    total_odds_check += 1
                    if odd and odd > 1:
                        valid_odds_count += 1
        
        if total_odds_check > 0 and valid_odds_count / total_odds_check < 0.5:
            score -= 15
        
        return max(0, min(100, score))
    
    def _collect_warnings(self, match_data: Dict, odds_data: List[Dict]) -> List[str]:
        """Collect warnings about data quality."""
        warnings = []
        
        # Check timestamp
        last_update = match_data.get("last_update")
        if not last_update:
            warnings.append("NO_TIMESTAMP")
        
        # Check for empty selections
        total_selections = 0
        empty_selections = 0
        for bookmaker in odds_data:
            for market in bookmaker.get("markets", []):
                for selection in market.get("selections", []):
                    total_selections += 1
                    if not selection.get("odd"):
                        empty_selections += 1
        
        if total_selections > 0 and empty_selections / total_selections > 0.3:
            warnings.append("HIGH_EMPTY_SELECTIONS")
        
        # Check bookmaker coverage
        bookmakers = self._extract_bookmakers(odds_data)
        if len(bookmakers) < 3:
            warnings.append("LOW_BOOKMAKER_COVERAGE")
        
        return warnings
    
    def _add_warning(self, result: DataScoutResult, warning: str):
        """Add a warning to a result."""
        result.warnings.append(warning)
        # Deduct from quality
        if result.data_quality > 0:
            result.data_quality -= 10
    
    def _add_warning_all(self, results: List[DataScoutResult], warning: str):
        """Add warning to all results (typically for API-level errors)."""
        for result in results:
            result.warnings.append(warning)


def get_data_scout(timeout: int = 30, max_retries: int = 3) -> DataScout:
    """Factory function to create DataScout instance."""
    return DataScout(timeout=timeout, max_retries=max_retries)