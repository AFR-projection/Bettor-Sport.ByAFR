"""Market Analyst Agent for AI Bettor.

Responsibilities:
- Compare odds across bookmakers
- Find best available odds
- Look at market consensus
- Detect price differences
- Detect line movement
- Provide market confidence

Uses real odds data from backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.models.probability_engine import DataQualityChecker


class MarketAnalystResult:
    """Structured output from Market Analyst agent."""
    
    def __init__(self):
        self.best_odds: float = 0.0
        self.best_bookmaker: str = ""
        self.market_consensus: Optional[float] = None
        self.price_difference: float = 0.0
        self.line_movement_detected: bool = False
        self.confidence: int = 0
        self.risk_level: str = "UNKNOWN"
        
        # Detailed
        self.all_odds: List[Dict[str, Any]] = []
        self.best_available: Dict[str, Any] = {}
        self.warnings: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_odds": self.best_odds,
            "best_bookmaker": self.best_bookmaker,
            "market_consensus": self.market_consensus,
            "price_difference": self.price_difference,
            "line_movement_detected": self.line_movement_detected,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "all_odds": self.all_odds,
            "best_available": self.best_available,
            "warnings": self.warnings,
        }


class MarketAnalyst:
    """
    Market Analyst Agent - responsible for market analysis.
    
    Compares odds across bookmakers, finds best price,
    and detects market movements.
    """
    
    def __init__(self):
        self.quality_checker = DataQualityChecker()
        self.price_difference = 0.0
    
    def analyze(self, 
                match_id: str,
                odds_data: List[Dict[str, Any]]) -> MarketAnalystResult:
        """
        Analyze market data for a match.
        
        Examines:
        - Odds from all bookmakers
        - Best available odds
        - Market consensus price
        - Price differences between bookies
        - Line movement indicators
        """
        result = MarketAnalystResult()
        
        if not odds_data:
            result.warnings.append("NO_ODDS_DATA")
            return result
        
        # Extract all odds by market and selection
        all_odds_entries = []
        
        for bookmaker_data in odds_data:
            bookie_name = bookmaker_data.get("name", "Unknown")
            markets = bookmaker_data.get("markets", [])
            
            for market in markets:
                market_key = market.get("key", "")
                selections = market.get("selections", [])
                
                for selection in selections:
                    selection_name = selection.get("name", "Unknown")
                    odd = selection.get("odd", 0)
                    line = selection.get("line", "")
                    
                    entry = {
                        "bookmaker": bookie_name,
                        "market": market_key,
                        "selection": selection_name,
                        "odd": odd,
                        "line": line,
                    }
                    all_odds_entries.append(entry)
        
        result.all_odds = all_odds_entries
        
        if not all_odds_entries:
            result.warnings.append("NO_SELECTIONS_FOUND")
            return result
        
        # Find best odds for each market type
        # Group by market key
        markets_by_key: Dict[str, List[Dict]] = {}
        for entry in all_odds_entries:
            mk = entry["market"]
            if mk not in markets_by_key:
                markets_by_key[mk] = []
            markets_by_key[mk].append(entry)
        
        # Find best odds for key markets (1X2, HDP, OU)
        best_odds_overall = 0.0
        best_bookmaker = ""
        consensus_values = []
        
        for market_key in ["1X2", "HDP", "OU"]:
            if market_key not in markets_by_key:
                continue
            
            market_entries = markets_by_key[market_key]
            market_odds = [e["odd"] for e in market_entries if e["odd"] and e["odd"] > 1]
            
            if not market_odds:
                continue
            
            best_for_market = max(market_odds)
            best_odds_overall = max(best_odds_overall, best_for_market)
            
            # Find which bookmaker offers the best odds
            best_entry = max(market_entries, key=lambda e: e["odd"] or 0)
            best_bookmaker = best_entry["bookmaker"]
            
            # Collect for consensus calculation
            consensus_values.extend(market_odds)
        
        result.best_odds = round(best_odds_overall, 4) if best_odds_overall > 0 else 0.0
        result.best_bookmaker = best_bookmaker
        
        # Calculate market consensus (average of best odds, or most common price)
        if consensus_values:
            # Weighted consensus: average of all valid odds
            valid_odds = [o for o in consensus_values if o and o > 1]
            if valid_odds:
                market_consensus = round(sum(valid_odds) / len(valid_odds), 4)
                # Also calculate implied probability consensus
                implied_avg = round(sum(1 / o for o in valid_odds) / len(valid_odds), 6)
                result.market_consensus = {
                    "average_odds": market_consensus,
                    "average_implied_probability": implied_avg,
                }
        
        # Detect price differences (spread between best and worst major bookie)
        all_valid_odds = [e["odd"] for e in all_odds_entries if e["odd"] and e["odd"] > 1]
        if len(all_valid_odds) >= 2:
            price_spread = round(max(all_valid_odds) - min(all_valid_odds), 4)
            self.price_difference = price_spread
            result.price_difference = price_spread
            
            # Large spread = more opportunity but also more variance
            if price_spread > 0.2:
                result.risk_level = "MEDIUM_HIGH"
            elif price_spread > 0.1:
                result.risk_level = "MEDIUM"
            else:
                result.risk_level = "LOW"
        
        # Detect line movement indicators
        # Check if there are significant differences in line offerings
        lines_seen = [e.get("line", "") for e in all_odds_entries if e.get("line")]
        if lines_seen:
            unique_lines = set(lines_seen)
            if len(unique_lines) >= 2:
                result.line_movement_detected = True
        
        # Confidence based on market quality
        confidence = 50
        
        # Best odds quality
        if best_odds_overall >= 1.90:
            confidence += 20
        elif best_odds_overall >= 1.80:
            confidence += 10
        
        # Bookmaker diversity
        unique_bookmakers = len(set(e["bookmaker"] for e in all_odds_entries))
        if unique_bookmakers >= 5:
            confidence += 10
        elif unique_bookmakers >= 3:
            confidence += 5
        
        # Price spread indicator
        if self.price_difference > 0.15:
            confidence -= 5  # High spread = more uncertainty
        
        result.confidence = max(0, min(100, confidence))
        
        # Warnings
        if not best_bookmaker:
            result.warnings.append("NO_VALID_BOOKMAKER")
        
        if best_odds_overall < 1.5:
            result.warnings.append("VERY_LOW_ODDS")
        
        if not consensus_values:
            result.warnings.append("NO_CONSENSUS_DATA")
        
        return result
    
    def quick_analyze(
        self, 
        odds_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Quick market analysis."""
        result = self.analyze(
            match_id="quick",
            odds_data=odds_data,
        )
        return result.to_dict()


def get_market_analyst() -> MarketAnalyst:
    """Factory function to create MarketAnalyst instance."""
    return MarketAnalyst()