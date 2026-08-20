"""Market Analyst Agent for AI Bettor.

Responsibilities:
- compare odds across every bookmaker (line shopping)
- remove each book's overround and build a cross-book consensus fair price
- estimate expected total goals from the Over/Under market
- measure price dispersion, overround and bookmaker coverage
- surface the best available price per selection

The analyst only reports what the fetched odds say. It never invents a price.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.core.market_math import (
    MarketGroup,
    collect_market_groups,
    market_totals_estimate,
)

logger = logging.getLogger("ai-bettor.market_analyst")


class MarketAnalystResult:
    """Structured output from Market Analyst agent."""

    def __init__(self):
        # Legacy summary fields (kept stable for the /decide contract).
        self.best_odds: float = 0.0
        self.best_bookmaker: str = ""
        self.market_consensus: Optional[Dict[str, Any]] = None
        self.price_difference: float = 0.0
        self.line_movement_detected: bool = False
        self.confidence: int = 0
        self.risk_level: str = "UNKNOWN"
        self.all_odds: List[Dict[str, Any]] = []
        self.best_available: Dict[str, Any] = {}
        self.warnings: List[str] = []

        # Structured market view used by the pipeline.
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.fair_probabilities: Dict[str, float] = {}
        self.total_goals_estimate: Optional[float] = None
        self.average_overround: Optional[float] = None
        self.bookmaker_count: int = 0
        self.market_disagreement: float = 0.0
        self.markets_covered: List[str] = []

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
            "groups": self.groups,
            "fair_probabilities": self.fair_probabilities,
            "total_goals_estimate": self.total_goals_estimate,
            "average_overround": self.average_overround,
            "bookmaker_count": self.bookmaker_count,
            "market_disagreement": self.market_disagreement,
            "markets_covered": self.markets_covered,
        }


class MarketAnalyst:
    """Compares bookmakers, removes the vig and reports the fair market price."""

    def __init__(self, default_total_goals: float = 2.7):
        self.default_total_goals = default_total_goals
        self.price_difference = 0.0
        self._groups: Dict[str, MarketGroup] = {}

    @property
    def market_groups(self) -> Dict[str, MarketGroup]:
        """Groups from the most recent analyse call (used by the pipeline)."""
        return self._groups

    def analyze(self, match_id: str, odds_data: List[Dict[str, Any]]) -> MarketAnalystResult:
        result = MarketAnalystResult()
        self._groups = {}

        if not odds_data:
            result.warnings.append("NO_ODDS_DATA")
            return result

        entries: List[Dict[str, Any]] = []
        for book in odds_data:
            if not isinstance(book, dict):
                continue
            book_name = str(book.get("name") or book.get("key") or "Unknown")
            for market in book.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                market_key = str(market.get("key") or "")
                for selection in market.get("selections") or []:
                    if not isinstance(selection, dict):
                        continue
                    entries.append({
                        "bookmaker": book_name,
                        "market": market_key,
                        "selection": str(selection.get("name") or "Unknown"),
                        "odd": selection.get("odd", 0),
                        "line": selection.get("point", market.get("point")),
                    })
        result.all_odds = entries
        if not entries:
            result.warnings.append("NO_SELECTIONS_FOUND")
            return result

        self._groups = collect_market_groups(odds_data)

        # ---- structured consensus per market + line ----
        best_overall = 0.0
        best_book = ""
        dispersions: List[float] = []
        overrounds: List[float] = []
        for key, group in self._groups.items():
            consensus = group.consensus()
            selections: Dict[str, Any] = {}
            for selection in group.prices:
                best = group.best_price(selection)
                selections[selection] = {
                    "fair_probability": consensus["fair_probabilities"].get(selection),
                    "dispersion": consensus["dispersion"].get(selection, 0.0),
                    "best_odds": round(best[1], 4) if best else None,
                    "best_bookmaker": best[0] if best else None,
                    "book_count": group.book_count(selection),
                    "price_spread": group.price_spread(selection),
                }
                if best and best[1] > best_overall:
                    best_overall, best_book = best[1], best[0]
            if consensus["average_overround"] is not None:
                overrounds.append(consensus["average_overround"])
            dispersions.extend(v for v in consensus["dispersion"].values())
            result.groups[key] = {
                "market": group.market,
                "point": group.point,
                "label": group.label,
                "books_used": consensus["books_used"],
                "average_overround": consensus["average_overround"],
                "selections": selections,
            }

        result.best_odds = round(best_overall, 4) if best_overall > 0 else 0.0
        result.best_bookmaker = best_book
        result.best_available = {"odds": result.best_odds, "bookmaker": best_book}
        result.markets_covered = sorted({g.market for g in self._groups.values()})
        result.bookmaker_count = len({e["bookmaker"] for e in entries})
        result.average_overround = (
            round(sum(overrounds) / len(overrounds), 6) if overrounds else None
        )
        result.market_disagreement = round(max(dispersions), 6) if dispersions else 0.0

        # ---- 1X2 fair probabilities + implied total goals ----
        for group in self._groups.values():
            if group.market == "1X2":
                consensus = group.consensus()
                if consensus["fair_probabilities"]:
                    result.fair_probabilities = consensus["fair_probabilities"]
                break
        result.total_goals_estimate = market_totals_estimate(self._groups, self.default_total_goals)

        # ---- legacy aggregate consensus (average price across all quotes) ----
        valid_odds = [
            float(e["odd"]) for e in entries
            if isinstance(e["odd"], (int, float)) and e["odd"] > 1
        ]
        if valid_odds:
            result.market_consensus = {
                "average_odds": round(sum(valid_odds) / len(valid_odds), 4),
                "average_implied_probability": round(
                    sum(1 / o for o in valid_odds) / len(valid_odds), 6
                ),
                "quotes": len(valid_odds),
            }
            if len(valid_odds) >= 2:
                self.price_difference = round(max(valid_odds) - min(valid_odds), 4)
                result.price_difference = self.price_difference
        else:
            result.warnings.append("NO_CONSENSUS_DATA")

        # ---- risk / confidence from market structure ----
        result.risk_level = self._risk_level(result)
        result.confidence = self._confidence(result)

        # ---- line movement: several distinct lines quoted for one market ----
        lines_per_market: Dict[str, set] = {}
        for group in self._groups.values():
            if group.point is not None:
                lines_per_market.setdefault(group.market, set()).add(group.point)
        result.line_movement_detected = any(len(v) >= 2 for v in lines_per_market.values())

        if not best_book:
            result.warnings.append("NO_VALID_BOOKMAKER")
        if result.bookmaker_count < 3:
            result.warnings.append("LOW_BOOKMAKER_COVERAGE")
        if not result.fair_probabilities:
            result.warnings.append("NO_1X2_CONSENSUS")
        if result.average_overround is not None and result.average_overround > 0.12:
            result.warnings.append(f"HIGH_OVERROUND:{result.average_overround:.3f}")
        return result

    @staticmethod
    def _risk_level(result: MarketAnalystResult) -> str:
        """Wide prices / few books / fat margins = a less trustworthy market."""
        score = 0
        if result.bookmaker_count < 3:
            score += 2
        elif result.bookmaker_count < 5:
            score += 1
        if result.market_disagreement > 0.06:
            score += 2
        elif result.market_disagreement > 0.03:
            score += 1
        if result.average_overround is not None:
            if result.average_overround > 0.10:
                score += 2
            elif result.average_overround > 0.06:
                score += 1
        if score >= 4:
            return "HIGH"
        if score >= 3:
            return "MEDIUM_HIGH"
        if score >= 1:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _confidence(result: MarketAnalystResult) -> int:
        confidence = 45
        confidence += min(25, result.bookmaker_count * 4)
        if result.fair_probabilities:
            confidence += 10
        if result.total_goals_estimate:
            confidence += 5
        if result.average_overround is not None and result.average_overround <= 0.06:
            confidence += 10
        if result.market_disagreement > 0.06:
            confidence -= 10
        return max(0, min(100, confidence))

    def quick_analyze(self, odds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Quick market analysis."""
        return self.analyze(match_id="quick", odds_data=odds_data).to_dict()


def get_market_analyst(default_total_goals: float = 2.7) -> MarketAnalyst:
    """Factory function to create a MarketAnalyst instance."""
    return MarketAnalyst(default_total_goals=default_total_goals)
