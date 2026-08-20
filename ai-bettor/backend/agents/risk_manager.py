"""Risk Manager Agent for AI Bettor.

The Risk Manager is the last line of defence before the Bettor Brain and it can
veto a bet that every other agent likes. It evaluates:

- data quality (bookmaker coverage, market coverage, odds validity)
- edge / EV sufficiency
- price stability across books (is the market disagreeing with itself?)
- bankroll exposure and drawdown
- correlation with what is already staked
- overall uncertainty

A veto always produces NO BET.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ai-bettor.risk_manager")


def _field(source: Any, name: str, default: Any) -> Any:
    """Read an attribute from an agent result object or a plain dict."""
    if isinstance(source, dict):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    return default if value is None else value



class RiskManagerResult:
    """Structured output from Risk Manager agent."""

    def __init__(self):
        self.risk_level: str = "UNKNOWN"
        self.veto_decision: bool = False
        self.veto_reason: str = ""

        # Detailed evaluations
        self.data_quality_score: float = 100.0
        self.edge_sufficient: bool = False
        self.ev_sufficient: bool = False
        self.odds_stable: bool = False
        self.exposure_concern: bool = False
        self.drawdown_concern: bool = False
        self.correlation_concern: bool = False

        # Quantified
        self.uncertainty_score: float = 0.0
        self.risk_adjusted_edge: float = 0.0
        self.exposure_percent: float = 0.0
        self.drawdown_percent: float = 0.0
        self.recommended_max_stake: float = 0.0

        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "veto_decision": self.veto_decision,
            "veto_reason": self.veto_reason,
            "data_quality_score": self.data_quality_score,
            "edge_sufficient": self.edge_sufficient,
            "ev_sufficient": self.ev_sufficient,
            "odds_stable": self.odds_stable,
            "exposure_concern": self.exposure_concern,
            "drawdown_concern": self.drawdown_concern,
            "correlation_concern": self.correlation_concern,
            "uncertainty_score": self.uncertainty_score,
            "risk_adjusted_edge": self.risk_adjusted_edge,
            "exposure_percent": self.exposure_percent,
            "drawdown_percent": self.drawdown_percent,
            "recommended_max_stake": self.recommended_max_stake,
            "warnings": self.warnings,
        }


class RiskManager:
    """Assesses risk and can veto a recommendation."""

    def __init__(
        self,
        min_data_quality: int = 50,
        max_uncertainty: float = 0.5,
        max_correlation_risk: float = 0.3,
        veto_edge_threshold: float = 0.01,
        veto_ev_threshold: float = 0.01,
        max_exposure_percent: float = 20.0,
        max_drawdown_percent: float = 15.0,
        max_stake_percent: float = 2.0,
    ):
        self.min_data_quality = min_data_quality
        self.max_uncertainty = max_uncertainty
        self.max_correlation_risk = max_correlation_risk
        self.veto_edge_threshold = veto_edge_threshold
        self.veto_ev_threshold = veto_ev_threshold
        self.max_exposure_percent = max_exposure_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.max_stake_percent = max_stake_percent

    def assess(
        self,
        quant_result: Any,
        simulation_result: Any,
        odds_data: List[Dict[str, Any]],
        current_exposure: float = 0.0,
        current_drawdown: float = 0.0,
        bankroll: float = 1000.0,
        recent_results: Optional[List[str]] = None,
        data_quality: Optional[float] = None,
        market_dispersion: Optional[float] = None,
        book_count: Optional[int] = None,
        open_bets_same_match: int = 0,
    ) -> RiskManagerResult:
        """Assess risk for one betting opportunity and decide on a veto."""
        result = RiskManagerResult()

        edge = float(_field(quant_result, "edge", 0.0))
        ev = float(_field(quant_result, "ev", 0.0))
        sim_warnings = list(_field(simulation_result, "warnings", []))
        stability = float(_field(simulation_result, "stability", 1.0))

        # 1. Data quality — supplied by the scout/market when available.
        if data_quality is None:
            data_quality = self._assess_data_quality(odds_data, sim_warnings)
        result.data_quality_score = round(float(data_quality), 2)
        if result.data_quality_score < self.min_data_quality:
            result.warnings.append("POOR_DATA_QUALITY")

        # 2 & 3. Edge / EV sufficiency.
        result.edge_sufficient = edge > self.veto_edge_threshold
        if not result.edge_sufficient:
            result.warnings.append("INSUFFICIENT_EDGE")
        result.ev_sufficient = ev > self.veto_ev_threshold
        if not result.ev_sufficient:
            result.warnings.append("INSUFFICIENT_EV")

        # 4. Price stability across bookmakers.
        result.odds_stable = self._check_odds_stability(odds_data, market_dispersion)
        if not result.odds_stable:
            result.warnings.append("ODDS_VOLATILITY")

        # 5 & 6. Bankroll exposure and drawdown.
        exposure_pct = (current_exposure / bankroll * 100.0) if bankroll > 0 else 0.0
        drawdown_pct = (current_drawdown / bankroll * 100.0) if bankroll > 0 else 0.0
        result.exposure_percent = round(exposure_pct, 2)
        result.drawdown_percent = round(drawdown_pct, 2)
        result.exposure_concern = exposure_pct > self.max_exposure_percent
        if result.exposure_concern:
            result.warnings.append("HIGH_EXPOSURE")
        result.drawdown_concern = drawdown_pct > self.max_drawdown_percent
        if result.drawdown_concern:
            result.warnings.append("HIGH_DRAWDOWN")

        # 7. Uncertainty score (0-1).
        uncertainty_factors = [
            not result.edge_sufficient,
            not result.ev_sufficient,
            not result.odds_stable,
            result.data_quality_score < self.min_data_quality,
            exposure_pct > self.max_exposure_percent * 0.75,
            drawdown_pct > self.max_drawdown_percent * 0.66,
            stability < 0.5,
            (book_count is not None and book_count < 3),
        ]
        result.uncertainty_score = round(sum(1 for f in uncertainty_factors if f) / len(uncertainty_factors), 4)

        # 8. Correlation: already exposed to the same match, or a losing streak
        #    while carrying open risk.
        result.correlation_concern = bool(
            open_bets_same_match > 0
            or (exposure_pct > self.max_exposure_percent * 0.5 and len(recent_results or []) > 3)
        )
        if result.correlation_concern:
            result.warnings.append("CORRELATION_RISK")

        # 9. Risk-adjusted edge: shrink the edge by the uncertainty we carry.
        result.risk_adjusted_edge = round(edge * (1.0 - result.uncertainty_score), 6)

        # 10. Risk level.
        risk_score = (
            result.uncertainty_score * 50
            + (0 if result.edge_sufficient else 20)
            + (0 if result.ev_sufficient else 20)
            + (0 if result.odds_stable else 10)
        )
        if risk_score >= 70:
            result.risk_level = "HIGH"
        elif risk_score >= 40:
            result.risk_level = "MEDIUM"
        else:
            result.risk_level = "LOW"

        # 11. Stake ceiling this bet must respect.
        stake_cap = bankroll * (self.max_stake_percent / 100.0)
        if result.risk_level == "HIGH":
            stake_cap *= 0.4
        elif result.risk_level == "MEDIUM":
            stake_cap *= 0.7
        result.recommended_max_stake = round(max(0.0, stake_cap), 2)

        # 12. Veto decision.
        veto_conditions: List[str] = []
        if result.data_quality_score < self.min_data_quality:
            veto_conditions.append("POOR_DATA_QUALITY")
        if not result.edge_sufficient:
            veto_conditions.append("INSUFFICIENT_EDGE")
        if not result.ev_sufficient:
            veto_conditions.append("INSUFFICIENT_EV")
        if result.uncertainty_score > self.max_uncertainty:
            veto_conditions.append("HIGH_UNCERTAINTY")
        if result.exposure_concern:
            veto_conditions.append("HIGH_EXPOSURE")
        if result.drawdown_concern:
            veto_conditions.append("HIGH_DRAWDOWN")
        if open_bets_same_match > 0:
            veto_conditions.append("ALREADY_EXPOSED_TO_MATCH")

        result.veto_decision = bool(veto_conditions)
        result.veto_reason = "; ".join(veto_conditions)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assess_data_quality(self, odds_data: List[Dict[str, Any]], sim_warnings: List[str]) -> int:
        """Fallback data-quality estimate derived from the odds themselves."""
        if not odds_data:
            return 0

        score = 100
        books = {str(b.get("name") or b.get("key") or "") for b in odds_data if isinstance(b, dict)}
        books.discard("")
        if len(books) == 0:
            return 0
        if len(books) == 1:
            score -= 20
        elif len(books) == 2:
            score -= 10

        markets = set()
        total = valid = 0
        for book in odds_data:
            if not isinstance(book, dict):
                continue
            for market in book.get("markets", []) or []:
                markets.add(str(market.get("key") or ""))
                for selection in market.get("selections", []) or []:
                    total += 1
                    odd = selection.get("odd")
                    if isinstance(odd, (int, float)) and odd > 1:
                        valid += 1
        if total == 0:
            return 0
        if valid / total < 0.9:
            score -= 20
        if "1X2" not in markets:
            score -= 10
        score -= min(15, 5 * len(sim_warnings))
        return max(0, min(100, score))

    @staticmethod
    def _check_odds_stability(
        odds_data: List[Dict[str, Any]],
        market_dispersion: Optional[float] = None,
    ) -> bool:
        """Stable = books broadly agree on the fair probability.

        When the market analyst supplied a de-vigged dispersion we use it (that
        is the statistically correct measure). Otherwise fall back to comparing
        prices *within each selection* — comparing prices across different
        selections, as the previous implementation did, always looked volatile.
        """
        if market_dispersion is not None:
            return market_dispersion <= 0.05

        by_selection: Dict[str, List[float]] = {}
        for book in odds_data or []:
            if not isinstance(book, dict):
                continue
            for market in book.get("markets", []) or []:
                for selection in market.get("selections", []) or []:
                    odd = selection.get("odd")
                    if not isinstance(odd, (int, float)) or odd <= 1:
                        continue
                    key = f"{market.get('key')}|{selection.get('point', market.get('point'))}|{selection.get('name')}"
                    by_selection.setdefault(key, []).append(float(odd))

        comparable = [values for values in by_selection.values() if len(values) >= 2]
        if not comparable:
            return False
        for values in comparable:
            mean = statistics.fmean(values)
            if mean <= 0:
                return False
            if (statistics.pstdev(values) / mean) > 0.08:
                return False
        return True

    def quick_assess(
        self,
        quant_result: Any,
        simulation_result: Any,
        odds_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Quick risk assessment."""
        return self.assess(quant_result, simulation_result, odds_data).to_dict()


def get_risk_manager(
    min_data_quality: int = 50,
    max_uncertainty: float = 0.5,
    max_correlation_risk: float = 0.3,
    veto_edge_threshold: float = 0.01,
    veto_ev_threshold: float = 0.01,
    max_exposure_percent: float = 20.0,
    max_drawdown_percent: float = 15.0,
    max_stake_percent: float = 2.0,
) -> RiskManager:
    """Factory function to create a RiskManager instance."""
    return RiskManager(
        min_data_quality=min_data_quality,
        max_uncertainty=max_uncertainty,
        max_correlation_risk=max_correlation_risk,
        veto_edge_threshold=veto_edge_threshold,
        veto_ev_threshold=veto_ev_threshold,
        max_exposure_percent=max_exposure_percent,
        max_drawdown_percent=max_drawdown_percent,
        max_stake_percent=max_stake_percent,
    )
