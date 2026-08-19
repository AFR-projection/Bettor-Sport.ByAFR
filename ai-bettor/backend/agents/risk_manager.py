"""Risk Manager Agent for AI Bettor.

Responsibilities:
- Evaluate uncertainty
- Evaluate data quality
- Evaluate odds
- Evaluate market movement
- Evaluate exposure
- Evaluate drawdown
- Evaluate correlation
- Provide risk level
- Can veto recommendations

Risk Manager can decide NO BET even if other agents recommend BET.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.models.probability_engine import DataQualityChecker


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
            "warnings": self.warnings,
        }


class RiskManager:
    """
    Risk Manager Agent - responsible for risk assessment.
    
    Can veto recommendations from other agents.
    Returns NO BET if risk is too high.
    """
    
    def __init__(
        self,
        min_data_quality: int = 50,
        max_uncertainty: float = 0.3,
        max_correlation_risk: float = 0.3,
        veto_edge_threshold: float = 0.01,
        veto_ev_threshold: float = 0.01,
    ):
        self.min_data_quality = min_data_quality
        self.max_uncertainty = max_uncertainty
        self.max_correlation_risk = max_correlation_risk
        self.veto_edge_threshold = veto_edge_threshold
        self.veto_ev_threshold = veto_ev_threshold
        self.quality_checker = DataQualityChecker()
    
    def assess(self,
                quant_result: "QuantAnalystResult",
                simulation_result: "SimulationAnalystResult",
                odds_data: List[Dict[str, Any]],
                current_exposure: float = 0.0,
                current_drawdown: float = 0.0,
                bankroll: float = 1000.0,
                recent_results: Optional[List[str]] = None) -> RiskManagerResult:
        """
        Assess risk for a betting opportunity.
        
        Evaluates multiple factors and can issue a veto.
        Returns risk level and whether to veto.
        """
        result = RiskManagerResult()
        
        # 1. Data quality assessment
        data_quality = self._assess_data_quality(
            quant_result, simulation_result, odds_data
        )
        result.data_quality_score = data_quality
        
        if data_quality < self.min_data_quality:
            result.warnings.append("POOR_DATA_QUALITY")
        
        # 2. Edge sufficiency
        edge_sufficient = quant_result.edge > self.veto_edge_threshold
        result.edge_sufficient = edge_sufficient
        
        if not edge_sufficient:
            result.warnings.append("INSUFFICIENT_EDGE")
        
        # 3. EV sufficiency
        ev_sufficient = quant_result.ev > self.veto_ev_threshold
        result.ev_sufficient = ev_sufficient
        
        if not ev_sufficient:
            result.warnings.append("INSUFFICIENT_EV")
        
        # 4. Odds stability
        odds_stable = self._check_odds_stability(odds_data)
        result.odds_stable = odds_stable
        
        if not odds_stable:
            result.warnings.append("ODDS_VOLATILITY")
        
        # 5. Exposure check
        exposure_pct = (current_exposure / bankroll) * 100 if bankroll > 0 else 0
        result.exposure_concern = exposure_pct > 20  # >20% of bankroll is concerning
        
        if result.exposure_concern:
            result.warnings.append("HIGH_EXPOSURE")
        
        # 6. Drawdown check
        drawdown_pct = (current_drawdown / bankroll) * 100 if bankroll > 0 else 0
        result.drawdown_concern = drawdown_pct > 15  # >15% drawdown is concerning
        
        if result.drawdown_concern:
            result.warnings.append("HIGH_DRAWDOWN")
        
        # 7. Uncertainty score
        uncertainty_factors = 0
        
        if not edge_sufficient:
            uncertainty_factors += 1
        if not ev_sufficient:
            uncertainty_factors += 1
        if not odds_stable:
            uncertainty_factors += 1
        if data_quality < 50:
            uncertainty_factors += 1
        if exposure_pct > 15:
            uncertainty_factors += 1
        if drawdown_pct > 10:
            uncertainty_factors += 1
        
        self_uncertainty = uncertainty_factors / 6.0  # Normalize to 0-1
        result.uncertainty_score = round(self_uncertainty, 4)
        
        # 8. Correlation check (simplified)
        # Check if this bet is correlated with existing exposure
        result.correlation_concern = (
            exposure_pct > 10 and len(recent_results or []) > 3
        )
        if result.correlation_concern:
            result.warnings.append("CORRELATION_RISK")
        
        # 9. Determine risk level
        risk_score = (
            self_uncertainty * 50 +  # 50 points max for uncertainty
            (1 if not edge_sufficient else 0) * 20 +  # 20 max for edge
            (1 if not ev_sufficient else 0) * 20 +  # 20 max for EV
            (1 if not odds_stable else 0) * 10  # 10 for odds stability
        )
        
        risk_level_pct = min(100, risk_score)
        
        if risk_level_pct >= 70:
            result.risk_level = "HIGH"
        elif risk_level_pct >= 40:
            result.risk_level = "MEDIUM"
        else:
            result.risk_level = "LOW"
        
        # 10. Veto decision
        # Veto if:
        # - Data quality too poor
        # - Insufficient edge
        # - Insufficient EV  
        # - High uncertainty
        # - High exposure
        # - High drawdown
        
        veto_conditions = []
        
        if data_quality < self.min_data_quality:
            veto_conditions.append("POOR_DATA_QUALITY")
        
        if not edge_sufficient:
            veto_conditions.append("INSUFFICIENT_EDGE")
        
        if not ev_sufficient:
            veto_conditions.append("INSUFFICIENT_EV")
        
        if self_uncertainty > self.max_uncertainty:
            veto_conditions.append("HIGH_UNCERTAINTY")
        
        if result.exposure_concern:
            veto_conditions.append("HIGH_EXPOSURE")
        
        if result.drawdown_concern:
            veto_conditions.append("HIGH_DRAWDOWN")
        
        result.veto_decision = len(veto_conditions) > 0
        result.veto_reason = "; ".join(veto_conditions) if veto_conditions else ""
        
        return result
    
    def _assess_data_quality(self,
                             quant_result: "QuantAnalystResult",
                             simulation_result: "SimulationAnalystResult",
                             odds_data: List[Dict[str, Any]]) -> int:
        """Assess data quality score 0-100."""
        score = 100
        
        # Edge too small = lower quality
        if quant_result.edge < 0.01:
            score -= 30
        
        # EV negative or very small
        if quant_result.ev < 0.01:
            score -= 25
        
        # Simulation warnings
        if simulation_result.warnings:
            score -= 15 * min(3, len(simulation_result.warnings))
        
        # Odds data quality
        if odds_data:
            total_odds = 0
            valid_odds = 0
            for bookie in odds_data:
                for market in bookie.get("markets", []):
                    for selection in market.get("selections", []):
                        odd = selection.get("odd", 0)
                        total_odds += 1
                        if odd and odd > 1:
                            valid_odds += 1
            
            if total_odds > 0 and valid_odds / total_odds < 0.6:
                score -= 20
        
        return max(0, min(100, score))
    
    def _check_odds_stability(self, odds_data: List[Dict[str, Any]]) -> bool:
        """Check if odds are stable (not extremely volatile)."""
        if not odds_data:
            return False
        
        all_odds = []
        for bookie in odds_data:
            for market in bookie.get("markets", []):
                for selection in market.get("selections", []):
                    odd = selection.get("odd", 0)
                    if odd and odd > 1:
                        all_odds.append(odd)
        
        if len(all_odds) < 3:
            return False
        
        import numpy as np
        odds_arr = np.array(all_odds)
        mean_odd = np.mean(odds_arr)
        std_odd = np.std(odds_arr)
        
        if mean_odd == 0:
            return False
        
        cv = std_odd / mean_odd  # Coefficient of variation
        
        # CV > 0.2 indicates high volatility for odds
        return cv < 0.2
    
    def quick_assess(self,
                     quant_result: Dict[str, Any],
                     simulation_result: Dict[str, Any],
                     odds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Quick risk assessment."""
        # Need model instance for full assessment
        # This is a simplified version
        result = self.assess(
            quant_result=quant_result,
            simulation_result=simulation_result,
            odds_data=odds_data,
        )
        return result.to_dict()


def get_risk_manager(
    min_data_quality: int = 50,
    max_uncertainty: float = 0.3,
    max_correlation_risk: float = 0.3,
    veto_edge_threshold: float = 0.01,
    veto_ev_threshold: float = 0.01,
) -> RiskManager:
    """Factory function to create RiskManager instance."""
    return RiskManager(
        min_data_quality=min_data_quality,
        max_uncertainty=max_uncertainty,
        max_correlation_risk=max_correlation_risk,
        veto_edge_threshold=veto_edge_threshold,
        veto_ev_threshold=veto_ev_threshold,
    )