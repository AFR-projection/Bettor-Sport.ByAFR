"""Quant Analyst Agent for AI Bettor.

Responsibilities:
- Calculate implied probability from odds
- Analyze probabilities
- Calculate expected value
- Calculate edge
- Provide quantitative score
- Compare model probability with market probability

Uses real data from backend - LLM is not the source of probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.models.probability_engine import (
    DataQualityChecker,
    EVEngine,
    ImpliedProbabilityEngine,
    ProbabilityEnsemble,
)


class QuantAnalystResult:
    """Structured output from Quant Analyst agent."""
    
    def __init__(self):
        self.model_probability: float = 0.0  # Our calculated probability
        self.market_probability: float = 0.0  # Implied from odds
        self.edge: float = 0.0  # model - market
        self.ev: float = 0.0  # Expected value per unit
        self.confidence_score: int = 0  # 0-100
        self.risk_level: str = "UNKNOWN"
        self.recommendation: str = "NO BET"
        
        # Detailed metrics
        self.implied_probability: float = 0.0
        self.net_profit: float = 0.0
        self.probability_difference: float = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_probability": self.model_probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "ev": self.ev,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "implied_probability": self.implied_probability,
            "net_profit": self.net_profit,
            "probability_difference": self.probability_difference,
        }


class QuantAnalyst:
    """
    Quant Analyst Agent - responsible for quantitative analysis.
    
    Uses statistical models and real data. LLM is NOT the source of
    probabilities - this agent uses the probability engine.
    """
    
    def __init__(self, 
                 min_edge: float = 0.01,
                 min_ev: float = 0.01,
                 min_confidence: int = 60):
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.probability_engine = ProbabilityEnsemble()
        self.ev_engine = EVEngine()
        self.quality_checker = DataQualityChecker()
    
    def analyze(self, 
                match_id: str,
                model_prob: float,
                decimal_odds: float,
                bookmaker_margin: Optional[float] = None) -> QuantAnalystResult:
        """
        Analyze a betting opportunity.
        
        Calculates:
        - Implied probability from odds
        - Edge (model prob - market prob)
        - Expected value
        - Confidence score
        - Risk level
        - Recommendation (BET/NO BET)
        """
        result = QuantAnalystResult()
        
        # Calculate market implied probability
        result.market_probability = ImpliedProbabilityEngine.decimal_to_implied(
            decimal_odds, bookmaker_margin
        )
        result.implied_probability = result.market_probability
        
        # Calculate model probability (clamp to valid range)
        model_prob = max(0.01, min(0.99, model_prob))
        result.model_probability = model_prob
        
        # Calculate edge
        result.edge = round(model_prob - result.market_probability, 6)
        
        # Calculate EV
        ev_result = self.ev_engine.calculate_ev(model_prob, decimal_odds)
        result.ev = ev_result["ev_per_unit"]
        result.net_profit = ev_result["net_profit"]
        
        # Calculate probability difference
        result.probability_difference = round(
            abs(model_prob - result.market_probability), 6
        )
        
        # Determine confidence score
        confidence = self._calculate_confidence(
            model_prob, result.market_probability, result.edge, result.ev
        )
        result.confidence_score = confidence
        
        # Determine risk level
        result.risk_level = self._calculate_risk(
            model_prob, result.market_probability, result.edge
        )
        
        # Make recommendation
        result.recommendation = self._make_recommendation(
            result.edge, result.ev, result.confidence_score
        )
        
        return result
    
    def _calculate_confidence(
        self, 
        model_prob: float, 
        market_prob: float, 
        edge: float, 
        ev: float
    ) -> int:
        """Calculate confidence score 0-100."""
        score = 50  # Base score
        
        # Edge contributes to confidence
        edge_margin = abs(edge) * 100  # Convert to percentage points
        score += min(20, edge_margin * 5)
        
        # EV contributes
        if ev > 0:
            ev_contribution = min(20, ev * 20)
            score += ev_contribution
        
        # Probability confidence (not too extreme)
        if 0.3 <= model_prob <= 0.7:
            score += 10
        
        # Market agreement (edge and market prob not too far apart)
        prob_diff = abs(model_prob - market_prob)
        if prob_diff <= 0.1:
            score += 10
        elif prob_diff <= 0.2:
            score += 5
        
        return max(0, min(100, round(score)))
    
    def _calculate_risk(
        self, 
        model_prob: float, 
        market_prob: float, 
        edge: float
    ) -> str:
        """Calculate risk level."""
        prob_diff = abs(model_prob - market_prob)
        
        if abs(edge) < 0.01:
            return "HIGH"  # Very small edge = high risk
        elif prob_diff > 0.2:
            return "HIGH"  # Major disagreement
        elif prob_diff > 0.1:
            return "MEDIUM"
        elif edge > 0.02:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _make_recommendation(
        self, 
        edge: float, 
        ev: float, 
        confidence: int
    ) -> str:
        """Make BET/NO BET recommendation."""
        # NO BET conditions
        if edge < self.min_edge:
            return "NO BET"
        if ev < self.min_ev:
            return "NO BET"
        if confidence < self.min_confidence:
            return "NO BET"
        
        # Positive conditions
        if edge > 0.05 and confidence >= 80:
            return "PREMIUM CANDIDATE"
        elif edge > 0.03 and confidence >= 70:
            return "BET CANDIDATE"
        elif edge > 0.01 and confidence >= 60:
            return "PASS"
        
        return "NO BET"
    
    def quick_analyze(
        self, 
        model_prob: float, 
        decimal_odds: float,
    ) -> Dict[str, Any]:
        """Quick analysis without full object construction."""
        result = self.analyze(
            match_id="quick",
            model_prob=model_prob,
            decimal_odds=decimal_odds,
        )
        return result.to_dict()


def get_quant_analyst(
    min_edge: float = 0.01,
    min_ev: float = 0.01,
    min_confidence: int = 60,
) -> QuantAnalyst:
    """Factory function to create QuantAnalyst instance."""
    return QuantAnalyst(
        min_edge=min_edge,
        min_ev=min_ev,
        min_confidence=min_confidence,
    )