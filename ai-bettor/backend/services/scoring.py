"""Pick scoring engine 0-100 for AI Bettor.

Scores betting candidates using multiple configurable factors:
- model probability
- EV
- edge
- simulation stability
- market agreement
- odds quality
- line movement
- data quality
- uncertainty
- risk
- bookmaker consensus

Thresholds (configurable, single source of truth):
0-59 = NO BET
60-69 = PASS
70-79 = WATCH
80-89 = BET CANDIDATE
90-100 = PREMIUM CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.config import get_settings


@dataclass
class ScoreWeights:
    """Configurable weights for scoring factors."""
    model_probability: float = 0.10
    ev: float = 0.20
    edge: float = 0.20
    simulation_stability: float = 0.10
    market_agreement: float = 0.10
    odds_quality: float = 0.10
    data_quality: float = 0.10
    risk: float = 0.10

    @classmethod
    def from_settings(cls) -> "ScoreWeights":
        return cls()

    def normalize(self) -> "ScoreWeights":
        total = sum(self.__dict__.values())
        if total <= 0:
            return self
        for key in self.__dict__:
            setattr(self, key, getattr(self, key) / total)
        return self


@dataclass
class ScoringThresholds:
    no_bet_max: int = 59
    pass_max: int = 69
    watch_max: int = 79
    bet_max: int = 89
    premium_min: int = 90

    def label(self, score: int) -> str:
        if score <= self.no_bet_max:
            return "NO BET"
        if score <= self.pass_max:
            return "PASS"
        if score <= self.watch_max:
            return "WATCH"
        if score <= self.bet_max:
            return "BET CANDIDATE"
        return "PREMIUM CANDIDATE"


class PickScoringEngine:
    """Scores a candidate pick 0-100 based on quantitative factors."""

    def __init__(self, weights: Optional[ScoreWeights] = None):
        settings = get_settings()
        self.weights = (weights or ScoreWeights()).normalize()
        self.thresholds = ScoringThresholds()

    def score(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Score a candidate pick. Returns score + per-factor breakdown.

        Expected candidate keys:
        model_probability, ev, edge, simulation_stability, variance,
        market_agreement, odds_quality, data_quality, risk_level,
        bookmaker_consensus
        """
        factors: Dict[str, float] = {}

        # 1. Model probability factor (0-100): peak at moderate-high prob
        mp = candidate.get("model_probability", 0)
        factors["model_probability"] = min(100.0, max(0.0, mp * 100 * 1.2))

        # 2. EV factor: scale EV per unit stake
        ev = candidate.get("ev", 0)
        factors["ev"] = min(100.0, max(0.0, ev * 200))

        # 3. Edge factor: edge percentage points
        edge = candidate.get("edge", 0)
        factors["edge"] = min(100.0, max(0.0, edge * 100 * 8))

        # 4. Simulation stability
        stability = candidate.get("simulation_stability", 0)
        factors["simulation_stability"] = min(100.0, max(0.0, stability * 100))

        # 5. Market agreement (lower prob disagreement = better)
        disagreement = candidate.get("market_disagreement", 0)
        factors["market_agreement"] = min(100.0, max(0.0, (1 - disagreement) * 100))

        # 6. Odds quality (odds above min acceptable)
        odds = candidate.get("odds", 0)
        min_odds = candidate.get("minimum_acceptable_odds", 1.5)
        if odds >= min_odds:
            factors["odds_quality"] = min(100.0, 60 + (odds - min_odds) * 100)
        else:
            factors["odds_quality"] = 20.0

        # 7. Data quality
        dq = candidate.get("data_quality", 50)
        factors["data_quality"] = min(100.0, max(0.0, float(dq)))

        # 8. Risk factor (lower risk = higher score)
        risk_map = {"LOW": 90, "MEDIUM": 65, "MEDIUM_HIGH": 45, "HIGH": 20, "UNKNOWN": 40}
        risk_level = candidate.get("risk_level", "UNKNOWN")
        factors["risk"] = risk_map.get(risk_level, 40)

        # Weighted sum
        score = sum(
            getattr(self.weights, key) * value
            for key, value in factors.items()
        )
        score = round(min(100.0, max(0.0, score)), 1)

        # Bookmaker consensus penalty (less consensus = deduction)
        consensus = candidate.get("bookmaker_consensus", 1.0)
        if consensus < 0.6:
            score -= 5
        elif consensus < 0.4:
            score -= 10
        score = round(min(100.0, max(0.0, score)), 1)

        return {
            "score": score,
            "label": self.thresholds.label(score),
            "factors": {k: round(v, 2) for k, v in factors.items()},
            "weights": self.weights.__dict__,
            "thresholds": {
                "no_bet": self.thresholds.no_bet_max,
                "pass": self.thresholds.pass_max,
                "watch": self.thresholds.watch_max,
                "bet": self.thresholds.bet_max,
                "premium": self.thresholds.premium_min,
            },
        }

    def is_bettable(self, score_result: Dict[str, Any]) -> bool:
        return score_result["score"] >= self.thresholds.bet_max


def get_scoring_engine() -> PickScoringEngine:
    return PickScoringEngine()