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

    # Calibration: the edge/EV a genuinely excellent soccer pick carries.
    # 8% edge and 0.15 EV per unit are already top-of-the-market numbers, so
    # they earn full marks on their factor instead of an unreachable ceiling.
    EDGE_FULL_MARKS = 0.08
    EV_FULL_MARKS = 0.15

    def __init__(self, weights: Optional[ScoreWeights] = None,
                 bet_threshold: Optional[int] = None):
        self.weights = (weights or ScoreWeights()).normalize()
        self.thresholds = ScoringThresholds()
        if bet_threshold is None:
            from backend.services.settings_service import get_setting
            bet_threshold = get_setting("SCORE_BET_THRESHOLD", get_settings().SCORE_BET_THRESHOLD)
        # A pick qualifies at or above this score. The label boundaries above
        # stay fixed; this is the *action* threshold and it is configurable.
        self.bet_threshold = max(0, min(100, int(bet_threshold)))

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

        # 2. EV factor: full marks at EV_FULL_MARKS per unit staked.
        # Scaled to what a de-vig + line-shopping engine really finds, not to a
        # fantasy ceiling — otherwise no honest pick could ever reach the
        # action threshold and the engine would never bet at all.
        ev = candidate.get("ev", 0)
        factors["ev"] = min(100.0, max(0.0, ev / self.EV_FULL_MARKS * 100))

        # 3. Edge factor: full marks at EDGE_FULL_MARKS of edge
        edge = candidate.get("edge", 0)
        factors["edge"] = min(100.0, max(0.0, edge / self.EDGE_FULL_MARKS * 100))

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

        # Bookmaker consensus penalty (fewer books quoting = bigger deduction).
        # Deepest band first: an `elif` chain the other way round never fires.
        consensus = candidate.get("bookmaker_consensus", 1.0)
        if consensus < 0.4:
            score -= 10
        elif consensus < 0.6:
            score -= 5
        score = round(min(100.0, max(0.0, score)), 1)

        return {
            "score": score,
            "label": self.thresholds.label(score),
            "factors": {k: round(v, 2) for k, v in factors.items()},
            "weights": self.weights.__dict__,
            "bet_threshold": self.bet_threshold,
            "thresholds": {
                "no_bet": self.thresholds.no_bet_max,
                "pass": self.thresholds.pass_max,
                "watch": self.thresholds.watch_max,
                "bet": self.thresholds.bet_max,
                "premium": self.thresholds.premium_min,
            },
        }

    def is_bettable(self, score_result: Dict[str, Any]) -> bool:
        """True when the score reaches the configured action threshold."""
        return float(score_result.get("score", 0)) >= self.bet_threshold


def get_scoring_engine(bet_threshold: Optional[int] = None) -> PickScoringEngine:
    return PickScoringEngine(bet_threshold=bet_threshold)