"""Backtesting framework for AI Bettor.

Flow: historical data → model → simulation → decision → virtual bet → result → performance

Metrics:
- total bets, wins, losses, win rate, ROI, profit/loss
- average odds, EV, max drawdown, calibration, CLV (if available)
- performance by market, league, confidence score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agents.bettor_brain import BettorBrain
from backend.agents.market_analyst import MarketAnalyst
from backend.agents.quant_analyst import QuantAnalyst
from backend.agents.risk_manager import RiskManager
from backend.agents.simulation_analyst import SimulationAnalyst


@dataclass
class BacktestResult:
    """Aggregated backtest performance metrics."""
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    profit_loss: float = 0.0
    total_staked: float = 0.0
    average_odds: float = 0.0
    average_ev: float = 0.0
    max_drawdown: float = 0.0
    calibration_error: float = 0.0
    clv: Optional[float] = None
    by_market: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_league: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_confidence: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bet_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "win_rate": round(self.win_rate, 4),
            "roi": round(self.roi, 4),
            "profit_loss": round(self.profit_loss, 2),
            "total_staked": round(self.total_staked, 2),
            "average_odds": round(self.average_odds, 4),
            "average_ev": round(self.average_ev, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "calibration_error": round(self.calibration_error, 4),
            "clv": round(self.clv, 4) if self.clv is not None else None,
            "by_market": self.by_market,
            "by_league": self.by_league,
            "by_confidence": self.by_confidence,
            "bet_history": self.bet_history,
        }


class Backtester:
    """Runs the full backtest pipeline on historical matches."""

    def __init__(
        self,
        quant: Optional[QuantAnalyst] = None,
        market: Optional[MarketAnalyst] = None,
        simulation: Optional[SimulationAnalyst] = None,
        risk: Optional[RiskManager] = None,
        brain: Optional[BettorBrain] = None,
        stake_per_bet: float = 10.0,
    ):
        self.quant = quant or QuantAnalyst()
        self.market = market or MarketAnalyst()
        self.simulation = simulation or SimulationAnalyst(simulations=5000)
        self.risk = risk or RiskManager()
        self.brain = brain or BettorBrain()
        self.stake_per_bet = stake_per_bet

    def run(
        self,
        historical_matches: List[Dict[str, Any]],
    ) -> BacktestResult:
        """Run backtest over historical matches.

        Each match dict must include:
        home_team, away_team, league, actual_home_goals, actual_away_goals,
        market_odds (dict), and optionally closing_odds and model_probability.
        """
        result = BacktestResult()
        cumulative_profit = 0.0
        peak_profit = 0.0
        max_drawdown = 0.0
        total_odds = 0.0
        total_ev = 0.0
        calibration_diffs: List[float] = []

        for match in historical_matches:
            bet = self._evaluate_match(match)
            result.bet_history.append(bet)

            if not bet["decision"] == "BET":
                continue

            result.total_bets += 1
            total_odds += bet["odds"]
            total_ev += bet["ev"]

            outcome = bet["outcome"]  # win / loss / push
            if outcome == "win":
                result.wins += 1
                profit = self.stake_per_bet * (bet["odds"] - 1)
            elif outcome == "push":
                result.pushes += 1
                profit = 0.0
            else:
                result.losses += 1
                profit = -self.stake_per_bet

            result.profit_loss += profit
            cumulative_profit += profit
            peak_profit = max(peak_profit, cumulative_profit)
            max_drawdown = min(max_drawdown, cumulative_profit - peak_profit)

            # Calibration: how close predicted prob was to actual outcome
            predicted = bet.get("model_probability", 0.5)
            actual = 1.0 if outcome == "win" else (0.5 if outcome == "push" else 0.0)
            calibration_diffs.append(abs(predicted - actual))

            # CLV if closing odds available
            closing = match.get("closing_odds")
            if closing and closing > 0 and bet["odds"] > 1:
                clv = (bet["odds"] - closing) / closing
                result.clv = clv if result.clv is None else (result.clv * (result.total_bets - 1) + clv) / result.total_bets

            # Group by market / league / confidence
            self._aggregate(result.by_market, bet["market"], bet["odds"], profit, stake=self.stake_per_bet)
            self._aggregate(result.by_league, match.get("league", "UNKNOWN"), bet["odds"], profit, stake=self.stake_per_bet)
            conf_bucket = self._conf_bucket(bet.get("confidence", 0))
            self._aggregate(result.by_confidence, conf_bucket, bet["odds"], profit, stake=self.stake_per_bet)

        if result.total_bets > 0:
            result.total_staked = result.total_bets * self.stake_per_bet
            result.win_rate = result.wins / result.total_bets
            result.roi = result.profit_loss / (result.total_bets * self.stake_per_bet)
            result.average_odds = total_odds / result.total_bets
            result.average_ev = total_ev / result.total_bets
        if calibration_diffs:
            result.calibration_error = sum(calibration_diffs) / len(calibration_diffs)
        result.max_drawdown = abs(max_drawdown)

        return result

    def _evaluate_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full agent pipeline on one historical match."""
        market_odds = match.get("market_odds", {})
        odds = market_odds.get("odds", 1.91)
        home_goals = match.get("actual_home_goals", 0)
        away_goals = match.get("actual_away_goals", 0)

        model_prob = match.get("model_probability")
        if model_prob is None:
            home_lambda = match.get("home_lambda", 1.5)
            away_lambda = match.get("away_lambda", 1.2)
            sim = self.simulation.simulate(home_lambda, away_lambda)
            model_prob = sim.home_win_probability

        quant = self.quant.analyze("backtest", model_prob, odds)
        market_res = self.market.analyze("backtest", [{
            "name": "BacktestBook", "markets": [{
                "key": "1X2", "selections": [
                    {"name": "Home", "odd": odds},
                ],
            }],
        }])
        sim_res = self.simulation.simulate(match.get("home_lambda", 1.5), match.get("away_lambda", 1.2))
        risk_res = self.risk.assess(quant, sim_res, [{
            "name": "BacktestBook", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": odds}]}],
        }], bankroll=1000)
        brain = self.brain.decide(
            quant.to_dict(), market_res.to_dict(), sim_res.to_dict(), risk_res.to_dict(),
            {"home_team": match.get("home_team", ""), "away_team": match.get("away_team", "")},
            {"current_exposure": 0.0, "recent_results": [], "consecutive_losses": 0, "consecutive_wins": 0},
        )

        decision = brain.decision
        outcome = "unknown"
        if decision == "BET":
            selection = brain.selection.lower()
            if selection in ("home", "1x2 - home"):
                outcome = "win" if home_goals > away_goals else ("push" if home_goals == away_goals else "loss")
            elif selection in ("draw", "1x2 - draw"):
                outcome = "win" if home_goals == away_goals else "loss"
            else:
                outcome = "win" if away_goals > home_goals else ("push" if home_goals == away_goals else "loss")

        return {
            "match_id": match.get("match_id", ""),
            "home_team": match.get("home_team", ""),
            "away_team": match.get("away_team", ""),
            "league": match.get("league", "UNKNOWN"),
            "decision": decision,
            "market": brain.market,
            "selection": brain.selection,
            "odds": brain.odds,
            "ev": brain.ev,
            "edge": brain.edge,
            "model_probability": brain.probability,
            "confidence": brain.confidence,
            "outcome": outcome,
        }

    @staticmethod
    def _aggregate(group: Dict[str, Dict[str, float]], key: str, odds: float, profit: float, stake: float) -> None:
        if key not in group:
            group[key] = {"bets": 0, "wins": 0, "roi": 0.0, "profit": 0.0}
        g = group[key]
        g["bets"] += 1
        if profit > 0:
            g["wins"] += 1
        g["profit"] += profit
        g["roi"] = g["profit"] / (g["bets"] * stake) if g["bets"] else 0

    @staticmethod
    def _conf_bucket(confidence: int) -> str:
        if confidence >= 90:
            return "90+"
        if confidence >= 80:
            return "80-89"
        if confidence >= 70:
            return "70-79"
        return "<70"


def get_backtester(**kwargs) -> Backtester:
    return Backtester(**kwargs)