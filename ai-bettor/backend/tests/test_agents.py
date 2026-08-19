"""Tests for agents: quant analyst, market analyst, risk manager, bettor brain, scoring."""

import pytest

from backend.agents.quant_analyst import QuantAnalyst
from backend.agents.market_analyst import MarketAnalyst
from backend.agents.risk_manager import RiskManager
from backend.agents.bettor_brain import BettorBrain
from backend.agents.simulation_analyst import SimulationAnalyst
from backend.services.scoring import PickScoringEngine, ScoreWeights


class TestQuantAnalyst:
    def test_positive_edge_recommendation(self):
        q = QuantAnalyst(min_edge=0.01, min_ev=0.01, min_confidence=60)
        r = q.analyze("m1", model_prob=0.6, decimal_odds=2.0)
        assert r.edge > 0.05
        assert r.ev > 0
        assert r.recommendation in ("BET CANDIDATE", "PREMIUM CANDIDATE")

    def test_negative_edge_no_bet(self):
        q = QuantAnalyst()
        r = q.analyze("m2", model_prob=0.4, decimal_odds=2.5)
        assert r.recommendation == "NO BET"

    def test_no_bet_on_low_edge(self):
        q = QuantAnalyst(min_edge=0.02)
        r = q.analyze("m3", model_prob=0.505, decimal_odds=2.0)
        assert r.recommendation == "NO BET"

    def test_invalid_odds_rejected(self):
        q = QuantAnalyst()
        with pytest.raises(ValueError):
            q.analyze("m4", model_prob=0.5, decimal_odds=0.5)


class TestMarketAnalyst:
    SAMPLE_ODDS = [
        {"name": "BookA", "markets": [{"key": "1X2", "selections": [
            {"name": "Home", "odd": 1.85}, {"name": "Draw", "odd": 3.5}, {"name": "Away", "odd": 4.2}]}]},
        {"name": "BookB", "markets": [{"key": "1X2", "selections": [
            {"name": "Home", "odd": 1.91}, {"name": "Draw", "odd": 3.4}, {"name": "Away", "odd": 4.0}]}]},
    ]

    def test_best_odds_found(self):
        m = MarketAnalyst()
        r = m.analyze("m1", self.SAMPLE_ODDS)
        assert r.best_odds == pytest.approx(4.2, abs=0.01)
        assert r.best_bookmaker == "BookA"

    def test_empty_odds_warns(self):
        m = MarketAnalyst()
        r = m.analyze("m2", [])
        assert "NO_ODDS_DATA" in r.warnings

    def test_consensus_present(self):
        m = MarketAnalyst()
        r = m.analyze("m3", self.SAMPLE_ODDS)
        assert r.market_consensus is not None
        assert "average_odds" in r.market_consensus

    def test_price_spread_detected(self):
        m = MarketAnalyst()
        r = m.analyze("m4", self.SAMPLE_ODDS)
        assert r.price_difference > 0


class TestSimulationAnalyst:
    def test_reproducible(self):
        s1 = SimulationAnalyst(simulations=10000, random_seed=42)
        s2 = SimulationAnalyst(simulations=10000, random_seed=42)
        r1 = s1.simulate(1.5, 1.2)
        r2 = s2.simulate(1.5, 1.2)
        assert r1.home_win_probability == r2.home_win_probability

    def test_simulation_count(self):
        s = SimulationAnalyst(simulations=20000)
        r = s.simulate(1.5, 1.2)
        assert r.simulation_count == 20000
        assert "LOW_STABILITY" not in r.warnings or r.stability >= 0.5

    def test_handicap_simulation(self):
        s = SimulationAnalyst(simulations=5000)
        r = s.simulate_handicap(1.5, 1.2, -0.5)
        assert 0 < r["home_handicap_cover"] < 1

    def test_ou_simulation(self):
        s = SimulationAnalyst(simulations=5000)
        r = s.simulate_ou(1.5, 1.2, 2.5)
        assert abs(r["over_probability"] + r["under_probability"] - 1) < 0.01


class TestRiskManager:
    def _make_results(self):
        q = QuantAnalyst().analyze("m", 0.6, 2.0)
        s = SimulationAnalyst(simulations=5000).simulate(1.5, 1.2)
        odds = [{"name": "A", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": 2.0}]}]}]
        return q, s, odds

    def test_veto_on_poor_edge(self):
        risk = RiskManager(veto_edge_threshold=0.01, veto_ev_threshold=0.01)
        q, s, odds = self._make_results()
        q.edge = 0.0
        q.ev = 0.0
        r = risk.assess(q, s, odds, bankroll=1000)
        assert r.veto_decision is True

    def test_veto_on_high_exposure(self):
        risk = RiskManager()
        q, s, odds = self._make_results()
        r = risk.assess(q, s, odds, current_exposure=400, bankroll=1000)
        assert r.exposure_concern is True

    def test_no_veto_on_good_situation(self):
        risk = RiskManager(veto_edge_threshold=0.01, veto_ev_threshold=0.01)
        q, s, odds = self._make_results()
        r = risk.assess(q, s, odds, current_exposure=0, bankroll=1000)
        assert r.veto_decision is False

    def test_risk_level_present(self):
        risk = RiskManager()
        q, s, odds = self._make_results()
        r = risk.assess(q, s, odds, bankroll=1000)
        assert r.risk_level in ("LOW", "MEDIUM", "HIGH")


class TestBettorBrain:
    def _inputs(self):
        return {
            "quant_result": {"model_probability": 0.6, "edge": 0.1, "ev": 0.2, "confidence_score": 85},
            "market_result": {"best_odds": 2.0, "best_bookmaker": "BookA"},
            "simulation_result": {"home_win_probability": 0.6, "draw_probability": 0.2,
                                  "away_win_probability": 0.2, "stability": 0.9, "variance": 1.5},
            "risk_result": {"data_quality_score": 90, "veto_decision": False, "risk_level": "LOW"},
            "match_data": {"home_team": "Team A", "away_team": "Team B"},
            "bettor_state": {"current_exposure": 0, "recent_results": [], "consecutive_losses": 0, "consecutive_wins": 0},
        }

    def test_bet_decision_on_good_candidate(self):
        brain = BettorBrain(min_edge=0.01, min_ev=0.01, min_confidence=60)
        r = brain.decide(**self._inputs())
        assert r.decision == "BET"
        assert r.odds == 2.0
        assert r.minimum_acceptable_odds == 1.75

    def test_no_bet_on_risk_veto(self):
        brain = BettorBrain()
        inputs = self._inputs()
        inputs["risk_result"]["veto_decision"] = True
        r = brain.decide(**inputs)
        assert r.decision == "NO BET"

    def test_no_bet_on_bad_data(self):
        brain = BettorBrain()
        inputs = self._inputs()
        inputs["risk_result"]["data_quality_score"] = 20
        r = brain.decide(**inputs)
        assert r.decision == "NO BET"

    def test_no_bet_on_chasing_losses(self):
        brain = BettorBrain()
        inputs = self._inputs()
        inputs["bettor_state"]["consecutive_losses"] = 4
        inputs["quant_result"]["ev"] = 0.0
        r = brain.decide(**inputs)
        assert r.decision == "NO BET"

    def test_no_bet_when_odds_below_minimum(self):
        brain = BettorBrain(min_acceptable_odds=1.75)
        inputs = self._inputs()
        inputs["market_result"]["best_odds"] = 1.5
        r = brain.decide(**inputs)
        assert r.decision == "NO BET"

    def test_no_model_probability_no_bet(self):
        brain = BettorBrain()
        inputs = self._inputs()
        inputs["quant_result"]["model_probability"] = 0.0
        inputs["simulation_result"]["home_win_probability"] = 0.0
        r = brain.decide(**inputs)
        assert r.decision == "NO BET"


class TestScoringEngine:
    def test_good_candidate_scores_high(self):
        engine = PickScoringEngine()
        candidate = {
            "model_probability": 0.6, "ev": 0.2, "edge": 0.1,
            "simulation_stability": 0.9, "market_disagreement": 0.05,
            "odds": 2.0, "minimum_acceptable_odds": 1.75,
            "data_quality": 90, "risk_level": "LOW", "bookmaker_consensus": 0.8,
        }
        r = engine.score(candidate)
        assert r["score"] >= 70

    def test_bad_candidate_scores_low(self):
        engine = PickScoringEngine()
        candidate = {
            "model_probability": 0.3, "ev": -0.1, "edge": -0.05,
            "simulation_stability": 0.2, "market_disagreement": 0.5,
            "odds": 1.5, "minimum_acceptable_odds": 1.75,
            "data_quality": 20, "risk_level": "HIGH", "bookmaker_consensus": 0.3,
        }
        r = engine.score(candidate)
        assert r["score"] <= 60

    def test_threshold_labels(self):
        engine = PickScoringEngine()
        assert engine.thresholds.label(50) == "NO BET"
        assert engine.thresholds.label(65) == "PASS"
        assert engine.thresholds.label(75) == "WATCH"
        assert engine.thresholds.label(85) == "BET CANDIDATE"
        assert engine.thresholds.label(95) == "PREMIUM CANDIDATE"

    def test_configurable_weights(self):
        weights = ScoreWeights(ev=0.5, edge=0.5, model_probability=0, simulation_stability=0,
                               market_agreement=0, odds_quality=0, data_quality=0, risk=0)
        engine = PickScoringEngine(weights)
        r = engine.score({"model_probability": 0.5, "ev": 0.3, "edge": 0.1,
                          "simulation_stability": 0.5, "odds": 2.0,
                          "minimum_acceptable_odds": 1.75, "data_quality": 50,
                          "risk_level": "MEDIUM", "bookmaker_consensus": 1.0})
        assert r["score"] > 0