"""Tests for probability engine, EV, implied probability, simulation."""

import pytest

from backend.models.probability_engine import (
    EVEngine,
    ImpliedProbabilityEngine,
    MonteCarloSimulationEngine,
    ProbabilityEnsemble,
)


class TestImpliedProbability:
    def test_basic_decimal(self):
        assert ImpliedProbabilityEngine.decimal_to_implied(2.0) == pytest.approx(0.5)

    def test_odds_1_91(self):
        assert ImpliedProbabilityEngine.decimal_to_implied(1.91) == pytest.approx(0.52356, abs=0.001)

    def test_margin_normalization(self):
        p = ImpliedProbabilityEngine.decimal_to_implied(2.0, bookmaker_margin=0.05)
        assert p == pytest.approx(0.47619, abs=0.001)

    def test_invalid_odds(self):
        with pytest.raises(ValueError):
            ImpliedProbabilityEngine.decimal_to_implied(0)
        with pytest.raises(ValueError):
            ImpliedProbabilityEngine.decimal_to_implied(-1.5)

    def test_round_trip(self):
        odds = 1.85
        p = ImpliedProbabilityEngine.decimal_to_implied(odds)
        back = ImpliedProbabilityEngine.implied_to_decimal(p)
        assert back == pytest.approx(odds, abs=0.02)


class TestEVEngine:
    def test_positive_ev(self):
        # p=0.6, odds=2.0: EV = 0.6*1 - 0.4*1 = 0.2
        result = EVEngine.calculate_ev(0.6, 2.0)
        assert result["ev_per_unit"] == pytest.approx(0.2, abs=0.001)

    def test_negative_ev(self):
        # p=0.4, odds=2.0: EV = 0.4*1 - 0.6*1 = -0.2
        result = EVEngine.calculate_ev(0.4, 2.0)
        assert result["ev_per_unit"] == pytest.approx(-0.2, abs=0.001)

    def test_edge_calculation(self):
        result = EVEngine.calculate_ev(0.6, 2.0)
        assert result["edge"] == pytest.approx(0.1, abs=0.001)

    def test_total_ev_with_stake(self):
        result = EVEngine.calculate_ev(0.6, 2.0, stake=10)
        assert result["total_ev"] == pytest.approx(2.0, abs=0.001)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            EVEngine.calculate_ev(0, 2.0)
        with pytest.raises(ValueError):
            EVEngine.calculate_ev(0.5, 1.0)
        with pytest.raises(ValueError):
            EVEngine.calculate_ev(0.5, 2.0, stake=0)


class TestProbabilityEnsemble:
    def test_1x2_sums_to_one(self):
        model = ProbabilityEnsemble()
        probs = model.calculate_1x2_probabilities(1.5, 1.3, 1.1, 1.4)
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert total == pytest.approx(1.0, abs=0.05)
        assert 0 < probs["home_win"] < 1
        assert 0 < probs["draw"] < 1
        assert 0 < probs["away_win"] < 1

    def test_poisson_lambda_positive(self):
        model = ProbabilityEnsemble()
        lam = model.calculate_poisson_lambda(1.0, 1.0, 1.0)
        assert lam > 0


class TestMonteCarlo:
    def test_reproducible_with_seed(self):
        e1 = MonteCarloSimulationEngine(simulations=20000, random_seed=42)
        e2 = MonteCarloSimulationEngine(simulations=20000, random_seed=42)
        r1 = e1.simulate_match(1.5, 1.2)
        r2 = e2.simulate_match(1.5, 1.2)
        assert r1["home_win_probability"] == r2["home_win_probability"]
        assert r1["draw_probability"] == r2["draw_probability"]

    def test_probabilities_in_range(self):
        e = MonteCarloSimulationEngine(simulations=10000, random_seed=1)
        r = e.simulate_match(1.5, 1.2)
        assert 0 <= r["home_win_probability"] <= 1
        assert 0 <= r["draw_probability"] <= 1
        assert 0 <= r["away_win_probability"] <= 1
        assert abs(r["home_win_probability"] + r["draw_probability"] + r["away_win_probability"] - 1) < 0.01

    def test_simulation_count_reported(self):
        e = MonteCarloSimulationEngine(simulations=5000)
        r = e.simulate_match(1.5, 1.2)
        assert r["simulation_count"] == 5000

    def test_stability_and_variance_present(self):
        e = MonteCarloSimulationEngine(simulations=5000)
        r = e.simulate_match(1.5, 1.2)
        assert "stability" in r
        assert "variance" in r
        assert r["variance"] > 0