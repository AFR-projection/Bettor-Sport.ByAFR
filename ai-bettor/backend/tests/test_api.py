"""API tests using FastAPI TestClient."""

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_ai_bettor.db"

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "services" in data


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "mode" in r.json()


def test_agents(client):
    r = client.get("/agents")
    assert r.status_code == 200
    agents = r.json()
    assert "data_scout" in agents
    assert "bettor_brain" in agents
    assert agents["data_scout"]["status"] in ("IDLE", "ACTIVE", "COMPLETE", "ERROR")


def test_analyze_endpoint(client):
    r = client.post("/analyze", json={
        "match_id": "test-1", "model_probability": 0.6, "decimal_odds": 2.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["edge"] > 0
    assert data["ev"] > 0


def test_analyze_invalid_input(client):
    r = client.post("/analyze", json={
        "match_id": "test-2", "model_probability": 1.5, "decimal_odds": 2.0,
    })
    assert r.status_code == 422


def test_simulate_endpoint(client):
    r = client.post("/simulate", json={
        "home_lambda": 1.5, "away_lambda": 1.2, "simulation_count": 10000, "random_seed": 42,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["simulation_count"] == 10000
    assert data["home_win_probability"] >= 0


def test_risk_assess_endpoint(client):
    r = client.post("/risk-assess", json={
        "quant_probability": 0.6, "decimal_odds": 2.0, "bankroll": 1000,
    })
    assert r.status_code == 200
    data = r.json()
    assert "risk_level" in data
    assert "veto_decision" in data


def test_decide_endpoint(client):
    r = client.post("/decide", json={
        "quant_result": {"model_probability": 0.6, "edge": 0.1, "ev": 0.2, "confidence_score": 85},
        "market_result": {"best_odds": 2.0, "best_bookmaker": "BookA"},
        "simulation_result": {"home_win_probability": 0.6, "draw_probability": 0.2,
                              "away_win_probability": 0.2, "stability": 0.9, "variance": 1.5},
        "risk_result": {"data_quality_score": 90, "veto_decision": False, "risk_level": "LOW"},
        "match_data": {"home_team": "A", "away_team": "B"},
        "bettor_state": {"current_exposure": 0, "recent_results": [], "consecutive_losses": 0, "consecutive_wins": 0},
    })
    assert r.status_code == 200
    assert r.json()["decision"] == "BET"


def test_decide_no_bet_on_veto(client):
    r = client.post("/decide", json={
        "quant_result": {"model_probability": 0.6, "edge": 0.1, "ev": 0.2, "confidence_score": 85},
        "market_result": {"best_odds": 2.0, "best_bookmaker": "BookA"},
        "simulation_result": {"home_win_probability": 0.6, "draw_probability": 0.2,
                              "away_win_probability": 0.2, "stability": 0.9, "variance": 1.5},
        "risk_result": {"data_quality_score": 90, "veto_decision": True, "risk_level": "HIGH"},
        "match_data": {},
        "bettor_state": {},
    })
    assert r.status_code == 200
    assert r.json()["decision"] == "NO BET"


def test_bankroll_endpoint(client):
    r = client.get("/bankroll")
    assert r.status_code == 200
    data = r.json()
    assert "current_balance" in data
    assert data["mode"] == "PAPER"


def test_predictions_empty(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_matches_empty(client):
    r = client.get("/matches")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_logs_endpoint(client):
    r = client.get("/logs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_backtest_endpoint(client):
    matches = [
        {
            "match_id": "bt-1", "home_team": "A", "away_team": "B", "league": "L1",
            "market_odds": {"odds": 1.91}, "home_lambda": 1.5, "away_lambda": 1.2,
            "model_probability": 0.55, "actual_home_goals": 2, "actual_away_goals": 0,
        }
    ]
    r = client.post("/backtest", json={"matches": matches})
    assert r.status_code == 200
    data = r.json()
    assert "total_bets" in data
    assert "max_drawdown" in data
    assert "calibration_error" in data


def test_scoring_thresholds(client):
    r = client.get("/scoring/thresholds")
    assert r.status_code == 200
    assert r.json()["no_bet"] == 59


def test_match_404(client):
    r = client.get("/matches/nonexistent-id")
    assert r.status_code == 404