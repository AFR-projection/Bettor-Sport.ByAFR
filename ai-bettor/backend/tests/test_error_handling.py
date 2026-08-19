"""Tests for error handling: API timeouts, empty responses, malformed data, LLM failure."""

import pytest

from backend.agents.data_scout import DataScout, DataScoutResult
from backend.integrations.openrouter import LLMError, OpenRouterClient


class TestDataScoutErrorHandling:
    def test_no_api_key_returns_empty_with_warning(self, monkeypatch):
        monkeypatch.setenv("THE_ODDS_API_KEY", "")
        scout = DataScout()
        scout.api_key = ""
        results = scout.scan_matches()
        assert results == []

    def test_empty_response_processed_safely(self):
        scout = DataScout()
        results = scout._process_odds_data([])
        assert results == []

    def test_malformed_match_data(self):
        scout = DataScout()
        # Missing id, missing teams
        results = scout._process_odds_data([{"odds": []}])
        assert len(results) == 1
        assert results[0].data_quality <= 10

    def test_duplicate_bookmakers_deduplicated(self):
        scout = DataScout()
        data = [{
            "id": "m1", "teams": ["A", "B"], "last_update": "2026-01-01T00:00:00Z",
            "odds": [
                {"name": "BookA", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": 1.9}]}]},
                {"name": "BookA", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": 1.91}]}]},
            ],
        }]
        results = scout._process_odds_data(data)
        assert results[0].bookmakers == ["BookA"]

    def test_invalid_odds_detected(self):
        scout = DataScout()
        data = [{
            "id": "m2", "teams": ["A", "B"], "last_update": "2026-01-01T00:00:00Z",
            "odds": [{"name": "BookA", "markets": [{"key": "1X2", "selections": [
                {"name": "Home", "odd": 0}, {"name": "Draw", "odd": None}, {"name": "Away", "odd": -2}]}]}],
        }]
        results = scout._process_odds_data(data)
        assert results[0].data_quality < 100

    def test_missing_timestamp_warns(self):
        scout = DataScout()
        data = [{
            "id": "m3", "teams": ["A", "B"],
            "odds": [{"name": "BookA", "markets": [{"key": "1X2", "selections": [{"name": "Home", "odd": 1.9}]}]}],
        }]
        results = scout._process_odds_data(data)
        assert "NO_TIMESTAMP" in results[0].warnings

    def test_postponed_match_status_not_crash(self):
        scout = DataScout()
        data = [{
            "id": "m4", "teams": ["A", "B"], "commence_time": None, "status": "postponed",
            "odds": [],
        }]
        results = scout._process_odds_data(data)
        assert results[0].data_quality < 100


class TestOpenRouterErrorHandling:
    def test_not_configured_raises(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        client = OpenRouterClient(api_key="", model="test/model")
        with pytest.raises(LLMError):
            client.complete("sys", "user")

    def test_llm_failure_does_not_crash_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        client = OpenRouterClient(api_key="", model="test/model")
        result = client.analyze_match({"match_id": "x"})
        assert result["review"] == "LLM_UNAVAILABLE"
        assert result["agrees_with_quant"] is True


class TestPaperBettingErrors:
    def test_settle_unknown_bet(self):
        from backend.services.paper_betting import PaperBettingService
        service = PaperBettingService()
        r = service.settle_bet("does-not-exist", "win")
        assert r["status"] == "not_found"

    def test_invalid_outcome(self):
        from backend.services.paper_betting import PaperBettingService
        service = PaperBettingService()
        r = service.settle_bet("x", "invalid-outcome")
        assert r["status"] == "invalid"

    def test_no_bet_decision_not_placed(self):
        from backend.services.paper_betting import PaperBettingService
        service = PaperBettingService()
        r = service.place_bet({"decision": "NO BET"})
        assert r["status"] == "no_bet"