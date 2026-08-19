"""Tests for the automated pipeline: early-morning filter, batch simulations,
database persistence, and Telegram high-score threshold."""

from __future__ import annotations

import datetime
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_pipeline_automation.db"

import pytest

from backend.database.session import init_db, session_scope
from backend.database.models import (
    AgentAnalysis, Match, OddsSnapshot, Prediction, RiskAssessment, Simulation,
)
from backend.integrations.odds_router import OddsApiRouter


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    from backend.database.models import Base
    from backend.database.session import engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


# ------------------------------------------------------------------
# Early-morning window filter (dini hari WIB)
# ------------------------------------------------------------------

class TestEarlyMorningFilter:
    def test_to_wib_conversion(self):
        from backend.agents.data_scout import DataScout
        local = DataScout._to_wib("2026-08-20T18:00:00Z")
        assert local is not None
        assert local.hour == 1  # 18:00 UTC = 01:00 WIB (+7)
        assert local.tzinfo is not None

    def test_to_wib_invalid(self):
        from backend.agents.data_scout import DataScout
        assert DataScout._to_wib("not-a-date") is None
        assert DataScout._to_wib("") is None

    def test_early_morning_today_true(self, monkeypatch):
        from backend.agents.data_scout import DataScout
        monkeypatch.setenv("EARLY_MORNING_END_HOUR", "6")
        monkeypatch.setenv("EARLY_MORNING_DAYS", "2")
        scout = DataScout()
        now_wib = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        # 01:00 WIB today (today if now < 1am, else tomorrow)
        local = now_wib.replace(hour=1, minute=0, second=0, microsecond=0)
        if local < now_wib:
            local += datetime.timedelta(days=1)
        utc = local.astimezone(datetime.timezone.utc)
        assert scout._is_early_morning_window(utc.isoformat()) is True

    def test_early_morning_tomorrow_true(self, monkeypatch):
        from backend.agents.data_scout import DataScout
        monkeypatch.setenv("EARLY_MORNING_END_HOUR", "6")
        monkeypatch.setenv("EARLY_MORNING_DAYS", "2")
        scout = DataScout()
        now_wib = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        local = (now_wib + datetime.timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
        utc = local.astimezone(datetime.timezone.utc)
        assert scout._is_early_morning_window(utc.isoformat()) is True

    def test_late_morning_rejected(self, monkeypatch):
        from backend.agents.data_scout import DataScout
        monkeypatch.setenv("EARLY_MORNING_END_HOUR", "6")
        monkeypatch.setenv("EARLY_MORNING_DAYS", "2")
        scout = DataScout()
        now_wib = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        local = (now_wib + datetime.timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        utc = local.astimezone(datetime.timezone.utc)
        assert scout._is_early_morning_window(utc.isoformat()) is False

    def test_far_future_rejected(self, monkeypatch):
        from backend.agents.data_scout import DataScout
        monkeypatch.setenv("EARLY_MORNING_END_HOUR", "6")
        monkeypatch.setenv("EARLY_MORNING_DAYS", "2")
        scout = DataScout()
        now_wib = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        local = (now_wib + datetime.timedelta(days=5)).replace(hour=2, minute=0, second=0, microsecond=0)
        utc = local.astimezone(datetime.timezone.utc)
        assert scout._is_early_morning_window(utc.isoformat()) is False

    def test_missing_time_skipped_with_warning(self, monkeypatch):
        from backend.agents.data_scout import DataScout, DataScoutResult
        monkeypatch.setenv("EARLY_MORNING_ONLY", "true")
        scout = DataScout()
        r = DataScoutResult()
        r.match_id = "no-time"
        filtered = scout._filter_early_morning([r])
        assert filtered == []
        assert "NO_COMMENCE_TIME_SKIPPED" in r.warnings

    def test_scan_filters_by_window(self, monkeypatch):
        from backend.agents.data_scout import DataScout
        monkeypatch.setenv("EARLY_MORNING_ONLY", "true")
        monkeypatch.setenv("EARLY_MORNING_END_HOUR", "6")
        monkeypatch.setenv("EARLY_MORNING_DAYS", "2")
        router = OddsApiRouter(["FAKE_KEY"])

        class FakeSession:
            headers = {}
            def get(self, *args, **kwargs):
                now_wib = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
                tomorrow_3am = (now_wib + datetime.timedelta(days=1)).replace(
                    hour=3, minute=0, second=0, microsecond=0)
                today_noon = now_wib.replace(hour=12, minute=0, second=0, microsecond=0)
                class R:
                    status_code = 200
                    def json(self):
                        return [
                            {"id": "early-1", "teams": ["A", "B"],
                             "commence_time": tomorrow_3am.astimezone(datetime.timezone.utc).isoformat(),
                             "last_update": "2026-01-01T00:00:00Z", "odds": []},
                            {"id": "noon-1", "teams": ["C", "D"],
                             "commence_time": today_noon.astimezone(datetime.timezone.utc).isoformat(),
                             "last_update": "2026-01-01T00:00:00Z", "odds": []},
                        ]
                return R()

        scout = DataScout(router=router, max_retries=1)
        scout.session = FakeSession()
        results = scout.scan_matches()
        ids = [r.match_id for r in results]
        assert "early-1" in ids
        assert "noon-1" not in ids


# ------------------------------------------------------------------
# Repeated simulations (batches)
# ------------------------------------------------------------------

class TestSimulationBatches:
    def test_batch_count_reported(self):
        from backend.agents.simulation_analyst import SimulationAnalyst
        sim = SimulationAnalyst(simulations=5000, random_seed=42)
        result = sim.simulate(1.5, 1.2, batches=3)
        assert result.simulation_count == 15000

    def test_batches_reproducible_with_seed(self):
        from backend.agents.simulation_analyst import SimulationAnalyst
        r1 = SimulationAnalyst(simulations=5000, random_seed=42).simulate(1.5, 1.2, batches=3)
        r2 = SimulationAnalyst(simulations=5000, random_seed=42).simulate(1.5, 1.2, batches=3)
        assert abs(r1.home_win_probability - r2.home_win_probability) < 0.001
        assert abs(r1.away_win_probability - r2.away_win_probability) < 0.001

    def test_batches_without_seed_runs(self):
        from backend.agents.simulation_analyst import SimulationAnalyst
        sim = SimulationAnalyst(simulations=5000, random_seed=None)
        result = sim.simulate(1.5, 1.2, batches=2)
        assert 0 < result.home_win_probability < 1
        assert result.simulation_count == 10000

    def test_single_batch_unchanged(self):
        from backend.agents.simulation_analyst import SimulationAnalyst
        a = SimulationAnalyst(simulations=5000, random_seed=42).simulate(1.5, 1.2, batches=1)
        b = SimulationAnalyst(simulations=5000, random_seed=42).simulate(1.5, 1.2)
        assert a.home_win_probability == b.home_win_probability


# ------------------------------------------------------------------
# Pipeline persistence
# ------------------------------------------------------------------

def _make_result(match_id="persist-1"):
    from backend.agents.data_scout import DataScoutResult
    r = DataScoutResult()
    r.match_id = match_id
    r.data_quality = 90
    r.commence_time = "2026-08-21T17:30:00Z"
    r.kickoff_wib = "2026-08-22T00:30:00+07:00"
    r.normalized_data = {"home_team": "HomeX", "away_team": "AwayY"}
    r.raw_match_data = {
        "id": match_id, "teams": ["HomeX", "AwayY"], "sport_key": "soccer",
        "sport_title": "Soccer",
        "commence_time": "2026-08-21T17:30:00Z",
        "odds": [{"name": "BookA", "markets": [{"key": "1X2", "selections": [
            {"name": "Home", "odd": 1.9}, {"name": "Draw", "odd": 3.4}]}]}],
    }
    return r


class TestPersistence:
    def test_persist_match_and_odds(self):
        from backend.services.pipeline import AiBettorPipeline
        pipeline = AiBettorPipeline()
        pipeline._persist_match(_make_result("persist-match-1"))
        with session_scope() as session:
            m = session.query(Match).filter(Match.match_id == "persist-match-1").first()
            assert m is not None
            assert m.home_team == "HomeX"
            odds = session.query(OddsSnapshot).filter(OddsSnapshot.match_id == "persist-match-1").all()
            assert len(odds) == 2

    def test_analyze_match_persists_prediction(self):
        from backend.services.pipeline import AiBettorPipeline
        pipeline = AiBettorPipeline()
        result = pipeline.analyze_match(_make_result("persist-decide-1"))
        assert result is not None
        with session_scope() as session:
            p = session.query(Prediction).filter(Prediction.match_id == "persist-decide-1").first()
            assert p is not None
            assert p.decision in ("BET", "NO BET")
            sim = session.query(Simulation).filter(Simulation.match_id == "persist-decide-1").first()
            assert sim is not None
            assert sim.simulation_count > 0
            risk = session.query(RiskAssessment).filter(RiskAssessment.match_id == "persist-decide-1").first()
            assert risk is not None
            analyses = session.query(AgentAnalysis).filter(AgentAnalysis.match_id == "persist-decide-1").all()
            assert len(analyses) >= 4


# ------------------------------------------------------------------
# Telegram high-score threshold
# ------------------------------------------------------------------

class FakeNotifier:
    def __init__(self):
        self.sent = []
        self.summaries = []

    def send_pick(self, decision):
        self.sent.append(decision)
        return True

    def send_no_bet_summary(self, summary):
        self.summaries.append(summary)
        return True


class TestTelegramThreshold:
    def _pipeline_with_fake_scan(self, monkeypatch, results, notifier):
        from backend.services.pipeline import AiBettorPipeline
        from backend.services.scoring import get_scoring_engine

        class FakeScout:
            max_retries = 3
            def scan_matches(self, early_morning_only=None):
                return results

        pipeline = AiBettorPipeline(scout=FakeScout(), notifier=notifier)
        return pipeline

    def test_only_high_score_picks_sent(self, monkeypatch):
        from backend.services.pipeline import AiBettorPipeline
        notifier = FakeNotifier()
        r = _make_result("tg-1")
        pipeline = self._pipeline_with_fake_scan(monkeypatch, [r], notifier)
        summary = pipeline.run_cycle()
        sent_scores = [p.get("score", 0) for p in notifier.sent]
        assert all(s >= pipeline.telegram_min_score for s in sent_scores)
        assert summary["telegram_sent"] == len(sent_scores)

    def test_no_bet_summary_when_no_high_score(self, monkeypatch):
        from backend.services.pipeline import AiBettorPipeline
        notifier = FakeNotifier()
        r = _make_result("tg-2")
        pipeline = self._pipeline_with_fake_scan(monkeypatch, [r], notifier)
        pipeline.telegram_min_score = 200  # impossible threshold
        summary = pipeline.run_cycle()
        assert notifier.sent == []
        assert len(notifier.summaries) == 1
        assert "no qualifying picks" in notifier.summaries[0]

    def test_scan_summary_shape(self, monkeypatch):
        notifier = FakeNotifier()
        r = _make_result("tg-3")
        pipeline = self._pipeline_with_fake_scan(monkeypatch, [r], notifier)
        summary = pipeline.run_cycle()
        assert summary["status"] == "completed"
        assert summary["matches_scanned"] == 1
        assert "picks" in summary
        assert "telegram_sent" in summary


# ------------------------------------------------------------------
# CLI module imports cleanly
# ------------------------------------------------------------------

class TestCLI:
    def test_cli_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_agent", "scripts/run_agent.py")
        assert spec is not None