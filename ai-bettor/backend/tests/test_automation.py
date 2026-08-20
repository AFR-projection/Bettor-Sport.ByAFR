"""Tests for the automation scheduler: the "agent tidak berjalan" bug.

The scheduler is what makes the agent run on its own, so what is pinned here is
observability (status reflects live settings), control (toggle starts/stops the
task and persists the switch), and the overlap guard (a manual trigger can never
run two cycles at once).
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.database.models import SystemSetting
from backend.database.session import init_db, session_scope
from backend.services.settings_service import SettingsService, reset_settings_service


class StubPipeline:
    """Stands in for the real pipeline so no cycle touches the network."""

    def __init__(self, summary=None, error: Exception | None = None):
        self.summary = summary or {
            "status": "ok", "matches_scanned": 3, "matches_analyzed": 3,
            "bet_candidates": 1, "picks": [{"label": "OU 2.5 — Over"}],
            "no_bets": [], "telegram_sent": 0, "errors": [],
        }
        self.error = error
        self.calls = 0

    def run_cycle(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.summary

    def refresh_settings(self):
        pass

    def agent_status(self):
        return {}


@pytest.fixture(autouse=True)
def clean_state():
    init_db()
    with session_scope() as session:
        session.query(SystemSetting).delete()
    reset_settings_service()
    main._automation.update({
        "task": None, "running": False, "last_trigger": None,
        "last_started_at": None, "last_finished_at": None, "last_summary": None,
        "last_error": None, "cycles_completed": 0, "next_run_at": None,
    })
    yield
    with session_scope() as session:
        session.query(SystemSetting).delete()
    reset_settings_service()


@pytest.fixture
def stub(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(main, "_get_pipeline", lambda: pipeline)
    return pipeline


@pytest.fixture
def client(stub):
    """A client whose scheduler never fires on its own.

    The lifespan arms the loop when SCAN_AUTOMATION_ENABLED is on; the tests
    below drive cycles explicitly, so the timer staying asleep is the point.
    """
    with TestClient(main.app) as c:
        yield c


class TestAutomationStatus:
    def test_status_reports_the_scheduler_task(self, client):
        body = client.get("/automation/status").json()
        assert body["scheduler_alive"] is True  # armed by the lifespan
        assert body["cycle_running"] is False
        assert body["cycles_completed"] == 0

    def test_status_follows_live_settings(self, client):
        client.put("/settings", json={
            "AGENT_SCAN_INTERVAL_SECONDS": 450, "EARLY_MORNING_ONLY": True})
        body = client.get("/automation/status").json()
        assert body["interval_seconds"] == 450
        assert body["early_morning_only"] is True

    def test_interval_never_drops_below_the_floor(self, client):
        # Even a hand-written database row cannot make the loop spin.
        SettingsService().update({"AGENT_SCAN_INTERVAL_SECONDS": 1})
        assert client.get("/automation/status").json()["interval_seconds"] >= 30

    def test_status_carries_the_settings_revision(self, client):
        before = client.get("/automation/status").json()["settings_revision"]
        client.put("/settings", json={"MIN_EV": 0.042})
        after = client.get("/automation/status").json()["settings_revision"]
        assert after > before


class TestAutomationToggle:
    def test_toggle_off_persists_and_stops_the_task(self, client):
        body = client.post("/automation/toggle", json={"enabled": False}).json()
        assert body["enabled"] is False
        assert body["scheduler_alive"] is False
        assert body["next_run_at"] is None
        # Persisted, so a restart does not silently re-enable the agent.
        assert SettingsService().get_bool("SCAN_AUTOMATION_ENABLED") is False

    def test_toggle_on_restarts_the_task(self, client):
        client.post("/automation/toggle", json={"enabled": False})
        body = client.post("/automation/toggle", json={"enabled": True}).json()
        assert body["enabled"] is True
        assert body["scheduler_alive"] is True
        assert SettingsService().get_bool("SCAN_AUTOMATION_ENABLED") is True

    def test_toggle_is_idempotent(self, client):
        first = client.post("/automation/toggle", json={"enabled": True}).json()
        second = client.post("/automation/toggle", json={"enabled": True}).json()
        assert first["scheduler_alive"] == second["scheduler_alive"] is True

    def test_toggle_requires_the_enabled_flag(self, client):
        assert client.post("/automation/toggle", json={}).status_code == 422


class TestManualTrigger:
    def test_trigger_runs_one_cycle_and_records_it(self, client, stub):
        summary = client.post("/automation/trigger").json()
        assert summary["status"] == "ok"
        assert summary["bet_candidates"] == 1
        assert stub.calls == 1

        status = client.get("/automation/status").json()
        assert status["cycles_completed"] == 1
        assert status["last_trigger"] == "manual"
        assert status["last_error"] is None
        assert status["last_started_at"] and status["last_finished_at"]

    def test_last_summary_drops_the_bulky_pick_lists(self, client):
        client.post("/automation/trigger")
        last = client.get("/automation/status").json()["last_summary"]
        assert "picks" not in last and "no_bets" not in last
        assert last["matches_scanned"] == 3

    def test_a_failing_cycle_is_reported_not_raised(self, client, monkeypatch):
        monkeypatch.setattr(
            main, "_get_pipeline",
            lambda: StubPipeline(error=RuntimeError("odds api down")))
        body = client.post("/automation/trigger").json()
        assert body["status"] == "failed"
        assert "odds api down" in body["error"]

        status = client.get("/automation/status").json()
        assert "odds api down" in status["last_error"]
        assert status["cycle_running"] is False  # the guard was released
        assert status["cycles_completed"] == 0

    def test_overlapping_cycles_are_skipped_not_queued(self, client):
        main._automation["running"] = True
        try:
            body = client.post("/automation/trigger").json()
        finally:
            main._automation["running"] = False
        assert body == {"status": "skipped", "reason": "cycle_already_running"}

    def test_a_later_cycle_clears_an_earlier_error(self, client, monkeypatch):
        monkeypatch.setattr(
            main, "_get_pipeline",
            lambda: StubPipeline(error=RuntimeError("transient")))
        client.post("/automation/trigger")
        assert client.get("/automation/status").json()["last_error"]

        monkeypatch.setattr(main, "_get_pipeline", lambda: StubPipeline())
        client.post("/automation/trigger")
        assert client.get("/automation/status").json()["last_error"] is None
