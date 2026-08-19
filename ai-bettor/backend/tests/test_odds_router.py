"""Tests for the multi-key Odds API router and settings service."""

from __future__ import annotations

import os
import time

os.environ["DATABASE_URL"] = "sqlite:///./test_ai_bettor.db"

import pytest

from backend.database.session import init_db
from backend.integrations.odds_router import OddsApiRouter, mask_key
from backend.services.settings_service import SettingsService


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    from backend.database.models import SystemSetting
    from backend.database.session import session_scope
    with session_scope() as session:
        session.query(SystemSetting).delete()


# ------------------------------------------------------------------
# OddsApiRouter
# ------------------------------------------------------------------

class TestOddsApiRouter:
    def test_add_and_rotate_keys(self):
        router = OddsApiRouter(["KEY_A", "KEY_B", "KEY_C"])
        seen = {router.get_key() for _ in range(3)}
        assert seen == {"KEY_A", "KEY_B", "KEY_C"}

    def test_failover_on_429(self):
        router = OddsApiRouter(["KEY_A", "KEY_B"])
        router.report_failure("KEY_A", "RATE_LIMITED", 429)
        key = router.get_key()
        assert key == "KEY_B"

    def test_failover_on_401_disables_key(self):
        router = OddsApiRouter(["KEY_A", "KEY_B"])
        router.report_failure("KEY_A", "UNAUTHORIZED", 401)
        for _ in range(3):
            assert router.get_key() == "KEY_B"

    def test_all_keys_unavailable_returns_none(self):
        router = OddsApiRouter(["KEY_A"])
        router.report_failure("KEY_A", "RATE_LIMITED", 429)
        assert router.get_key() is None

    def test_success_resets_cooldown(self):
        router = OddsApiRouter(["KEY_A", "KEY_B"])
        router.report_failure("KEY_A", "RATE_LIMITED", 429)
        assert router.get_key() == "KEY_B"
        router.report_success("KEY_A", remaining=50)
        assert router.get_key() == "KEY_A"

    def test_cooldown_expiry(self):
        router = OddsApiRouter(["KEY_A"])
        router.report_failure("KEY_A", "RATE_LIMITED", 429)
        router._status["KEY_A"]["cooldown_until"] = time.time() - 1
        assert router.get_key() == "KEY_A"

    def test_reset_key(self):
        router = OddsApiRouter(["KEY_A"])
        router.report_failure("KEY_A", "UNAUTHORIZED", 401)
        router.reset_key("KEY_A")
        assert router.get_key() == "KEY_A"

    def test_status_report(self):
        router = OddsApiRouter(["KEY_A", "KEY_B"])
        router.report_failure("KEY_A", "RATE_LIMITED", 429)
        status = router.status()
        assert len(status) == 2
        status_a = next(s for s in status if s["full_key"] == "KEY_A")
        assert status_a["status"] == "COOLDOWN"
        assert status_a["cooldown_remaining_seconds"] > 0
        status_b = next(s for s in status if s["full_key"] == "KEY_B")
        assert status_b["status"] == "OK"

    def test_set_keys_sync(self):
        router = OddsApiRouter(["KEY_A"])
        router.set_keys(["KEY_B", "KEY_C"])
        keys = {k["full_key"] for k in router.status()}
        assert keys == {"KEY_B", "KEY_C"}

    def test_mask_key(self):
        assert mask_key("abcdefgh1234WXYZ") == "abcd...WXYZ"
        assert mask_key("short") == "***"
        assert mask_key("") == ""

    def test_remove_key(self):
        router = OddsApiRouter(["KEY_A", "KEY_B"])
        router.remove_key("KEY_A")
        assert router.key_count() == 1
        assert router.get_key() == "KEY_B"


# ------------------------------------------------------------------
# SettingsService
# ------------------------------------------------------------------

class TestSettingsService:
    def test_defaults_loaded(self):
        svc = SettingsService(router=OddsApiRouter())
        assert svc.get("BETTING_MODE") == "PAPER"
        assert svc.get("MIN_ODDS") == 1.75

    def test_update_odds_keys_goes_to_router(self):
        router = OddsApiRouter()
        svc = SettingsService(router=router)
        svc.update({"THE_ODDS_API_KEYS": ["KEY_ONE", "KEY_TWO"]})
        assert router.key_count() == 2
        assert router.get_key() in ("KEY_ONE", "KEY_TWO")

    def test_update_and_masked_view(self):
        router = OddsApiRouter()
        svc = SettingsService(router=router)
        svc.update({"OPENROUTER_API_KEY": "sk-or-v1-SUPERSECRET1234"})
        view = svc.masked_view()
        assert "SUPERSECRET" not in str(view)
        assert view["OPENROUTER_API_KEY_SET"] is True
        assert view["OPENROUTER_API_KEY"].endswith("1234")

    def test_update_coerces_types(self):
        router = OddsApiRouter()
        svc = SettingsService(router=router)
        svc.update({"MIN_ODDS": "2.5", "MIN_CONFIDENCE": "75"})
        assert svc.get("MIN_ODDS") == 2.5
        assert svc.get("MIN_CONFIDENCE") == 75

    def test_update_ignores_unknown_keys(self):
        router = OddsApiRouter()
        svc = SettingsService(router=router)
        svc.update({"NOT_A_REAL_KEY": "x", "MIN_EV": 0.05})
        assert svc.get("NOT_A_REAL_KEY") is None
        assert svc.get("MIN_EV") == 0.05

    def test_persists_to_db(self):
        router = OddsApiRouter()
        svc = SettingsService(router=router)
        svc.update({"RANDOM_SEED": 777})
        svc2 = SettingsService(router=OddsApiRouter())
        assert svc2.get("RANDOM_SEED") == 777

    def test_odds_router_in_masked_view(self):
        router = OddsApiRouter(["KEY_ABC", "KEY_XYZ"])
        svc = SettingsService(router=router)
        view = svc.masked_view()
        assert "odds_router" in view
        assert len(view["odds_router"]) == 2


# ------------------------------------------------------------------
# Settings API endpoints
# ------------------------------------------------------------------

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestSettingsAPI:
    def test_get_settings(self):
        r = client.get("/settings")
        assert r.status_code == 200
        body = r.json()
        assert "BETTING_MODE" in body
        assert "THE_ODDS_API_KEYS" in body
        assert "odds_router" in body

    def test_put_settings(self):
        r = client.put("/settings", json={"MIN_EDGE": 0.03, "MIN_CONFIDENCE": 65})
        assert r.status_code == 200
        body = r.json()
        assert body["MIN_EDGE"] == 0.03
        assert body["MIN_CONFIDENCE"] == 65

    def test_put_odds_keys_masked(self):
        r = client.put("/settings", json={"THE_ODDS_API_KEYS": ["abc123secret456KEY"]})
        assert r.status_code == 200
        body = r.json()
        assert "secret456" not in str(body)
        assert body["THE_ODDS_API_KEYS_COUNT"] == 1

    def test_odds_api_router_status(self):
        r = client.get("/settings/odds-api/status")
        assert r.status_code == 200
        body = r.json()
        assert "keys" in body
        assert "has_keys" in body

    def test_test_odds_key_invalid(self):
        r = client.post("/settings/test-odds-key", json={"api_key": ""})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_test_openrouter_no_key(self):
        r = client.post("/settings/test-openrouter", json={})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_health_uses_router(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["services"]["the_odds_api"] in ("configured", "missing_key")


# ------------------------------------------------------------------
# DataScout router integration
# ------------------------------------------------------------------

from backend.agents.data_scout import DataScout


class TestDataScoutRouter:
    def test_scout_uses_router_when_keys_available(self, monkeypatch):
        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, *args, **kwargs):
                class R:
                    status_code = 200

                    def json(self):
                        return []
                return R()

        router = OddsApiRouter(["FAKE_KEY"])
        scout = DataScout(router=router, max_retries=1)
        scout.session = FakeSession()
        results = scout.scan_matches()
        assert results == []
        assert router.status()[0]["requests"] == 1

    def test_scout_no_keys(self):
        scout = DataScout(router=OddsApiRouter(), max_retries=1)
        results = scout.scan_matches()
        assert results == []
        assert any("NO_API_KEY" in r.warnings for r in results) or results == []