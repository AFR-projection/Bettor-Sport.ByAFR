"""Tests for settings persistence: the "settingan tidak bisa disimpan" bug.

Covers the whole path — service coercion/clamping/persistence, the revision
counter consumers watch, and the API contract (every writable key must survive
a PUT, secrets must never leak or be wiped by an empty field).
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest
from fastapi.testclient import TestClient

from backend.database.session import init_db, session_scope
from backend.database.models import SystemSetting
from backend.main import app
from backend.services.settings_service import (
    DEFAULT_SETTINGS,
    SENSITIVE_KEYS,
    SettingsService,
    get_revision,
    get_setting,
    reset_settings_service,
)


@pytest.fixture(autouse=True)
def clean_settings_table():
    """Each test starts from an empty system_settings table."""
    init_db()
    with session_scope() as session:
        session.query(SystemSetting).delete()
    reset_settings_service()
    yield
    with session_scope() as session:
        session.query(SystemSetting).delete()
    reset_settings_service()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestSettingsService:
    def test_update_persists_across_instances(self):
        service = SettingsService()
        result = service.update({"MIN_EDGE": 0.055, "MAX_LEAGUES_PER_SCAN": 9})
        assert result["saved"] is True
        assert result["failed"] == []
        # A brand new service reads the row back from the database.
        assert SettingsService().get("MIN_EDGE") == 0.055
        assert SettingsService().get_int("MAX_LEAGUES_PER_SCAN") == 9

    def test_revision_increases_only_on_real_changes(self):
        service = SettingsService()
        assert service.revision == 0
        service.update({"MIN_EV": 0.03})
        assert service.revision == 1
        # An unknown key changes nothing, so consumers must not rebuild.
        service.update({"NOT_A_SETTING": 1})
        assert service.revision == 1

    def test_unknown_keys_are_reported_not_swallowed(self):
        result = SettingsService().update({"NOPE": 1, "MIN_ODDS": 1.8})
        assert result["rejected"]["NOPE"] == "unknown_setting"
        assert "MIN_ODDS" in result["applied"]

    def test_values_are_clamped_to_their_bounds(self):
        service = SettingsService()
        service.update({
            "SIMULATION_BATCHES": 999,          # max 20
            "AGENT_SCAN_INTERVAL_SECONDS": 1,   # min 30
            "KELLY_FRACTION": 5.0,              # max 1.0
        })
        assert service.get_int("SIMULATION_BATCHES") == 20
        assert service.get_int("AGENT_SCAN_INTERVAL_SECONDS") == 30
        assert service.get_float("KELLY_FRACTION") == 1.0

    def test_enum_rejects_garbage_mode(self):
        service = SettingsService()
        service.update({"BETTING_MODE": "casino"})
        assert service.get("BETTING_MODE") == "PAPER"

    def test_booleans_accept_strings_from_the_form(self):
        service = SettingsService()
        service.update({"SCAN_AUTOMATION_ENABLED": "false", "EARLY_MORNING_ONLY": "true"})
        assert service.get_bool("SCAN_AUTOMATION_ENABLED") is False
        assert service.get_bool("EARLY_MORNING_ONLY") is True

    def test_empty_secret_keeps_the_stored_value(self):
        service = SettingsService()
        service.update({"OPENROUTER_API_KEY": "sk-or-v1-secret-value"})
        result = service.update({"OPENROUTER_API_KEY": ""})
        assert result["rejected"]["OPENROUTER_API_KEY"] == "empty_ignored"
        assert service.get("OPENROUTER_API_KEY") == "sk-or-v1-secret-value"

    def test_masked_view_never_leaks_a_secret(self):
        service = SettingsService()
        service.update({
            "OPENROUTER_API_KEY": "sk-or-v1-secret-value",
            "THE_ODDS_API_KEYS": ["key-one-abcdefgh", "key-two-ijklmnop"],
        })
        view = service.masked_view()
        assert "secret-value" not in str(view)
        assert view["OPENROUTER_API_KEY_SET"] is True
        assert view["THE_ODDS_API_KEYS_COUNT"] == 2
        assert all(entry["is_set"] for entry in view["THE_ODDS_API_KEYS"])

    def test_listeners_are_notified_once_per_change(self):
        service = SettingsService()
        seen = []
        service.on_change(lambda changed: seen.append(changed))
        service.update({"MIN_CONFIDENCE": 70})
        assert len(seen) == 1
        assert seen[0]["MIN_CONFIDENCE"] == 70

    def test_module_helpers_see_saved_values(self):
        SettingsService().update({"MIN_CONFIDENCE": 75})
        reset_settings_service()
        assert get_setting("MIN_CONFIDENCE") == 75
        assert get_revision() == 0  # a freshly built service starts at 0


class TestSettingsApi:
    def test_get_settings_exposes_every_writable_key(self, client):
        body = client.get("/settings").json()
        for key in DEFAULT_SETTINGS:
            assert key in body, f"{key} missing from GET /settings"

    def test_put_accepts_every_non_secret_key(self, client):
        """The original bug: SettingsPayload listed only a third of the keys, so
        the rest were dropped by the request model before reaching the service."""
        payload = {
            "MIN_ODDS": 1.8, "MAX_ODDS": 5.5, "MIN_EDGE": 0.025, "MIN_EV": 0.03,
            "MIN_CONFIDENCE": 62, "SCORE_BET_THRESHOLD": 78, "MIN_BOOKMAKERS": 4,
            "MIN_DATA_QUALITY": 55, "MAX_UNCERTAINTY": 0.45,
            "MONTE_CARLO_SIMULATIONS": 15000, "SIMULATION_BATCHES": 4,
            "RANDOM_SEED": 7, "MODEL_BLEND_WEIGHT": 0.4,
            "DEFAULT_TOTAL_GOALS": 2.8, "INITIAL_BANKROLL": 2500.0,
            "KELLY_FRACTION": 0.2, "MAX_STAKE_PERCENT": 1.5,
            "MAX_EXPOSURE_PERCENT": 15.0, "EARLY_MORNING_END_HOUR": 5,
            "EARLY_MORNING_DAYS": 3, "MAX_LEAGUES_PER_SCAN": 8,
            "TELEGRAM_MIN_SCORE": 88, "TELEGRAM_MAX_PICKS": 4,
            "AGENT_SCAN_INTERVAL_SECONDS": 600, "TIMEZONE": "Asia/Makassar",
            "BETTING_MODE": "PAPER", "DEFAULT_SPORT": "soccer",
            "DEFAULT_REGIONS": "eu,uk,us", "DEFAULT_MARKETS": "h2h,totals",
            "LLM_REVIEW_ENABLED": False, "EARLY_MORNING_ONLY": False,
            "TELEGRAM_SEND_NO_BET": False, "SCAN_AUTOMATION_ENABLED": False,
        }
        response = client.put("/settings", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["failed"] == []
        assert set(payload) <= set(body["applied"]), "some keys never reached the service"

        # And every one of them is readable straight back.
        stored = client.get("/settings").json()
        for key, value in payload.items():
            assert stored[key] == value, f"{key} did not persist"

    def test_put_bumps_the_revision_so_consumers_rebuild(self, client):
        before = client.get("/settings").json()["revision"]
        client.put("/settings", json={"MIN_EV": 0.04})
        after = client.get("/settings").json()["revision"]
        assert after > before

    def test_put_does_not_echo_secrets(self, client):
        response = client.put("/settings", json={
            "TELEGRAM_BOT_TOKEN": "123456:SUPER-SECRET-TOKEN",
            "THE_ODDS_API_KEYS": ["odds-key-abcdefgh"],
        })
        body = response.json()
        assert "SUPER-SECRET-TOKEN" not in str(body)
        assert "odds-key-abcdefgh" not in str(body)
        assert body["TELEGRAM_BOT_TOKEN_SET"] is True
        assert body["THE_ODDS_API_KEYS_COUNT"] == 1

    def test_empty_put_is_a_no_op_not_a_wipe(self, client):
        client.put("/settings", json={"MIN_ODDS": 1.9})
        body = client.put("/settings", json={}).json()
        assert body["applied"] == {}
        assert body["MIN_ODDS"] == 1.9

    def test_reload_endpoint_rereads_the_database(self, client):
        # Write behind the service's back, then force a reload.
        SettingsService().update({"MIN_ODDS": 2.05})
        body = client.post("/settings/reload").json()
        assert body["MIN_ODDS"] == 2.05

    def test_secret_keys_are_masked_in_get(self, client):
        client.put("/settings", json={"OPENROUTER_API_KEY": "sk-or-v1-abcdefghijkl"})
        body = client.get("/settings").json()
        for key in SENSITIVE_KEYS:
            assert "abcdefghijkl" not in str(body.get(key, ""))
        assert body["OPENROUTER_API_KEY"].endswith("ijkl")  # masked, not raw
        assert body["OPENROUTER_API_KEY"].startswith("sk-o")

    def test_test_odds_key_by_index_never_needs_the_secret(self, client):
        client.put("/settings", json={"THE_ODDS_API_KEYS": ["stored-key-abcdefgh"]})
        body = client.post("/settings/test-odds-key", json={"index": 0}).json()
        # The dashboard only holds a mask, so it asks by index. Whether the key
        # works depends on the network; what matters is that it was resolved.
        assert "success" in body
        assert body.get("tested") not in (None, "pasted key")

    def test_test_odds_key_rejects_a_missing_index(self, client):
        body = client.post("/settings/test-odds-key", json={"index": 5}).json()
        assert body["success"] is False
