"""Tests for the pick score reaching the database and the dashboard.

`pick_score`/`score_label` come from the scoring gate and are what the dashboard
ranks picks by, so this pins the whole hop: pipeline persist → column exists on
an older database (there is no migration framework) → API serialisation.
"""

from __future__ import annotations

import datetime as dt
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from backend.database.models import Match, Prediction
from backend.database.session import _ADDED_COLUMNS, engine, init_db, session_scope
from backend.main import app
from backend.services.pipeline import AiBettorPipeline


def a_result(**extra) -> dict:
    """A pipeline result dict as `_persist_prediction` expects it."""
    result = {
        "match_id": "m-score-1", "decision": "BET", "market": "OU",
        "selection": "Over", "odds": 1.95, "bookmaker": "Pinnacle",
        "probability": 0.58, "implied_probability": 0.5128,
        "edge": 0.067, "ev": 0.131, "confidence": 74,
        "score": 86.5, "score_label": "STRONG", "risk": "MEDIUM",
        "reasoning": "consensus 0.58 vs best price 1.95",
    }
    result.update(extra)
    return result


@pytest.fixture(autouse=True)
def clean_predictions():
    init_db()
    with session_scope() as session:
        session.query(Prediction).delete()
        session.query(Match).filter(Match.match_id.like("m-score-%")).delete(
            synchronize_session=False)
    yield
    with session_scope() as session:
        session.query(Prediction).delete()
        session.query(Match).filter(Match.match_id.like("m-score-%")).delete(
            synchronize_session=False)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestSchema:
    def test_predictions_table_has_the_score_columns(self):
        columns = {c["name"] for c in inspect(engine).get_columns("predictions")}
        assert {"pick_score", "score_label"} <= columns

    def test_added_columns_are_declared_for_the_migration(self):
        assert set(_ADDED_COLUMNS["predictions"]) == {"pick_score", "score_label"}

    def test_init_db_readds_a_missing_column(self):
        """An existing database predates these columns; `create_all` never alters
        a table, so `init_db()` must ALTER them in or every query breaks."""
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE predictions DROP COLUMN score_label"))
        except Exception as e:  # older SQLite cannot drop columns
            pytest.skip(f"backend cannot drop a column: {e}")

        assert "score_label" not in {
            c["name"] for c in inspect(engine).get_columns("predictions")}
        init_db()
        assert "score_label" in {
            c["name"] for c in inspect(engine).get_columns("predictions")}


class TestPersistence:
    def test_pipeline_stores_the_gate_score(self):
        AiBettorPipeline()._persist_prediction(a_result())
        with session_scope() as session:
            row = session.query(Prediction).filter(
                Prediction.match_id == "m-score-1").one()
            assert row.pick_score == pytest.approx(86.5)
            assert row.score_label == "STRONG"
            # The brain's own conviction stays in its own column.
            assert row.confidence_score == 74

    def test_a_no_bet_row_is_still_stored_with_its_score(self):
        AiBettorPipeline()._persist_prediction(
            a_result(decision="NO BET", score=41.0, score_label="WEAK"))
        with session_scope() as session:
            row = session.query(Prediction).filter(
                Prediction.match_id == "m-score-1").one()
            assert row.decision == "NO BET"
            assert row.pick_score == pytest.approx(41.0)

    def test_a_missing_score_is_null_not_a_crash(self):
        AiBettorPipeline()._persist_prediction(
            a_result(score=None, score_label=None))
        with session_scope() as session:
            row = session.query(Prediction).filter(
                Prediction.match_id == "m-score-1").one()
            assert row.pick_score in (None, 0.0)
            assert row.score_label is None

    def test_an_overlong_label_is_truncated_to_the_column(self):
        AiBettorPipeline()._persist_prediction(a_result(score_label="X" * 80))
        with session_scope() as session:
            row = session.query(Prediction).filter(
                Prediction.match_id == "m-score-1").one()
            assert len(row.score_label) == 30


class TestApiExposure:
    def test_list_predictions_exposes_the_score(self, client):
        AiBettorPipeline()._persist_prediction(a_result())
        row = client.get("/predictions").json()[0]
        assert row["pick_score"] == pytest.approx(86.5)
        assert row["score_label"] == "STRONG"

    def test_prediction_detail_exposes_the_score(self, client):
        AiBettorPipeline()._persist_prediction(a_result())
        prediction_id = client.get("/predictions").json()[0]["prediction_id"]
        body = client.get(f"/predictions/{prediction_id}").json()
        assert body["pick_score"] == pytest.approx(86.5)
        assert body["score_label"] == "STRONG"
        assert body["reasoning"]

    def test_match_detail_exposes_the_score(self, client):
        with session_scope() as session:
            session.add(Match(
                match_id="m-score-1", home_team="A", away_team="B",
                kickoff=dt.datetime.utcnow(), league="Test League"))
        AiBettorPipeline()._persist_prediction(a_result())
        body = client.get("/matches/m-score-1").json()
        assert body["predictions"][0]["pick_score"] == pytest.approx(86.5)
        assert body["predictions"][0]["score_label"] == "STRONG"

    def test_predictions_can_be_filtered_by_decision(self, client):
        AiBettorPipeline()._persist_prediction(a_result())
        AiBettorPipeline()._persist_prediction(
            a_result(match_id="m-score-2", decision="NO BET"))
        bets = client.get("/predictions", params={"decision": "BET"}).json()
        assert [row["decision"] for row in bets] == ["BET"]
