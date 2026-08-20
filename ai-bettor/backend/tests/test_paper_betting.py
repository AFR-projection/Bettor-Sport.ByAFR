"""Tests for the money path: stake accounting, payouts and reported P/L.

These are the bugs that silently eat a bankroll rather than crash, so the
arithmetic is pinned end to end: place → settle → bankroll → history →
/performance. The key invariant is that the stake is withdrawn once when the bet
is placed, so a winner must be credited stake + profit, not the profit alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ai_bettor.db")

import pytest
from fastapi.testclient import TestClient

from backend.database.models import BankrollRecord, Bet, Prediction
from backend.database.session import init_db, session_scope
from backend.main import app
from backend.services.paper_betting import PaperBettingService
from backend.services.settings_service import SettingsService, reset_settings_service


def a_bet(stake: float = 100.0, odds: float = 2.5, **extra) -> dict:
    """A BettorBrain-shaped decision the service will accept."""
    decision = {
        "decision": "BET", "match_id": "m-1", "market": "1X2",
        "selection": "Home", "odds": odds, "stake": stake, "bookmaker": "Pinnacle",
    }
    decision.update(extra)
    return decision


@pytest.fixture(autouse=True)
def clean_money_tables():
    init_db()
    with session_scope() as session:
        session.query(Bet).delete()
        session.query(BankrollRecord).delete()
        session.query(Prediction).delete()
    reset_settings_service()
    yield
    with session_scope() as session:
        session.query(Bet).delete()
        session.query(BankrollRecord).delete()
        session.query(Prediction).delete()
    reset_settings_service()


@pytest.fixture
def service():
    return PaperBettingService(mode="PAPER", initial_bankroll=1000.0)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestPlacingBets:
    def test_stake_leaves_the_bankroll_once(self, service):
        result = service.place_bet(a_bet(stake=100.0, odds=2.5))
        assert result["status"] == "placed"
        assert result["potential_profit"] == pytest.approx(150.0)
        assert result["balance_after"] == pytest.approx(900.0)

        bankroll = service.get_bankroll()
        assert bankroll["current_balance"] == pytest.approx(900.0)
        assert bankroll["total_staked"] == pytest.approx(100.0)

    def test_mode_is_always_paper_by_default(self, service):
        assert service.is_paper is True
        assert service.get_bankroll()["mode"] == "PAPER"

    def test_a_stake_bigger_than_the_bankroll_is_refused(self, service):
        assert service.place_bet(a_bet(stake=5000.0))["status"] == "insufficient_funds"
        assert service.get_bankroll()["current_balance"] == pytest.approx(1000.0)

    def test_nonsense_stakes_and_odds_are_refused(self, service):
        assert service.place_bet(a_bet(stake=0.0))["status"] == "invalid"
        assert service.place_bet(a_bet(odds=1.0))["status"] == "invalid"

    def test_label_is_used_when_there_is_no_selection(self, service):
        service.place_bet(a_bet(selection=None, label="OU 2.5 — Over"))
        assert service.pending_bets()[0]["selection"] == "OU 2.5 — Over"

    def test_initial_bankroll_follows_live_settings(self):
        SettingsService().update({"INITIAL_BANKROLL": 2500.0})
        assert PaperBettingService().get_bankroll()["current_balance"] == pytest.approx(2500.0)


class TestSettlement:
    def test_a_winner_is_credited_stake_plus_profit(self, service):
        """The bug this pins: crediting only the profit ate the stake, so a
        winning 100 @ 2.5 left the bankroll at 1050 instead of 1150."""
        bet_id = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        result = service.settle_bet(bet_id, "win")
        assert result["payout"] == pytest.approx(250.0)
        assert result["balance_after"] == pytest.approx(1150.0)

        bankroll = service.get_bankroll()
        assert bankroll["total_profit"] == pytest.approx(150.0)
        assert bankroll["roi"] == pytest.approx(1.5)

    def test_a_push_returns_the_stake_and_nets_nothing(self, service):
        bet_id = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        assert service.settle_bet(bet_id, "push")["payout"] == pytest.approx(100.0)

        bankroll = service.get_bankroll()
        assert bankroll["current_balance"] == pytest.approx(1000.0)
        assert bankroll["total_profit"] == pytest.approx(0.0)

    def test_a_loser_pays_nothing_back(self, service):
        bet_id = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        assert service.settle_bet(bet_id, "loss")["payout"] == pytest.approx(0.0)

        bankroll = service.get_bankroll()
        assert bankroll["current_balance"] == pytest.approx(900.0)
        assert bankroll["total_profit"] == pytest.approx(-100.0)
        assert bankroll["roi"] == pytest.approx(-1.0)

    def test_a_bet_cannot_be_settled_twice(self, service):
        bet_id = service.place_bet(a_bet())["bet_id"]
        service.settle_bet(bet_id, "win")
        again = service.settle_bet(bet_id, "win")
        assert again["status"] == "already_settled"
        # And the second call did not pay out again.
        assert service.get_bankroll()["current_balance"] == pytest.approx(1150.0)

    def test_outcomes_are_case_insensitive(self, service):
        bet_id = service.place_bet(a_bet())["bet_id"]
        assert service.settle_bet(bet_id, "WIN")["status"] == "settled"

    def test_a_settled_bet_leaves_the_pending_list(self, service):
        bet_id = service.place_bet(a_bet())["bet_id"]
        assert len(service.pending_bets()) == 1
        service.settle_bet(bet_id, "loss")
        assert service.pending_bets() == []

    def test_a_mixed_book_adds_up(self, service):
        # 3 bets of 100: one wins at 2.5 (+150), one pushes (0), one loses (-100).
        winner = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        pushed = service.place_bet(a_bet(stake=100.0, odds=1.9))["bet_id"]
        loser = service.place_bet(a_bet(stake=100.0, odds=3.0))["bet_id"]
        service.settle_bet(winner, "win")
        service.settle_bet(pushed, "push")
        service.settle_bet(loser, "loss")

        bankroll = service.get_bankroll()
        assert bankroll["total_staked"] == pytest.approx(300.0)
        assert bankroll["current_balance"] == pytest.approx(1050.0)
        assert bankroll["total_profit"] == pytest.approx(50.0)
        assert bankroll["roi"] == pytest.approx(50.0 / 300.0)


class TestHistoryProfitLoss:
    def test_pending_bets_report_no_profit_yet(self, service):
        service.place_bet(a_bet())
        assert service.history()[0]["profit_loss"] is None

    def test_settled_bets_report_realised_profit_loss(self, service):
        winner = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        loser = service.place_bet(a_bet(stake=50.0, odds=2.0))["bet_id"]
        pushed = service.place_bet(a_bet(stake=40.0, odds=1.9))["bet_id"]
        service.settle_bet(winner, "win")
        service.settle_bet(loser, "loss")
        service.settle_bet(pushed, "push")

        by_id = {row["bet_id"]: row for row in service.history()}
        assert by_id[winner]["profit_loss"] == pytest.approx(150.0)
        assert by_id[loser]["profit_loss"] == pytest.approx(-50.0)
        assert by_id[pushed]["profit_loss"] == pytest.approx(0.0)

    def test_history_carries_the_fields_the_dashboard_reads(self, service):
        service.place_bet(a_bet())
        row = service.history()[0]
        for key in ("bet_id", "market", "selection", "odds", "stake",
                    "potential_profit", "profit_loss", "status", "result",
                    "created_at", "settled_at"):
            assert key in row, f"{key} missing from history()"

    def test_history_respects_its_limit(self, service):
        for _ in range(4):
            service.place_bet(a_bet(stake=10.0))
        assert len(service.history(limit=2)) == 2


class TestMoneyApi:
    def test_bets_endpoint_serves_the_history(self, client, service):
        bet_id = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        client.post("/bets/settle", params={"bet_id": bet_id, "outcome": "win"})
        rows = client.get("/bets").json()
        assert rows[0]["bet_id"] == bet_id
        assert rows[0]["profit_loss"] == pytest.approx(150.0)

    def test_bankroll_endpoint_reflects_settlement(self, client, service):
        bet_id = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        client.post("/bets/settle", params={"bet_id": bet_id, "outcome": "win"})
        assert client.get("/bankroll").json()["current_balance"] == pytest.approx(1150.0)

    def test_performance_uses_gross_return_not_bare_odds(self, client, service):
        """A winner returns stake x odds; profit is that minus the stake."""
        winner = service.place_bet(a_bet(stake=100.0, odds=2.5))["bet_id"]
        loser = service.place_bet(a_bet(stake=100.0, odds=2.0))["bet_id"]
        service.settle_bet(winner, "win")
        service.settle_bet(loser, "loss")

        body = client.get("/performance").json()
        assert body["total_bets"] == 2
        assert body["wins"] == 1 and body["losses"] == 1
        assert body["total_staked"] == pytest.approx(200.0)
        assert body["total_returned"] == pytest.approx(250.0)
        assert body["profit_loss"] == pytest.approx(50.0)
        assert body["roi"] == pytest.approx(0.25)
        assert body["by_market"]["1X2"]["profit"] == pytest.approx(50.0)

    def test_performance_is_empty_not_broken_without_bets(self, client):
        body = client.get("/performance").json()
        assert body["total_bets"] == 0
        assert body["roi"] == 0
        assert body["win_rate"] == 0
