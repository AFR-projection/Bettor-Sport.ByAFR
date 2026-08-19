"""Paper betting / bankroll management for AI Bettor.

BETTING_MODE=PAPER (default): all bets are virtual, no real money.
No automatic real-money transactions are ever made by this system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import BankrollRecord, Bet
from backend.database.session import session_scope

logger = logging.getLogger("ai-bettor.paper_betting")

DEFAULT_BANKROLL = 1000.0


class PaperBettingService:
    """Manages virtual bankroll and bet records."""

    def __init__(self, mode: Optional[str] = None, initial_bankroll: float = DEFAULT_BANKROLL):
        settings = get_settings()
        self.mode = (mode or settings.BETTING_MODE).upper()
        self.initial_bankroll = initial_bankroll

    @property
    def is_paper(self) -> bool:
        return self.mode == "PAPER"

    def get_bankroll(self) -> Dict[str, Any]:
        """Get current bankroll state."""
        with session_scope() as session:
            record = session.query(BankrollRecord).order_by(BankrollRecord.id.desc()).first()
            if record is None:
                record = BankrollRecord(current_balance=self.initial_bankroll)
                session.add(record)
                session.flush()
            return {
                "mode": self.mode,
                "current_balance": record.current_balance,
                "total_staked": record.total_staked,
                "total_won": record.total_won,
                "total_profit": record.total_profit,
                "roi": record.roi,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            }

    def place_bet(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Record a virtual bet from a BettorBrain decision."""
        if decision.get("decision") != "BET":
            return {"status": "no_bet", "message": "Decision is not BET, no bet placed"}

        stake = decision.get("stake", 0.0) or 0.0
        odds = decision.get("odds", 0.0)
        if stake <= 0 or odds <= 1:
            return {"status": "invalid", "message": "Invalid stake or odds"}

        with session_scope() as session:
            record = session.query(BankrollRecord).order_by(BankrollRecord.id.desc()).first()
            if record is None:
                record = BankrollRecord(current_balance=self.initial_bankroll)
                session.add(record)
                session.flush()

            if record.current_balance < stake:
                return {"status": "insufficient_funds", "message": "Bankroll too low for this stake"}

            bet = Bet(
                match_id=decision.get("match_id"),
                decision="BET",
                market=decision.get("market"),
                selection=decision.get("selection"),
                odds=odds,
                bookmaker=decision.get("bookmaker"),
                stake=stake,
                potential_profit=stake * (odds - 1),
                status="pending",
            )
            session.add(bet)

            record.current_balance -= stake
            record.total_staked += stake
            session.flush()

            return {
                "status": "placed",
                "bet_id": bet.bet_id,
                "stake": stake,
                "odds": odds,
                "potential_profit": bet.potential_profit,
                "balance_after": record.current_balance,
                "mode": self.mode,
            }

    def settle_bet(self, bet_id: str, outcome: str) -> Dict[str, Any]:
        """Settle a virtual bet. outcome: win / loss / push."""
        outcome = outcome.lower()
        if outcome not in ("win", "loss", "push"):
            return {"status": "invalid", "message": "Outcome must be win/loss/push"}

        with session_scope() as session:
            bet = session.query(Bet).filter(Bet.bet_id == bet_id).first()
            if bet is None:
                return {"status": "not_found", "message": "Bet not found"}
            if bet.status != "pending":
                return {"status": "already_settled", "message": f"Bet already settled ({bet.status})"}

            record = session.query(BankrollRecord).order_by(BankrollRecord.id.desc()).first()

            if outcome == "win":
                payout = bet.potential_profit
            elif outcome == "push":
                payout = bet.stake  # stake returned
            else:
                payout = 0.0

            bet.status = "settled"
            bet.result = outcome
            bet.settled_at = datetime.utcnow()

            if record:
                record.current_balance += payout
                record.total_won += payout
                record.total_profit = record.total_won - record.total_staked
                record.roi = (record.total_profit / record.total_staked) if record.total_staked > 0 else 0.0

            return {
                "status": "settled",
                "bet_id": bet_id,
                "outcome": outcome,
                "payout": payout,
                "balance_after": record.current_balance if record else None,
            }

    def pending_bets(self) -> List[Dict[str, Any]]:
        """List pending virtual bets."""
        with session_scope() as session:
            bets = session.query(Bet).filter(Bet.status == "pending").all()
            return [
                {
                    "bet_id": b.bet_id,
                    "match_id": b.match_id,
                    "market": b.market,
                    "selection": b.selection,
                    "odds": b.odds,
                    "stake": b.stake,
                    "potential_profit": b.potential_profit,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                }
                for b in bets
            ]

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List bet history (settled + pending)."""
        with session_scope() as session:
            bets = session.query(Bet).order_by(Bet.created_at.desc()).limit(limit).all()
            return [
                {
                    "bet_id": b.bet_id,
                    "match_id": b.match_id,
                    "decision": b.decision,
                    "market": b.market,
                    "selection": b.selection,
                    "odds": b.odds,
                    "stake": b.stake,
                    "status": b.status,
                    "result": b.result,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                }
                for b in bets
            ]


def get_paper_betting() -> PaperBettingService:
    return PaperBettingService()