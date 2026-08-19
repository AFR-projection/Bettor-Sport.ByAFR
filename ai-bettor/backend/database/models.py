"""SQLAlchemy models for AI Bettor database."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, BigInteger, Boolean, Column, DateTime, Float, ForeignKey,
    Identity, Integer, String, Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Match(Base):
    __tablename__ = "matches"

    match_id = Column(String(50), primary_key=True)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    kickoff = Column(DateTime(timezone=True), nullable=False)
    league = Column(String(100), nullable=False)
    sport = Column(String(50), nullable=False, default="football")
    status = Column(String(20), nullable=False, default="upcoming")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Bookmaker(Base):
    __tablename__ = "bookmakers"

    bookmaker_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Market(Base):
    __tablename__ = "markets"

    market_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    key = Column(String(50), nullable=False, unique=True)
    sport = Column(String(50), nullable=False, default="football")
    is_active = Column(Boolean, nullable=False, default=True)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(BigInteger, Identity(), primary_key=True)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    market = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    line = Column(String(20), nullable=True)
    odds = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String(50), primary_key=True, default=_new_id)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=False)
    decision = Column(String(10), nullable=False)
    market = Column(String(50), nullable=True)
    selection = Column(String(100), nullable=True)
    odds = Column(Float, nullable=True)
    bookmaker = Column(String(50), nullable=True)
    model_probability = Column(Float, nullable=False)
    implied_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    ev = Column(Float, nullable=False)
    confidence_score = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Simulation(Base):
    __tablename__ = "simulations"

    simulation_id = Column(String(50), primary_key=True, default=_new_id)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=False)
    home_win_probability = Column(Float, nullable=False)
    draw_probability = Column(Float, nullable=False)
    away_win_probability = Column(Float, nullable=False)
    handicap_probability = Column(Float, nullable=True)
    over_probability = Column(Float, nullable=True)
    under_probability = Column(Float, nullable=True)
    variance = Column(Float, nullable=False)
    stability = Column(Float, nullable=False)
    simulation_count = Column(Integer, nullable=False, default=20000)
    random_seed = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class AgentAnalysis(Base):
    __tablename__ = "agent_analyses"

    analysis_id = Column(String(50), primary_key=True, default=_new_id)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    output = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    assessment_id = Column(String(50), primary_key=True, default=_new_id)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=True)
    bankroll_risk_percent = Column(Float, nullable=False)
    exposure = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    correlation_risk = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    veto_decision = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Bet(Base):
    __tablename__ = "bets"

    bet_id = Column(String(50), primary_key=True, default=_new_id)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=True)
    decision = Column(String(10), nullable=False)
    market = Column(String(50), nullable=True)
    selection = Column(String(100), nullable=True)
    odds = Column(Float, nullable=True)
    bookmaker = Column(String(50), nullable=True)
    stake = Column(Float, nullable=False)
    potential_profit = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    settled_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(String(20), nullable=True)


class BankrollRecord(Base):
    __tablename__ = "bankroll"

    id = Column(Integer, Identity(), primary_key=True)
    current_balance = Column(Float, nullable=False, default=1000.0)
    total_staked = Column(Float, nullable=False, default=0.0)
    total_won = Column(Float, nullable=False, default=0.0)
    total_profit = Column(Float, nullable=False, default=0.0)
    roi = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemLog(Base):
    __tablename__ = "system_logs"

    log_id = Column(BigInteger, Identity(), primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    service = Column(String(50), nullable=False)
    agent = Column(String(50), nullable=True)
    match_id = Column(String(50), nullable=True)
    action = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    latency = Column(Float, nullable=True)
    error_details = Column(Text, nullable=True)