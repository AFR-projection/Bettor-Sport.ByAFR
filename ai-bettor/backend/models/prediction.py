from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, JSON, Index
from database.base import BaseModel
import uuid

class Prediction(BaseModel):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("idx_pred_match", "match_id"),
        Index("idx_pred_decision", "decision"),
        Index("idx_pred_conf", "confidence"),
    )
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    market = Column(String(50), nullable=False)
    selection = Column(String(100))
    odds = Column(Float)
    bookmaker = Column(String(100))
    model_probability = Column(Float)
    implied_probability = Column(Float)
    edge = Column(Float)
    ev = Column(Float)
    simulation_h = Column(Float, default=0.0)
    simulation_d = Column(Float, default=0.0)
    simulation_a = Column(Float, default=0.0)
    simulation_target = Column(Float, default=0.0)
    simulation_stability = Column(Float, default=0.0)
    simulation_variance = Column(Float, default=0.0)
    monte_carlo_runs = Column(Integer, default=0)
    confidence = Column(Float, default=0.0)
    score = Column(Float, default=0.0)
    risk_level = Column(String(20), default="UNKNOWN")
    decision = Column(String(10), default="NO_BET")
    minimum_acceptable_odds = Column(Float, nullable=True)
    value_invalidated = Column(Boolean, default=False)
    reasons = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    agent_data_scout_status = Column(String(20))
    agent_quant_status = Column(String(20))
    agent_market_status = Column(String(20))
    agent_simulation_status = Column(String(20))
    agent_risk_status = Column(String(20))
    agent_bettor_brain_status = Column(String(20))

