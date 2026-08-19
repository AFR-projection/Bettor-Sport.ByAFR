from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, ForeignKey, JSON
from database.base import BaseModel; import uuid

class RiskAssessment(BaseModel):
    __tablename__ = "risk_assessments"
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    uncertainty_level = Column(String(20), default='UNKNOWN')
    data_quality_score = Column(Float, default=0.0)
    odds_quality = Column(String(20), default='UNKNOWN')
    market_movement = Column(Float, default=0.0)
    exposure_pct = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)
    correlation_score = Column(Float, default=0.0)
    risk_level = Column(String(20), default='UNKNOWN')
    veto = Column(Boolean, default=False)
    veto_reason = Column(String(500), nullable=True)
    details = Column(JSON, default=dict)
