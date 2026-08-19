from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, JSON, ForeignKey
from database.base import BaseModel; import uuid

class AgentAnalysis(BaseModel):
    __tablename__ = "agent_analyses"
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=False)
    agent = Column(String(50), nullable=False)
    status = Column(String(20), default='PENDING')
    task = Column(String(255))
    execution_time_ms = Column(Float, default=0.0)
    confidence = Column(Float, nullable=True)
    output = Column(JSON, default=dict)
    error = Column(String(500), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
