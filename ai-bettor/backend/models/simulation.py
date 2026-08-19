from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, JSON
from database.base import BaseModel
import uuid

class SimulationResult(BaseModel):
    __tablename__ = "simulations"
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    market = Column(String(50), nullable=False)
    selection = Column(String(100))
    num_simulations = Column(Integer, default=0)
    home_win_prob = Column(Float, default=0.0)
    draw_prob = Column(Float, default=0.0)
    away_win_prob = Column(Float, default=0.0)
    handicap_prob = Column(Float, default=0.0)
    over_prob = Column(Float, default=0.0)
    under_prob = Column(Float, default=0.0)
    distribution = Column(JSON, default=list)
    variance = Column(Float, default=0.0)
    stability = Column(Float, default=0.0)
    seed = Column(Integer, nullable=True)
