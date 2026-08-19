from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from database.base import BaseModel
import uuid

class Match(BaseModel):
    __tablename__ = "matches"
    __table_args__ = (
        Index("idx_match_start", "commence_time"),
        Index("idx_match_sport", "sport_key"),
        Index("idx_match_home", "home_team"),
        Index("idx_match_away", "away_team"),
    )
    uuid = Column(PGUUID(as_uuid=False), default=lambda: str(uuid.uuid4()),
                  unique=True, nullable=False, index=True)
    api_match_id = Column(String(255), unique=True, nullable=False, index=True)
    sport_key = Column(String(50), nullable=False, index=True)
    sport_title = Column(String(100))
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    commence_time = Column(DateTime(timezone=True), nullable=False, index=True)
    completed = Column(Boolean, default=False)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String(50), default="scheduled")
    data_quality_score = Column(Float, default=0.0)
    metadata = Column(JSON, default=dict)

class Team(BaseModel):
    __tablename__ = "teams"
    name = Column(String(255), unique=True, nullable=False)
    api_key = Column(String(255), unique=True)
    country = Column(String(100))
    league = Column(String(100))
    metadata = Column(JSON, default=dict)

class Bookmaker(BaseModel):
    __tablename__ = "bookmakers"
    api_key = Column(String(100), unique=True, nullable=False)
    title = Column(String(255))
    is_active = Column(Boolean, default=True)

class Market(BaseModel):
    __tablename__ = "markets"
    api_key = Column(String(50), unique=True, nullable=False)
    title = Column(String(100))
    description = Column(String(255))

