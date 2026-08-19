from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, Index
from database.base import BaseModel
import uuid

class OddsSnapshot(BaseModel):
    __tablename__ = "odds_snapshots"
    __table_args__ = (
        Index("idx_snap_match", "match_id"),
        Index("idx_snap_bookmaker", "bookmaker"),
        Index("idx_snap_market", "market"),
        Index("idx_snap_composite", "match_id", "bookmaker", "market", "selection"),
    )
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker = Column(String(100), nullable=False)
    market = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    line = Column(Float, nullable=True)
    odds = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    data_quality = Column(Integer, default=100)

