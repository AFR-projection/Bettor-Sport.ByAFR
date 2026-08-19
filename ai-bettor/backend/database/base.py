from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, func, Integer, String, Text, Float, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from database.db import Base
import uuid

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class BaseModel(Base, TimestampMixin):
    __abstract__ = True
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(PGUUID(as_uuid=False), default=lambda: str(uuid.uuid4()),
                  unique=True, nullable=False, index=True)

