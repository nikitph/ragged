"""
Base model classes
TODO: Add tenant isolation mixin
"""

from sqlalchemy import Column, Integer, DateTime, func
from ragged.core.database import Base

class BaseModel(Base):
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # TODO: Add tenant isolation
    # tenant_id = Column(String, nullable=False, index=True)
