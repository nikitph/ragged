"""
User model
TODO: Add enterprise user features
"""

from sqlalchemy import Column, String, Boolean
from ragged.models.base import BaseModel

class User(BaseModel):
    __tablename__ = "users"
    
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # TODO: Add enterprise fields
    # sso_provider = Column(String)
    # tenant_id = Column(String, ForeignKey("tenants.id"))
