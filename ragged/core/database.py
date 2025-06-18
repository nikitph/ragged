"""
Database configuration and session management
TODO: Add multi-tenant database routing
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ragged.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# TODO: Add multi-tenant database routing
# def get_tenant_db(tenant_id: str):
#     """Get tenant-specific database session"""
#     pass
