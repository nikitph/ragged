"""
Core configuration settings
TODO: Add multi-tenant configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Basic settings
    app_name: str = "Ragged"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql://user:password@localhost/ragged"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Storage
    storage_backend: str = "local"  # local, s3, minio
    storage_path: str = "./storage"
    
    # TODO: Add enterprise settings
    # multi_tenant_enabled: bool = False
    # isolation_strategy: str = "shared_db_shared_schema"
    
    class Config:
        env_file = ".env"

settings = Settings()
