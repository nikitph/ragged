# config.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parents[1] / ".env")

    R2_BUCKET: str
    R2_ENDPOINT: str
    R2_ACCESS_KEY: str
    R2_SECRET_KEY: str

settings = Settings()