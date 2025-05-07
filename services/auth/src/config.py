"""
Authentication Service Configuration

This module provides configuration settings for the authentication service.
It loads environment variables and sets default values.
"""
import os
import secrets
from typing import List

from pydantic import BaseSettings, AnyHttpUrl, validator


class Settings(BaseSettings):
    """
    Settings for the authentication service.
    Environment variables will override default values.
    """
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8010"))
    
    # Security settings
    SECRET_KEY: str = os.getenv("JWT_SECRET", secrets.token_hex(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE", "1440"))  # 24 hours
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/aimpact"
    )
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v) -> List[str]:
        """Parse CORS_ORIGINS from comma-separated string if provided as such"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

