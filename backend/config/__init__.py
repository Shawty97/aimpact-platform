"""
AImpact Platform Configuration Module

This module handles all configuration settings for the AImpact platform,
including environment variable loading, settings management, and logging configuration.
"""

import os
import logging
import logging.config
import json
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field, validator, SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Determine base directory paths
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
ENV_FILE = CONFIG_DIR / ".env"

# Load environment variables from .env file
load_dotenv(ENV_FILE)

# Define environment enum
class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# Base settings class
class Settings(BaseSettings):
    """Base settings for the AImpact platform."""
    
    # General settings
    APP_NAME: str = "AImpact Platform"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    
    # API settings
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = "sqlite:///aimpact.db"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Security settings
    SECRET_KEY: str = "replace_with_secure_random_string"  # Should be overridden in .env
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    TOKEN_EXPIRATION: int = 86400  # 1 day in seconds (for compatibility)
    ALGORITHM: str = "HS256"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # LLM Provider settings
    OPENAI_API_KEY: SecretStr = Field(default="")
    ANTHROPIC_API_KEY: SecretStr = Field(default="")
    MISTRAL_API_KEY: SecretStr = Field(default="")
    
    # Default models
    DEFAULT_OPENAI_MODEL: str = "gpt-4"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-3-opus-20240229"
    DEFAULT_MISTRAL_MODEL: str = "mistral-large-latest"
    
    # Voice processing settings
    STT_ENGINE: str = "whisper"  # whisper, google, azure
    TTS_ENGINE: str = "openai"   # openai, google, azure, local
    
    # Storage settings
    STORAGE_TYPE: str = "local"  # local, s3, azure
    STORAGE_PATH: str = str(BASE_DIR / "storage")
    
    # Class configuration
    class Config:
        env_file = ENV_FILE
        case_sensitive = True
        env_prefix = ""
        extra = "ignore"  # Allow extra fields in the settings

# LLM Provider settings
class LLMProviderSettings(BaseModel):
    """Settings for language model providers."""
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr
    mistral_api_key: SecretStr
    default_openai_model: str
    default_anthropic_model: str
    default_mistral_model: str
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMProviderSettings":
        """Create LLM settings from main settings."""
        return cls(
            openai_api_key=settings.OPENAI_API_KEY,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            mistral_api_key=settings.MISTRAL_API_KEY,
            default_openai_model=settings.DEFAULT_OPENAI_MODEL,
            default_anthropic_model=settings.DEFAULT_ANTHROPIC_MODEL,
            default_mistral_model=settings.DEFAULT_MISTRAL_MODEL
        )

# Database settings
class DatabaseSettings(BaseModel):
    """Settings for database connections."""
    url: str
    echo: bool
    pool_size: int
    max_overflow: int
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "DatabaseSettings":
        """Create database settings from main settings."""
        return cls(
            url=settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW
        )

# Voice processing settings
class VoiceSettings(BaseModel):
    """Settings for voice processing."""
    stt_engine: str
    tts_engine: str
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "VoiceSettings":
        """Create voice settings from main settings."""
        return cls(
            stt_engine=settings.STT_ENGINE,
            tts_engine=settings.TTS_ENGINE
        )

# Security settings
class SecuritySettings(BaseModel):
    """Settings for security and authentication."""
    secret_key: str
    access_token_expire_minutes: int
    algorithm: str
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "SecuritySettings":
        """Create security settings from main settings."""
        return cls(
            secret_key=settings.SECRET_KEY,
            access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            algorithm=settings.ALGORITHM
        )

def configure_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.LOG_FORMAT,
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "default",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
            "propagate": True,
        },
        "loggers": {
            "aimpact_api": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
            "fastapi": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    
    # Add file handler if a log file is specified
    if settings.LOG_FILE:
        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "json",
            "filename": settings.LOG_FILE,
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
        }
        log_config["root"]["handlers"].append("file")
        log_config["loggers"]["aimpact_api"]["handlers"].append("file")
    
    # Apply logging configuration
    logging.config.dictConfig(log_config)

@lru_cache()
def get_settings() -> Settings:
    """Get application settings, cached for performance."""
    return Settings()

# Initialize settings and logging when module is imported
settings = get_settings()
configure_logging(settings)

# Export specific settings instances
llm_settings = LLMProviderSettings.from_settings(settings)
db_settings = DatabaseSettings.from_settings(settings)
voice_settings = VoiceSettings.from_settings(settings)
security_settings = SecuritySettings.from_settings(settings)

