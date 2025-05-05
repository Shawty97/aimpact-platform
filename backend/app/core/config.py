from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AImpact Platform"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "aimpact"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    class Config:
        case_sensitive = True
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.SQLALCHEMY_DATABASE_URI = (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
        )

settings = Settings() 