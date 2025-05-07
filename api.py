import os
import sys
import time
import platform
import psutil
import redis
import requests
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment configuration
class Settings(BaseSettings):
    app_name: str = "AImpact API"
    version: str = "0.1.0"
    debug: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    database_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().version,
    debug=get_settings().debug,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check response model
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    system_info: Dict[str, Any]
    dependencies: Dict[str, Any]

# Redis connection check
def check_redis_connection(settings: Settings) -> Dict[str, Any]:
    redis_status = {
        "status": "unavailable",
        "details": None,
        "error": None
    }
    
    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            socket_timeout=2
        )
        
        # Test the connection
        ping_response = r.ping()
        redis_status["status"] = "available" if ping_response else "unavailable"
        redis_status["details"] = {"ping": ping_response}
    except Exception as e:
        redis_status["status"] = "error"
        redis_status["error"] = str(e)
    
    return redis_status

# External service check
def check_external_services() -> Dict[str, Any]:
    services = {
        "google": {
            "status": "unavailable",
            "latency_ms": None,
            "error": None
        },
        "openai": {
            "status": "unavailable",
            "latency_ms": None,
            "error": None
        }
    }
    
    # Check Google connectivity
    try:
        start_time = time.time()
        response = requests.get("https://www.google.com", timeout=5)
        latency = (time.time() - start_time) * 1000
        
        services["google"]["status"] = "available" if response.status_code == 200 else "unavailable"
        services["google"]["latency_ms"] = round(latency, 2)
    except Exception as e:
        services["google"]["status"] = "error"
        services["google"]["error"] = str(e)
    
    # Check OpenAI connectivity
    try:
        start_time = time.time()
        response = requests.get("https://api.openai.com", timeout=5)
        latency = (time.time() - start_time) * 1000
        
        services["openai"]["status"] = "available" if response.status_code == 200 else "unavailable"
        services["openai"]["latency_ms"] = round(latency, 2)
    except Exception as e:
        services["openai"]["status"] = "error"
        services["openai"]["error"] = str(e)
    
    return services

# Database connection check (placeholder - implement based on your DB)
def check_database_connection(settings: Settings) -> Dict[str, Any]:
    db_status = {
        "status": "not_configured",
        "details": None,
        "error": None
    }
    
    if not settings.database_url:
        return db_status
    
    try:
        # This is a placeholder - implement actual DB connection check
        # based on your database type (PostgreSQL, MySQL, etc.)
        db_status["status"] = "not_implemented"
        db_status["details"] = "Database check not implemented yet"
    except Exception as e:
        db_status["status"] = "error"
        db_status["error"] = str(e)
    
    return db_status

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    # System information
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_percent": psutil.cpu_percent(),
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024 ** 3), 2),  # GB
            "available": round(psutil.virtual_memory().available / (1024 ** 3), 2),  # GB
            "percent_used": psutil.virtual_memory().percent
        },
        "disk": {
            "total": round(psutil.disk_usage('/').total / (1024 ** 3), 2),  # GB
            "free": round(psutil.disk_usage('/').free / (1024 ** 3), 2),  # GB
            "percent_used": psutil.disk_usage('/').percent
        }
    }
    
    # Check dependencies
    dependencies = {
        "redis": check_redis_connection(settings),
        "database": check_database_connection(settings),
        "external_services": check_external_services()
    }
    
    overall_status = "healthy"
    if dependencies["redis"]["status"] not in ["available"]:
        overall_status = "degraded"
    
    # If using database and it's configured but not available
    if settings.database_url and dependencies["database"]["status"] not in ["available", "not_implemented"]:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.version,
        timestamp=time.time(),
        system_info=system_info,
        dependencies=dependencies
    )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    return {"message": "Welcome to the AImpact API", "version": get_settings().version}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

