#!/usr/bin/env python3
"""
Health check script for the AImpact Authentication Service.

This script checks the health of the service and its dependencies:
- Database connection
- Redis connection
- API endpoints
- System resources

Usage:
    python health_check.py [options]

Options:
    --verbose       Show detailed information
    --output=json   Output in JSON format
    --endpoint=URL  API endpoint to check (default: http://localhost:8000)
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
import psutil
import redis
import sqlalchemy as sa
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

try:
    from core.config import settings
except ImportError:
    # Fallback if settings can't be imported
    class Settings:
        SQLALCHEMY_DATABASE_URI = os.environ.get(
            "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/aimpact_auth"
        )
        REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
        JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "test_secret")
        JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")

    settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("health_check")

# Rich console for pretty output
console = Console()


# Models
class HealthStatus(BaseModel):
    """Health status information."""
    status: str
    details: Dict[str, Union[str, Dict]]
    timestamp: str


class ComponentStatus(BaseModel):
    """Component status information."""
    status: str
    message: str
    latency_ms: Optional[float] = None
    details: Optional[Dict] = None


# Health Check Functions
async def check_database_connection() -> ComponentStatus:
    """Check database connection."""
    start_time = time.time()
    
    try:
        # Create engine and session
        engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Try to connect and run a simple query
        async with async_session() as session:
            # Run a simple query
            result = await session.execute(sa.text("SELECT 1"))
            value = result.scalar()
            
            if value != 1:
                return ComponentStatus(
                    status="error",
                    message="Database connection error: unexpected result",
                    latency_ms=(time.time() - start_time) * 1000
                )
        
        # Dispose engine
        await engine.dispose()
        
        return ComponentStatus(
            status="ok",
            message="Database connection successful",
            latency_ms=(time.time() - start_time) * 1000
        )
    
    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Database connection error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


def check_redis_connection() -> ComponentStatus:
    """Check Redis connection."""
    start_time = time.time()
    
    try:
        # Parse Redis URL
        redis_url = settings.REDIS_URL
        
        # Create Redis client and ping
        redis_client = redis.from_url(redis_url)
        result = redis_client.ping()
        
        if not result:
            return ComponentStatus(
                status="error",
                message="Redis connection error: ping failed",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        return ComponentStatus(
            status="ok",
            message="Redis connection successful",
            latency_ms=(time.time() - start_time) * 1000
        )
    
    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"Redis connection error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


async def check_api_endpoint(endpoint: str) -> ComponentStatus:
    """Check API endpoint."""
    start_time = time.time()
    
    try:
        # Create HTTP client
        async with httpx.AsyncClient() as client:
            # Send request
            response = await client.get(f"{endpoint}/health")
            
            # Parse response
            if response.status_code != 200:
                return ComponentStatus(
                    status="error",
                    message=f"API endpoint error: status code {response.status_code}",
                    latency_ms=(time.time() - start_time) * 1000,
                    details={"status_code": response.status_code, "response": response.text}
                )
            
            try:
                data = response.json()
                return ComponentStatus(
                    status="ok",
                    message="API endpoint is healthy",
                    latency_ms=(time.time() - start_time) * 1000,
                    details=data
                )
            except Exception:
                return ComponentStatus(
                    status="warning",
                    message="API endpoint returned non-JSON response",
                    latency_ms=(time.time() - start_time) * 1000,
                    details={"response": response.text}
                )
    
    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"API endpoint error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


def check_system_resources() -> ComponentStatus:
    """Check system resources."""
    start_time = time.time()
    
    try:
        # Get system stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Collect stats
        stats = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_free_gb": disk.free / (1024 * 1024 * 1024)
        }
        
        # Determine status
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "critical"
            message = "System resources critical"
        elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
            status = "warning"
            message = "System resources warning"
        else:
            status = "ok"
            message = "System resources ok"
        
        return ComponentStatus(
            status=status,
            message=message,
            latency_ms=(time.time() - start_time) * 1000,
            details=stats
        )
    
    except Exception as e:
        return ComponentStatus(
            status="error",
            message=f"System resources check error: {str(e)}",
            latency_ms=(time.time() - start_time) * 1000
        )


def check_network_connectivity() -> ComponentStatus:
    """Check network connectivity."""
    start_time = time.time()
    
    try:
        # Create socket and connect to host
        host = "8.8.8.8"  # Google DNS
        port = 53  # DNS port
        
        sock = socket.socket(

