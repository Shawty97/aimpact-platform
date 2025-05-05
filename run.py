#!/usr/bin/env python3
"""
AImpact Platform Entry Point

This script initializes and starts the AImpact platform API server.
It sets up the FastAPI application, registers all routers,
configures middleware, and handles startup/shutdown events.
"""

import os
import logging
import asyncio
import uvicorn
from typing import List

from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError

# Import config
from backend.config import settings, configure_logging

# Import routers
from backend.api.routers import agents, workflows, voice

# Set up logging
logger = logging.getLogger("aimpact_api")

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app with configuration
    app = FastAPI(
        title=settings.APP_NAME,
        description="AImpact Platform API - AI Agent and Workflow Management",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.DEBUG
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # --- Register routers ---
    
    # API routers
    app.include_router(
        agents.router,
        prefix=f"{settings.API_PREFIX}/agents",
        tags=["agents"]
    )
    
    app.include_router(
        workflows.router,
        prefix=f"{settings.API_PREFIX}/workflows",
        tags=["workflows"]
    )
    
    app.include_router(
        voice.router,
        prefix=f"{settings.API_PREFIX}/voice",
        tags=["voice"]
    )
    
    # Import the agent store router
    from backend.api.routers import agent_store
    
    app.include_router(
        agent_store.router,
        prefix=f"{settings.API_PREFIX}/agent-store",
        tags=["agent-store"]
    )
    
    # --- Error handlers ---
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )
    
    # --- Startup and shutdown events ---
    
    @app.on_event("startup")
    async def startup_event():
        """Run tasks on application startup."""
        logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        
        # In a real implementation, we would initialize database connections,
        # LLM clients, and other resources here
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run tasks on application shutdown."""
        logger.info(f"Shutting down {settings.APP_NAME}")
        
        # In a real implementation, we would close database connections
        # and perform cleanup here
    
    # Root endpoint
    @app.get("/", tags=["status"])
    async def root():
        """Root endpoint to check API status."""
        return {
            "app_name": settings.APP_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "operational"
        }
    
    @app.get("/health", tags=["status"])
    async def health_check():
        """Health check endpoint for monitoring."""
        # Check the status of various components
        components = {
            "api": "healthy",
            "database": "unknown",  # In a real implementation, check DB connection
            "agent_store": os.environ.get("AGENT_STORE_ENABLED", "false").lower() == "true",
            "knowledge_builder": os.environ.get("KNOWLEDGE_BUILDER_ENABLED", "false").lower() == "true",
            "voice_engine": os.environ.get("VOICE_ENGINE_TYPE", "none") != "none"
        }
        
        # Check LLM providers
        llm_providers = {
            "openai": len(os.environ.get("OPENAI_API_KEY", "")) > 0,
            "anthropic": len(os.environ.get("ANTHROPIC_API_KEY", "")) > 0,
            "cohere": len(os.environ.get("COHERE_API_KEY", "")) > 0,
            "azure_openai": len(os.environ.get("AZURE_OPENAI_API_KEY", "")) > 0
        }
        
        # Determine overall health status
        overall_status = "healthy"
        
        # In a real implementation, if critical components are down, change status
        # if components["database"] != "healthy":
        #     overall_status = "degraded"
        
        return {
            "status": overall_status,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "components": components,
            "llm_providers": llm_providers,
            "timestamp": datetime.now().isoformat()
        }
    
    return app

def start_server():
    """Start the Uvicorn server."""
    uvicorn.run(
        "run:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )

# Create the FastAPI application
app = create_application()

if __name__ == "__main__":
    # Start the server when script is run directly
    start_server()

