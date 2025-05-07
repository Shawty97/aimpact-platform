"""
Authentication Service Main Entry Point

This module serves as the main entry point for the AImpact authentication service.
It initializes the FastAPI application and includes all routes.
"""
import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import router

# Set up logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("auth_service")

# Create FastAPI app
app = FastAPI(
    title="AImpact Authentication Service",
    description="Authentication and user management for the AImpact platform",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/auth")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness_check():
    """Readiness check endpoint that verifies database connection"""
    # In the future, this will check database connection
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

