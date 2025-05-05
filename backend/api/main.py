import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", ".env"))

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aimpact_api")

# Initialize FastAPI app
app = FastAPI(
    title="AImpact Platform API",
    description="API endpoints for the AImpact AI Agent and Workflow Platform",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from routers import agents, workflows, voice

# Include routers
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Welcome to AImpact Platform API", "status": "operational"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting AImpact API server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=os.getenv("DEBUG", "False").lower() == "true")

