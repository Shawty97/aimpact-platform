from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title="AImpact Platform",
    description="A comprehensive AI platform for agent technology and quantum integration",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AImpact Platform",
        "version": "0.1.0",
        "status": "operational"
    }

# Import and include routers
# from app.api.v1.endpoints import agents, workflows, knowledge_base
# app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
# app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
# app.include_router(knowledge_base.router, prefix="/api/v1/knowledge", tags=["knowledge"]) 