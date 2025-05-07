#!/bin/bash

# AImpact Platform Startup Script
# This script initializes the AImpact platform, sets up the environment,
# and starts all necessary services.

set -e  # Exit on error

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print colored messages
print_header() {
    echo -e "${BLUE}${BOLD}$1${NC}"
    echo -e "${BLUE}${BOLD}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

print_section() {
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}$(printf '-%.0s' $(seq 1 ${#1}))${NC}"
}

print_step() {
    echo -e "${GREEN}➤ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check service health
check_health() {
    service_name=$1
    url=$2
    max_retries=$3
    interval=$4
    
    print_step "Checking health of $service_name..."
    
    retries=0
    while [ $retries -lt $max_retries ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$service_name is healthy"
            return 0
        else
            retries=$((retries+1))
            if [ $retries -lt $max_retries ]; then
                print_info "Waiting for $service_name to be ready... (attempt $retries/$max_retries)"
                sleep $interval
            fi
        fi
    done
    
    print_error "$service_name is not responding after $max_retries attempts"
    return 1
}

# Print welcome message
clear
print_header "AImpact Platform Startup"
echo -e "This script will set up and start the AImpact AI platform.\n"

# Check prerequisites
print_section "Checking Prerequisites"

# Check if Docker is installed
if ! command_exists docker; then
    print_error "Docker is not installed"
    print_info "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
else
    print_success "Docker is installed"
fi

# Check if Docker Compose is installed
if ! command_exists "docker compose"; then
    print_error "Docker Compose is not installed"
    print_info "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
else
    print_success "Docker Compose is installed"
fi

# Check if Git is installed
if ! command_exists git; then
    print_error "Git is not installed"
    print_info "Please install Git from https://git-scm.com/downloads"
    exit 1
else 
    print_success "Git is installed"
fi

# Set up directory structure
print_section "Setting Up Directory Structure"

print_step "Creating necessary directories..."
mkdir -p models/tts/coqui
mkdir -p models/stt/vosk
mkdir -p services/workflow-engine
mkdir -p services/voice-service
mkdir -p services/llm-orchestrator
mkdir -p services/knowledge-builder
mkdir -p services/agent-store
mkdir -p init-scripts
mkdir -p logs
mkdir -p data

print_success "Directories created successfully"

# Initialize directory for each service
for service in workflow-engine voice-service llm-orchestrator knowledge-builder agent-store; do
    if [ ! -f "services/$service/Dockerfile" ]; then
        print_step "Creating basic structure for $service..."
        mkdir -p "services/$service/src"
        touch "services/$service/requirements.txt"
        
        # Create a basic Dockerfile for the service
        cat > "services/$service/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
        
        # Create a minimal requirements.txt
        cat > "services/$service/requirements.txt" << 'EOF'
fastapi>=0.115.0
uvicorn>=0.34.0
pydantic>=2.0.0
redis>=5.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.7
httpx>=0.25.0
EOF
        
        # Create a basic main.py for each service
        mkdir -p "services/$service/src"
        cat > "services/$service/src/main.py" << EOF
"""
AImpact $service

This service provides advanced capabilities for the AImpact platform.
"""

import os
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="AImpact $service",
    description="Advanced $service for AImpact platform",
    version="0.1.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "$service"}

@app.get("/api/info")
async def get_info():
    """Get information about the service"""
    return {
        "name": "$service",
        "version": "0.1.0",
        "description": "Advanced $service for AImpact platform",
    }
EOF
        
        print_success "Basic structure for $service created"
    fi
done

# Download models
print_section "Downloading Required Models"

# Download Vosk model for speech recognition if not already downloaded
if [ ! -d "models/stt/vosk/model-en-us-0.22" ]; then
    print_step "Downloading Vosk speech recognition model..."
    if [ ! -f "models/stt/vosk/model-en-us-0.22.zip" ]; then
        curl -L -o models/stt/vosk/model-en-us-0.22.zip https://alphacephei.com/vosk/models/model-en-us-0.22.zip
    fi
    
    print_step "Extracting Vosk model..."
    (cd models/stt/vosk && unzip -q model-en-us-0.22.zip)
    print_success "Vosk model downloaded and extracted"
else
    print_success "Vosk model already downloaded"
fi

# Check if .env file exists, create it if not
if [ ! -f ".env" ]; then
    print_section "Creating Environment Configuration"
    print_step "Generating .env file..."
    
    cat > .env << 'EOF'
# AImpact Platform Environment Variables

# Core Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true
API_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/aimpact
VECTOR_DB_URL=postgresql://postgres:postgres@postgres:5432/aimpact
VECTOR_DB_TYPE=postgres
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30

# Redis
REDIS_URL=redis://redis:6379/0

# LLM Settings
DEFAULT_LLM_PROVIDER=local
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=mistral
LOCAL_LLM_MODEL=mistral-7b-instruct

# Voice Settings
VOICE_ENGINE_TYPE=coqui
SPEECH_RECOGNITION_PROVIDER=vosk
TTS_MODEL_PATH=/app/models/tts/coqui
STT_MODEL_PATH=/app/models/stt/vosk
USE_LOCAL_TTS=true
USE_LOCAL_STT=true

# Service URLs
WORKFLOW_ENGINE_URL=http://workflow-engine:8050/api
VOICE_SERVICE_URL=http://voice-service:8060/api
LLM_ORCHESTRATOR_URL=http://llm-orchestrator:8070/api

# Feature Flags
AGENT_STORE_ENABLED=true
KNOWLEDGE_BUILDER_ENABLED=true

# Monitoring
TELEMETRY_ENABLED=true
LOGGING_FORMAT=json
METRICS_ENABLED=true
TRACING_ENABLED=true
EOF
    
    print_success ".env file created successfully"
else
    print_success ".env file already exists"
fi

# Create PostgreSQL initialization scripts
print_section "Setting Up Database"

print_step "Creating database initialization scripts..."

# Create script for extensions
cat > init-scripts/01_init_extensions.sql << 'EOF'
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";
EOF

# Create script for schemas and tables
cat > init-scripts/02_init_schemas.sql << 'EOF'
-- Create schemas for different components
CREATE SCHEMA IF NOT EXISTS workflow_engine;
CREATE SCHEMA IF NOT EXISTS voice_service;
CREATE SCHEMA IF NOT EXISTS llm_orchestrator;
CREATE SCHEMA IF NOT EXISTS knowledge_builder;
CREATE SCHEMA IF NOT EXISTS agent_store;

-- Create common tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    settings JSONB DEFAULT '{}'
);

-- Workflow Engine tables
CREATE TABLE IF NOT EXISTS workflow_engine.workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_template BOOLEAN DEFAULT FALSE,
    definition JSONB NOT NULL,
    tags TEXT[]
);

-- Voice Service tables
CREATE TABLE IF NOT EXISTS voice_service.voice_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    parameters JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- LLM Orchestrator tables
CREATE TABLE IF NOT EXISTS llm_orchestrator.providers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    provider_type VARCHAR(50) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    config JSONB DEFAULT '{}'
);

-- Knowledge Builder tables
CREATE TABLE IF NOT EXISTS knowledge_builder.knowledge_bases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    vector_store_type VARCHAR(50) DEFAULT 'postgres',
    settings JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS knowledge_builder.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_base_id UUID REFERENCES knowledge_builder.knowledge_bases(id),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    source VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding VECTOR(1536)
);

-- Agent Store tables
CREATE TABLE IF NOT EXISTS agent_store.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL,
    author_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    is_public BOOLEAN DEFAULT FALSE,
    download_count INTEGER DEFAULT 0,
    rating FLOAT DEFAULT 0.0,
    config JSONB NOT NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON knowledge_builder.documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
EOF

print_success "Database scripts created successfully"

# Start the platform
print_section "Starting AImpact Platform"

print_step "Building and starting services..."
docker compose build --progress=plain
docker compose up -d

# Wait for services to be ready
print_section "Verifying Services"

sleep 10 # Initial wait to give services time to start

# Check core services
check_health "API" "http://localhost:8000/health" 5 10
check_health "Workflow Engine" "http://localhost:8050/health" 5 10
check_health "Voice Service" "http://localhost:8060/health" 5 10
check_health "LLM Orchestrator" "http://localhost:8070/health" 5 10
check_health "Knowledge Builder" "http://localhost:8080/health" 5 10
check_health "Agent Store" "http://localhost:8090/health" 5 10

print_section "AImpact Platform Status"
docker compose ps

# Print success message
print_header "AImpact Platform Successfully Started!"
echo -e "\nThe following services are available:"
echo -e "  • API:              ${BOLD}http://localhost:8000${NC}"
echo -e "  • Workflow Engine:  ${BOLD}http://localhost:8050${NC}"
echo -e "  • Voice Service:    ${BOLD}http://localhost:8060${NC}"
echo -e "  • LLM Orchestrator: ${BOLD}http://localhost:8070${NC}"
echo -e "  • Knowledge Builder:${BOLD}http://localhost:8080${NC}"
echo -e "  • Agent Store:      ${BOLD}http://localhost:8090${NC}"
echo -e "  • Ollama:           ${BOLD}http://localhost:11434${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Explore the API documentation at ${BOLD}http://localhost:8000/docs${NC}"
echo -e "2. Create your first workflow using the Workflow Engine"
echo -e "3. Test voice interactions with the Voice Service"
echo -e "4. Try different LLMs with the LLM Orchestrator\n"

echo -e "${GREEN}To stop the platform:${NC} docker compose down

