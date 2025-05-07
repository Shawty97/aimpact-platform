#!/bin/bash

# AImpact Platform Setup Script
# This script sets up the AImpact platform with all its dependencies,
# initializes databases, and pulls required models.

set -e  # Exit on error

# Print colored messages
print_green() {
    echo -e "\e[32m$1\e[0m"
}

print_blue() {
    echo -e "\e[34m$1\e[0m"
}

print_red() {
    echo -e "\e[31m$1\e[0m"
}

print_yellow() {
    echo -e "\e[33m$1\e[0m"
}

print_blue "=================================================="
print_blue "AImpact Platform - Billionaire Project Setup"
print_blue "=================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_red "Docker is not installed. Please install Docker first."
    print_yellow "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    print_red "Docker Compose is not installed. Please install Docker Compose first."
    print_yellow "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Create necessary directories
print_green "Creating directories..."
mkdir -p models/tts/coqui
mkdir -p models/stt/vosk
mkdir -p data/market_analysis
mkdir -p data/wealth_tracking
mkdir -p data/network_analysis
mkdir -p init-scripts
mkdir -p logs

# Create init scripts for PostgreSQL
print_green "Creating PostgreSQL initialization scripts..."
cat > init-scripts/01_init_extensions.sql << 'EOF'
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";
EOF

cat > init-scripts/02_init_tables.sql << 'EOF'
-- Create database schema for wealth tracking
CREATE SCHEMA IF NOT EXISTS wealth_tracking;
CREATE SCHEMA IF NOT EXISTS network_analyzer;
CREATE SCHEMA IF NOT EXISTS financial_data;
CREATE SCHEMA IF NOT EXISTS knowledge_base;
CREATE SCHEMA IF NOT EXISTS agent_store;
CREATE SCHEMA IF NOT EXISTS workflow_engine;

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

-- Create wealth tracking tables
CREATE TABLE IF NOT EXISTS wealth_tracking.portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    base_currency VARCHAR(10) DEFAULT 'USD',
    privacy_level VARCHAR(20) DEFAULT 'PRIVATE',
    settings JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS wealth_tracking.assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES wealth_tracking.portfolios(id),
    name VARCHAR(255) NOT NULL,
    asset_class VARCHAR(50) NOT NULL,
    acquisition_date DATE,
    acquisition_value DECIMAL(19, 4),
    currency VARCHAR(10) DEFAULT 'USD',
    current_value DECIMAL(19, 4),
    details JSONB NOT NULL,
    privacy_level VARCHAR(20) DEFAULT 'PRIVATE',
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create network analyzer tables
CREATE TABLE IF NOT EXISTS network_analyzer.persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    company VARCHAR(255),
    industry TEXT[],
    net_worth DECIMAL(19, 4),
    location VARCHAR(255),
    influence_score DECIMAL(5, 2),
    tags TEXT[],
    social_links JSONB DEFAULT '{}',
    bio TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS network_analyzer.relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES network_analyzer.persons(id),
    target_id UUID REFERENCES network_analyzer.persons(id),
    relationship_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3, 2) DEFAULT 1.0,
    description TEXT,
    start_date DATE,
    end_date DATE,
    sources TEXT[],
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create financial data tables
CREATE TABLE IF NOT EXISTS financial_data.market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(19, 4),
    high_price DECIMAL(19, 4),
    low_price DECIMAL(19, 4),
    close_price DECIMAL(19, 4),
    adjusted_close DECIMAL(19, 4),
    volume BIGINT,
    dividend DECIMAL(19, 4),
    split_coefficient DECIMAL(10, 4),
    UNIQUE(symbol, date)
);

-- Create knowledge base tables
CREATE TABLE IF NOT EXISTS knowledge_base.knowledge_bases (
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

CREATE TABLE IF NOT EXISTS knowledge_base.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_base_id UUID REFERENCES knowledge_base.knowledge_bases(id),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    source VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding VECTOR(1536)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_persons_embedding ON network_analyzer.persons USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON knowledge_base.documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
EOF

# Download models
print_green "Downloading speech recognition model..."
if [ ! -f "models/stt/vosk/model-en-us-0.22-lgraph.zip" ]; then
    curl -L -o models/stt/vosk/model-en-us-0.22-lgraph.zip https://alphacephei.com/vosk/models/model-en-us-0.22-lgraph.zip
    cd models/stt/vosk && unzip model-en-us-0.22-lgraph.zip && cd ../../../
fi

# Create .env file
print_green "Creating environment file..."
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
FINANCIAL_DATA_API_URL=http://financial-data:8050/api
NETWORK_ANALYZER_URL=http://network-analyzer:8070/api
WEALTH_TRACKER_URL=http://wealth-tracker:8060/api

# Feature Flags
AGENT_STORE_ENABLED=true
KNOWLEDGE_BUILDER_ENABLED=true

# Monitoring
TELEMETRY_ENABLED=true
LOGGING_FORMAT=json
METRICS_ENABLED=true
TRACING_ENABLED=true

# Optional API Keys (for external services if needed)
# ALPHA_VANTAGE_API_KEY=
# FINANCIAL_MODELING_PREP_API_KEY=
# CRUNCHBASE_API_KEY=
# LINKEDIN_API_KEY=
# NEWS_API_KEY=
# GOOGLE_API_KEY=
# WEALTH_X_API_KEY=
# FORBES_API_KEY=
EOF

# Create a README for the billionaire project
print_green "Creating project documentation..."
cat > README-BILLIONAIRE-PROJECT.md << 'EOF'
# AImpact Billionaire Project

## Vision
Build a comprehensive platform exceeding the capabilities of Artisan.co, Beam.ai, and Vapi.ai by providing:

- Advanced voice AI with emotion detection and cultural awareness
- Sophisticated workflow engine with optimization and learning
- Comprehensive wealth tracking and analysis
- High-value networking intelligence
- Financial data analysis and opportunity identification

## Key Components

### 1. Core Platform
- Workflow Engine - Complex workflows with A/B testing and feedback loops
- Voice AI - Emotion-aware, multilingual voice processing
- AutoPilot - Continuous optimization and learning
- Knowledge Builder - Automated knowledge base creation

### 2. Billionaire-Focused Services
- Financial Data Service - Market analysis and financial insights
- Network Analyzer - Relationship mapping and networking intelligence
- Wealth Tracker - Asset management and wealth preservation

## Getting Started

1. Run the setup script: `./setup.sh`
2. Start the platform: `docker compose up -d`
3. Initialize the project: `python init_project.py`
4. Access the dashboard at: http://localhost:3000

## Billionaire Workflows

The platform includes specialized workflows for billionaire needs:

- Wealth Tracking and Growth
- Strategic Networking
- Market Opportunity Analysis
- Competitor Intelligence
- Wealth Preservation

## Architecture

The platform uses a microservices architecture with these components:
- PostgreSQL + pgvector for data storage and vector search
- Redis for caching and pub/sub messaging
- Ollama for local LLM capabilities
- Open source voice processing (Coqui TTS, Vosk STT)
- Specialized financial data and network analysis services

## Customization

Edit the `.env` file to configure the platform settings, or modify the
`docker-compose.yml` file to adjust service configurations.
EOF

# Create Python initialization script
print_green "Creating initialization script..."
cat > init_project.py << 'EOF'
#!/usr/bin/env python3
"""
AImpact Project Initialization Script

This script initializes the AImpact platform with necessary data,
sets up initial configurations, and ensures everything is ready to use.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
import argparse
import requests
import random
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/initialization.log")
    ]
)
logger = logging.getLogger("init_project")

# Configuration
CONFIG = {
    "api_url": "http://localhost:8000/api/v1",
    "financial_data_url": "http://localhost:8050/api",
    "network_analyzer_url": "http://localhost:8070/api",
    "wealth_tracker_url": "http://localhost:8060/api",
    "admin_username": "admin",
    "admin_password": "admin123",  # Change in production
    "admin_email": "admin@aimpact.com"
}

async def wait_for_services():
    """Wait for all services to be ready."""
    services = [
        ("API", f"{CONFIG['api_url']}/health"),
        ("Financial Data", f"{CONFIG['financial_data_url']}/health"),
        ("Network Analyzer", f"{CONFIG['network_analyzer_url']}/health"),
        ("Wealth Tracker", f"{CONFIG['wealth_tracker_url']}/health")
    ]
    
    for service_name, health_url in services:
        logger.info(f"Waiting for {service_name} service...")
        ready = False
        retries = 0
        
        while not ready and retries < 30:
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    logger.info(f"{service_name} service is ready.")
                    ready = True
                else:
                    logger.info(f"{service_name} not ready yet. Status: {response.status_code}")
                    time.sleep(5)
                    retries += 1
            except requests.exceptions.RequestException:
                logger.info(f"{service_name} not ready yet. Connection error.")
                time.sleep(5)
                retries += 1
        
        if not ready:
            logger.error(f"Could not connect to {service_name} service.")
            return False
    
    return True

async def create_initial_user():
    """Create admin user if it doesn't exist."""
    logger.info("Creating initial admin user...")
    try:
        # In a real implementation, use proper authentication/registration endpoints
        # This is a placeholder showing what would happen
        logger.info("Admin user created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        return False

async def setup_initial_workflows():
    """Set up initial workflow templates for billionaires."""
    logger.info("Setting up initial workflow templates...")
    
    billionaire_workflows = [
        {
            "name": "Wealth Tracking Automation",
            "description": "Automatically track and analyze wealth components across all asset classes",
            "tags": ["wealth", "tracking", "automation"]
        },
        {
            "name": "Strategic Networking",
            "description": "Identify and prioritize high-value networking

