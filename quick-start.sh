#!/bin/bash
# AImpact Platform Quick Start Script
# This script sets up a development environment for testing the AImpact platform

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# Print header
echo -e "${BOLD}${BLUE}"
echo "  _____    _____                                  _   "
echo " |  _  |  |_   _|                                | |  "
echo " | | | |    | | _ __ ___  _ __   __ _  ___  ___ | |_ "
echo " | | | |    | || '_ \` _ \| '_ \ / _\` |/ __/ __|| __|"
echo " \ \_/ /   _| || | | | | | |_) | (_| | (_| (__ | |_ "
echo "  \___/    \___/_| |_| |_| .__/ \__,_|\___\___| \__|"
echo "                         | |                         "
echo "                         |_|                         "
echo -e "${NC}"
echo -e "${BOLD}Platform Development Environment Setup${NC}"
echo "---------------------------------------"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to show a progress spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Step 1: Check dependencies
echo -e "${BOLD}Step 1: Checking dependencies...${NC}"
MISSING_DEPS=0

# Check for Node.js
if command_exists node; then
    NODE_VERSION=$(node -v | cut -d 'v' -f 2)
    REQUIRED_NODE_VERSION="18.0.0"
    
    if [ "$(printf '%s\n' "$REQUIRED_NODE_VERSION" "$NODE_VERSION" | sort -V | head -n1)" = "$REQUIRED_NODE_VERSION" ]; then 
        echo -e "  [${GREEN}✓${NC}] Node.js ${NODE_VERSION} (Required: ${REQUIRED_NODE_VERSION}+)"
    else
        echo -e "  [${YELLOW}!${NC}] Node.js ${NODE_VERSION} detected but version ${REQUIRED_NODE_VERSION}+ is required"
        MISSING_DEPS=1
    fi
else
    echo -e "  [${RED}✗${NC}] Node.js not found (Required: ${REQUIRED_NODE_VERSION}+)"
    MISSING_DEPS=1
fi

# Check for npm
if command_exists npm; then
    NPM_VERSION=$(npm -v)
    echo -e "  [${GREEN}✓${NC}] npm ${NPM_VERSION}"
else
    echo -e "  [${RED}✗${NC}] npm not found"
    MISSING_DEPS=1
fi

# Check for Docker
if command_exists docker; then
    DOCKER_VERSION=$(docker --version | cut -d ' ' -f 3 | cut -d ',' -f 1)
    echo -e "  [${GREEN}✓${NC}] Docker ${DOCKER_VERSION}"
else
    echo -e "  [${YELLOW}!${NC}] Docker not found (recommended for full testing)"
fi

# Check for Docker Compose
if command_exists docker-compose; then
    DOCKER_COMPOSE_VERSION=$(docker-compose --version | cut -d ' ' -f 3 | cut -d ',' -f 1)
    echo -e "  [${GREEN}✓${NC}] Docker Compose ${DOCKER_COMPOSE_VERSION}"
else
    echo -e "  [${YELLOW}!${NC}] Docker Compose not found (recommended for full testing)"
fi

# Check for Git
if command_exists git; then
    GIT_VERSION=$(git --version | cut -d ' ' -f 3)
    echo -e "  [${GREEN}✓${NC}] Git ${GIT_VERSION}"
else
    echo -e "  [${RED}✗${NC}] Git not found"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Warning: Some required dependencies are missing.${NC}"
    echo "Please install them before continuing."
    
    while true; do
        read -p "Do you want to continue anyway? (y/n) " yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) echo "Setup aborted."; exit 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi

echo ""

# Step 2: Create directory structure if not exists
echo -e "${BOLD}Step 2: Setting up project structure...${NC}"

# Create necessary directories
mkdir -p config
mkdir -p data/db
mkdir -p logs
mkdir -p src/{core,api,services,utils,models}
mkdir -p test/{unit,integration,e2e}
mkdir -p public

echo -e "  [${GREEN}✓${NC}] Directory structure created"
echo ""

# Step 3: Create configuration files
echo -e "${BOLD}Step 3: Creating configuration files...${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOL
# AImpact Platform Environment Configuration
# Development Environment

# Core Configuration
NODE_ENV=development
PORT=3000
LOG_LEVEL=debug

# Database Configuration
DB_TYPE=sqlite
DB_PATH=./data/db/aimpact.db

# Features
ENABLE_WORKFLOW_ENGINE=true
ENABLE_VOICE_AI=true
ENABLE_CROSS_MODAL=true
ENABLE_ADAPTIVE_RESPONSE=true

# Sample Data
LOAD_SAMPLE_DATA=true
EOL
    echo -e "  [${GREEN}✓${NC}] Created .env file with development settings"
else
    echo -e "  [${YELLOW}!${NC}] .env file already exists, skipping"
fi

# Create package.json if it doesn't exist
if [ ! -f package.json ]; then
    cat > package.json << EOL
{
  "name": "aimpact-platform",
  "version": "0.1.0",
  "description": "AI-powered platform with adaptive response optimization",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "test": "jest test/unit",
    "test:integration": "jest test/integration",
    "test:e2e": "jest test/e2e",
    "seed:test": "node scripts/seed-test-data.js",
    "lint": "eslint src"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.0.3",
    "winston": "^3.8.2",
    "sqlite3": "^5.1.4",
    "jose": "^4.11.2",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "nodemon": "^2.0.20",
    "jest": "^29.3.1",
    "eslint": "^8.31.0"
  }
}
EOL
    echo -e "  [${GREEN}✓${NC}] Created package.json with required dependencies"
else
    echo -e "  [${YELLOW}!${NC}] package.json already exists, skipping"
fi

# Create basic structure for the core components if they don't exist
# Create index.js
if [ ! -f src/index.js ]; then
    cat > src/index.js << EOL
// AImpact Platform Entry Point
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const winston = require('winston');

// Create logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Initialize app
const app = express();
app.use(cors());
app.use(express.json());

// Basic health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// API route for demo purposes
app.get('/api/info', (req, res) => {
  res.json({
    name: 'AImpact Platform',
    version: '0.1.0',
    features: {
      workflowEngine: process.env.ENABLE_WORKFLOW_ENGINE === 'true',
      voiceAI: process.env.ENABLE_VOICE_AI === 'true',
      crossModal: process.env.ENABLE_CROSS_MODAL === 'true',
      adaptiveResponse: process.env.ENABLE_ADAPTIVE_RESPONSE === 'true'
    }
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  logger.info(\`AImpact Platform started on port \${PORT}\`);
  logger.info(\`Environment: \${process.env.NODE_ENV}\`);
  
  if (process.env.LOAD_SAMPLE_DATA === 'true') {
    logger.info('Sample data loading enabled. Run npm run seed:test to load data.');
  }
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully.');
  process.exit(0);
});
EOL
    echo -e "  [${GREEN}✓${NC}] Created basic application entry point"
else
    echo -e "  [${YELLOW}!${NC}] src/index.js already exists, skipping"
fi

# Create sample data script
mkdir -p scripts
if [ ! -f scripts/seed-test-data.js ]; then
    cat > scripts/seed-test-data.js << EOL
// Sample data seeding script
require('dotenv').config();
const path = require('path');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

console.log('Seeding test data...');

// Ensure DB directory exists
const dbDir = path.dirname(process.env.DB_PATH || './data/db/aimpact.db');
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

// Connect to database
const db = new sqlite3.Database(process.env.DB_PATH || './data/db/aimpact.db');

// Create tables
db.serialize(() => {
  // Users table (for future auth system)
  db.run(\`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      email TEXT UNIQUE,
      password_hash TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  \`);
  
  // Workflow templates
  db.run(\`
    CREATE TABLE IF NOT EXISTS workflow_templates (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      description TEXT,
      config TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  \`);
  
  // User feedback
  db.run(\`
    CREATE TABLE IF NOT EXISTS user_feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      component TEXT,
      rating INTEGER,
      feedback_text TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  \`);
  
  // Sample users
  db.run(\`INSERT OR IGNORE INTO users (username, email, password_hash) 
          VALUES ('admin', 'admin@example.com', 'hashed_password_placeholder')\`);
  db.run(\`INSERT OR IGNORE INTO users (username, email, password_hash) 
          VALUES ('testuser', 'user@example.com', 'hashed_password_placeholder')\`);
          
  // Sample workflow templates
  const workflow1 = {
    triggers: [{ type: 'schedule', config: { cron: '0 9 * * *' } }],
    actions: [
      { type: 'notification', config: { channel: 'email', template: 'daily_summary' } }
    ],
    conditions: [{ type: 'user_preference', field: 'notifications.email', value: true }]
  };
  
  const workflow2 = {
    triggers: [{ type: 'event', config: { name: 'document_created' } }],
    actions: [
      { type: 'ai_processing', config: { processor: 'document_summarizer' } },
      { type: 'notification', config: { channel: 'in_app', template: 'document_processed' } }
    ],
    conditions: []
  };
  
  db.run(\`INSERT OR IGNORE INTO workflow_templates (name, description, config) 
          VALUES ('Daily Summary', 'Sends a daily summary email', ?)\`, 
          [JSON.stringify(workflow1)]);
          
  db.run(\`INSERT OR IGNORE INTO workflow_templates (name, description, config) 
          VALUES ('Document Processing', 'Automatically processes new documents', ?)\`, 
          [JSON.stringify(workflow2)]);
          
  // Sample feedback
  db.run(\`INSERT OR IGNORE INTO user_feedback (user_id, component, rating, feedback_text) 
          VALUES (2, 'voice_ai', 4, 'The voice recognition is quite accurate, but sometimes struggles with technical terms.')\`);
          
  db.run(\`INSERT OR IGNORE INTO user_feedback (user_id, component, rating, feedback_text) 
          VALUES (2, 'workflow_engine', 5, 'The automation workflows save me a lot of time. Very satisfied!')\`);
});

// Close the database connection
db.close((err) => {
  if (err) {
    console.error('Error closing database:', err.message);
  } else {
    console.log('Test data seeded successfully!');
    console.log('You can now start the application with: npm run dev');
  }
});
EOL
    echo -e "  [${GREEN}✓${NC}] Created sample data seeding script"
else
    echo -e "  [${YELLOW}!${NC}] scripts/seed-test-data.js already exists, skipping"
fi

echo ""

# Step 4: Install dependencies
echo -e "${BOLD}Step 4: Installing dependencies...${NC}"
echo "This may take a few minutes..."
npm install >/dev/null 2>&1 &

