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
  logger.info(`AImpact Platform started on port ${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV}`);
  
  if (process.env.LOAD_SAMPLE_DATA === 'true') {
    logger.info('Sample data loading enabled. Run npm run seed:test to load data.');
  }
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully.');
  process.exit(0);
});
