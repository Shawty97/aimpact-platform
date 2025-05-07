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
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      email TEXT UNIQUE,
      password_hash TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  
  // Workflow templates
  db.run(`
    CREATE TABLE IF NOT EXISTS workflow_templates (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      description TEXT,
      config TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  
  // User feedback
  db.run(`
    CREATE TABLE IF NOT EXISTS user_feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      component TEXT,
      rating INTEGER,
      feedback_text TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `);
  
  // Sample users
  db.run(`INSERT OR IGNORE INTO users (username, email, password_hash) 
          VALUES ('admin', 'admin@example.com', 'hashed_password_placeholder')`);
  db.run(`INSERT OR IGNORE INTO users (username, email, password_hash) 
          VALUES ('testuser', 'user@example.com', 'hashed_password_placeholder')`);
          
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
  
  db.run(`INSERT OR IGNORE INTO workflow_templates (name, description, config) 
          VALUES ('Daily Summary', 'Sends a daily summary email', ?)`, 
          [JSON.stringify(workflow1)]);
          
  db.run(`INSERT OR IGNORE INTO workflow_templates (name, description, config) 
          VALUES ('Document Processing', 'Automatically processes new documents', ?)`, 
          [JSON.stringify(workflow2)]);
          
  // Sample feedback
  db.run(`INSERT OR IGNORE INTO user_feedback (user_id, component, rating, feedback_text) 
          VALUES (2, 'voice_ai', 4, 'The voice recognition is quite accurate, but sometimes struggles with technical terms.')`);
          
  db.run(`INSERT OR IGNORE INTO user_feedback (user_id, component, rating, feedback_text) 
          VALUES (2, 'workflow_engine', 5, 'The automation workflows save me a lot of time. Very satisfied!')`);
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
