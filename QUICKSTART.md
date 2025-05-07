# AImpact Platform - Quick Start Guide

This guide provides simple steps to set up and test the AImpact platform, focusing on the voice processing capabilities. This guide is designed to be accessible for non-developers.

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key (for speech recognition and language model features)
- Basic familiarity with command line/terminal

## 1. Set Up Development Environment

### 1.1 Clone the Repository (if you haven't already)

```bash
git clone https://github.com/yourusername/aimpact.git
cd aimpact
```

### 1.2 Create and Activate a Virtual Environment

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your OpenAI API key and other configuration to this file:

```
OPENAI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## 2. Run the Server

Start the AImpact server:

```bash
python run.py
```

You should see output indicating that the server is running, typically on http://localhost:8000.

## 3. Test Voice Processing Capabilities

You can test the voice processing capabilities in several ways:

### 3.1 Using the Test Script

The simplest way to test the voice capabilities is to run the test script:

```bash
python tests/test_voice_agent.py
```

This will run through a series of tests showcasing:
- Voice agent initialization
- Audio processing with emotion detection
- Real-time streaming capabilities
- Voice cloning and adaptation
- Voice-based workflow creation

### 3.2 Using the API (via Browser)

1. Open your web browser and navigate to: http://localhost:8000/docs
2. This will open the Swagger documentation where you can test the API directly
3. Try the following endpoints:
   - `/api/voice/transcribe` - Upload an audio file and convert it to text
   - `/api/agents/` - List available agents
   - `/api/voice/synthesize` - Convert text to speech

### 3.3 Creating a Sample Voice Recording

1. Record a short audio clip (WAV format is preferable) using your computer's microphone
2. Use the `/api/voice/transcribe` endpoint to transcribe it
3. Then try the emotion detection by using the `/api/voice/detect-emotion` endpoint with the same audio

## 4. Explore Voice Agent Features

### 4.1 Create a Voice Agent

Create a new voice agent with specific capabilities:

1. In the Swagger UI, find the `/api/agents/` POST endpoint
2. Create a new agent with this JSON:
   ```json
   {
     "name": "My Test Agent",
     "description": "A test agent with voice capabilities",
     "llm_provider": "openai",
     "model_name": "gpt-4",
     "capabilities": ["text_generation", "voice_interaction"],
     "system_prompt": "You are a helpful assistant."
   }
   ```
3. Note the `agent_id` in the response for the next step

### 4.2 Interact with the Agent

Send a voice message to your agent:

1. In the Swagger UI, find the `/api/agents/{agent_id}/interact` endpoint
2. Upload your audio file
3. Specify the agent ID from the previous step
4. Submit and observe the agent's response

### 4.3 Try Voice-Based Workflow Creation

Use voice commands to create a workflow:

1. Record an audio message saying "create workflow called customer support automation"
2. Use the `/api/agents/{agent_id}/interact` endpoint again
3. The agent will create a workflow and provide instructions for next steps

## 5. Running the Test Suite

### 5.1 Run All Tests

To run the complete test suite:

```bash
pytest
```

### 5.2 Run Specific Tests

To run only the voice agent tests:

```bash
pytest tests/test_voice_agent.py
```

To run with verbose output:

```bash
pytest -v tests/test_voice_agent.py
```

## 6. WebSocket Streaming Demo

To test real-time voice streaming:

1. Open a WebSocket client (like [Postman](https://www.postman.com/) or [Simple WebSocket Client](https://chrome.google.com/webstore/detail/simple-websocket-client/pfdhoblngboilpfeibdedpjgfnlcodoo) browser extension)
2. Connect to: `ws://localhost:8000/api/voice/ws/{session_id}` (replace `{session_id}` with a new UUID or leave empty for auto-generation)
3. Send audio chunks as binary messages
4. You'll receive real-time transcriptions and AI responses

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your OpenAI API key is correctly set in the .env file
   - Check that you have sufficient credits in your OpenAI account

2. **Dependency Issues**
   - If you encounter dependency errors, try updating your packages:
     ```bash
     pip install --upgrade -r requirements.txt
     ```

3. **Audio Format Issues**
   - The system works best with WAV files at 16kHz, mono channel
   - Convert other formats using tools like [Audacity](https://www.audacityteam.org/)

4. **"Module not found" Errors**
   - Ensure you're running commands from the root directory of the project
   - Check that your virtual environment is activated

### Getting Help

If you encounter issues not addressed here, check the fuller documentation in the `docs/` directory or create an issue in the repository.

## Next Steps

After exploring the basic functionality, you might want to:

1. Create custom workflows using the voice commands
2. Experiment with different emotion states in your voice recordings
3. Create a custom voice model using the voice cloning feature
4. Try multi-turn conversations with the agents

For more detailed instructions and advanced features, refer to the full documentation.

