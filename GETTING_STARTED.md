# Getting Started with AImpact Platform

This guide will help you quickly start using the AImpact platform and explore its core features. The platform provides advanced AI capabilities including voice processing, agent orchestration, and workflow automation.

## Quick Start Guide for Developers

### 1. Set Up Your Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/aimpact.git
cd aimpact

# Create and configure your environment file
cp .env.example .env
# Edit .env with your preferred settings

# Start the development environment with Docker
docker-compose up -d
```

### 2. Verify Everything is Running

```bash
# Check the status of all services
docker-compose ps

# View logs to ensure everything started correctly
docker-compose logs -f
```

### 3. Access Key Endpoints

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Dashboard (if available): http://localhost:3000
- Monitoring: http://localhost:9090 (Prometheus) and http://localhost:3001 (Grafana)

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB free space
- **Docker**: 20.10.x or newer
- **Docker Compose**: 1.29.x or newer
- **OS**: Linux (recommended), macOS, or Windows with WSL2

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (for local LLM and advanced voice processing)
- **Storage**: 50GB SSD
- **OS**: Ubuntu 20.04/22.04 or newer

### For Production

See the [DEPLOYMENT.md](DEPLOYMENT.md) for detailed production requirements.

## First Steps

### Voice AI Testing

AImpact provides advanced voice processing capabilities. Here's how to get started:

#### 1. Test Text-to-Speech (TTS)

```bash
# Using the API directly
curl -X POST "http://localhost:8000/api/voice/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from AImpact Platform!", "voice_id": "default"}'
```

This will return an audio file containing the synthesized speech.

#### 2. Test Speech-to-Text (STT)

```bash
# Upload an audio file for transcription
curl -X POST "http://localhost:8000/api/voice/stt" \
  -F "file=@/path/to/your/audio_file.mp3" \
  -F "language=en"
```

#### 3. Try Emotional Voice Synthesis

```bash
curl -X POST "http://localhost:8000/api/voice/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am really excited about this new feature!",
    "voice_id": "default",
    "emotion": "happy",
    "intensity": 0.8
  }'
```

#### 4. Test Voice Cloning (if available)

```bash
# First upload a sample of the voice you want to clone
curl -X POST "http://localhost:8000/api/voice/clone" \
  -F "file=@/path/to/voice_sample.mp3" \
  -F "name=my_custom_voice"

# Then use the cloned voice for TTS
curl -X POST "http://localhost:8000/api/voice/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is my cloned voice speaking.", "voice_id": "my_custom_voice"}'
```

### Agent Creation

Agents in AImpact are AI entities that can perform specific tasks. Here's how to create and interact with them:

#### 1. Create a Simple Agent

```bash
curl -X POST "http://localhost:8000/api/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GreeterAgent",
    "description": "An agent that greets users",
    "capabilities": ["greeting", "introduction"],
    "prompt_template": "You are a friendly assistant named {{name}}. Your goal is to greet the user and make them feel welcome.",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 150
    }
  }'
```

#### 2. Interact with Your Agent

```bash
# Get the agent ID from the creation response
AGENT_ID="your_agent_id_here"

curl -X POST "http://localhost:8000/api/agents/${AGENT_ID}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello there, who are you?",
    "session_id": "test_session_123"
  }'
```

#### 3. Create a Voice-Enabled Agent

```bash
curl -X POST "http://localhost:8000/api/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "VoiceAssistant",
    "description": "A voice-enabled assistant",
    "capabilities": ["voice_chat", "answering_questions"],
    "prompt_template": "You are a helpful voice assistant named {{name}}. Respond conversationally to the user's questions.",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 150
    },
    "voice_config": {
      "voice_id": "default",
      "enable_emotion": true,
      "speech_rate": 1.0
    }
  }'
```

### Workflow Development

Workflows allow you to create complex multi-step processes. Here's how to create and run workflows:

#### 1. Create a Basic Workflow

```bash
curl -X POST "http://localhost:8000/api/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support Workflow",
    "description": "A workflow for handling customer inquiries",
    "steps": [
      {
        "id": "step1",
        "name": "Initial Greeting",
        "type": "agent_interaction",
        "agent_id": "GreeterAgent_ID",
        "input_template": "Hello, how can I help you today?"
      },
      {
        "id": "step2",
        "name": "Issue Classification",
        "type": "llm_task",
        "prompt_template": "Classify the following customer issue: {{user_response}}",
        "depends_on": ["step1"]
      },
      {
        "id": "step3",
        "name": "Resolution",
        "type": "conditional",
        "conditions": [
          {
            "condition": "{{classification}} == 'technical'",
            "target": "step4"
          },
          {
            "condition": "{{classification}} == 'billing'",
            "target": "step5"
          }
        ],
        "depends_on": ["step2"]
      },
      {
        "id": "step4",
        "name": "Technical Support",
        "type": "agent_interaction",
        "agent_id": "TechSupportAgent_ID"
      },
      {
        "id": "step5",
        "name": "Billing Support",
        "type": "agent_interaction",
        "agent_id": "BillingAgent_ID"
      }
    ]
  }'
```

#### 2. Execute a Workflow

```bash
# Get the workflow ID from the creation response
WORKFLOW_ID="your_workflow_id_here"

curl -X POST "http://localhost:8000/api/workflows/${WORKFLOW_ID}/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_workflow_session",
    "input_data": {
      "customer_name": "John Doe",
      "issue_description": "I can't log in to my account"
    }
  }'
```

#### 3. Check Workflow Status

```bash
EXECUTION_ID="execution_id_from_previous_response"

curl -X GET "http://localhost:8000/api/workflows/executions/${EXECUTION_ID}" \
  -H "Content-Type: application/json"
```

## Examples of Common Use Cases

### 1. Interactive Voice Assistant

Combine the Voice AI and Agent capabilities to create an interactive voice assistant:

```bash
# Create a voice agent
AGENT_ID=$(curl -s -X POST "http://localhost:8000/api/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "CustomerSupport",
    "description": "A voice-enabled customer support agent",
    "capabilities": ["voice_chat", "troubleshooting"],
    "prompt_template": "You are a helpful customer support agent. Help the user solve their problems.",
    "voice_config": {
      "voice_id": "default",
      "enable_emotion": true
    }
  }' | jq -r '.id')

# Start a voice conversation
SESSION_ID=$(curl -s -X POST "http://localhost:8000/api/voice/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "'$AGENT_ID'",
    "mode": "two_way"
  }' | jq -r '.session_id')

# Now you can use WebSocket to stream audio:
# ws://localhost:8000/api/voice/stream/${SESSION_ID}
```

### 2. Document Processing Pipeline

Create a workflow for processing and analyzing documents:

```bash
curl -X POST "http://localhost:8000/api/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Document Analysis Pipeline",
    "description": "Process, extract information, and summarize documents",
    "steps": [
      {
        "id": "step1",
        "name": "Document Upload",
        "type": "input",
        "input_key": "document_url"
      },
      {
        "id": "step2",
        "name": "Text Extraction",
        "type": "document_processor",
        "processor": "text_extractor",
        "depends_on": ["step1"]
      },
      {
        "id": "step3",
        "name": "Content Analysis",
        "type": "llm_task",
        "prompt_template": "Analyze the following document and extract key information: {{extracted_text}}",
        "depends_on": ["step2"]
      },
      {
        "id": "step4",
        "name": "Summary Generation",
        "type": "llm_task",
        "prompt_template": "Create a concise summary of this document: {{extracted_text}}",
        "depends_on": ["step2"]
      }
    ]
  }'
```

### 3. Multi-Modal Customer Support

Create a workflow that handles both text and voice interactions:

```bash
curl -X POST "http://localhost:8000/api/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Multi-Modal Support",
    "description": "Handle customer support via text and voice",
    "steps": [
      {
        "id": "step1",
        "name": "Input Detection",
        "type": "conditional",
        "conditions": [
          {
            "condition": "{{input_type}} == 'voice'",
            "target": "step2"
          },
          {
            "condition": "{{input_type}} == 'text'",
            "target": "step3"
          }
        ]
      },
      {
        "id": "step2",
        "name": "Voice Processing",
        "type": "voice_task",
        "task": "speech_to_text",
        "input_key": "voice_input"
      },
      {
        "id": "step3",
        "name": "Support Agent",
        "type": "agent_interaction",
        "agent_id": "SupportAgent_ID",
        "input_template": "{{text_input or transcribed_text}}"
      },
      {
        "id": "step4",
        "name": "Response Format",
        "type": "conditional",
        "conditions": [
          {
            "condition": "{{input_type}} == 'voice'",
            "target": "step5"
          },
          {
            "condition": "{{input_type}} == 'text'",
            "target": "step6"
          }
        ],
        "depends_on": ["step3"]
      },
      {
        "id": "step5",
        "name": "Text to Speech",
        "type": "voice_task",
        "task": "text_to_speech",
        "input_template": "{{agent_response}}"
      },
      {
        "id": "step6",
        "name": "Text Response",
        "type": "output",
        "output_template": "{{agent_response}}"
      }
    ]
  }'
```

## Troubleshooting Tips

### Common Issues and Solutions

#### 1. Docker Container Not Starting

```bash
# Check for error messages
docker-compose logs <service_name>

# Common solutions:
# - Ensure ports are not already in use
# - Check environment variables in .env file
# - Verify you have sufficient disk space and memory
```

#### 2. Voice Processing Issues

```bash
# Check if required models are downloaded
docker-compose exec backend ls -la /app/models/tts/coqui
docker-compose exec backend ls -la /app/models/stt/vosk

# If models are missing, you may need to download them:
docker-compose exec backend python -m scripts.download_models
```

#### 3. LLM Connection Problems

```bash
# For local LLM issues:
docker-compose logs ollama

# Check if Ollama has the required models:
curl http://localhost:11434/api/tags

# For OpenAI API issues:
# - Verify your API key in the .env file
# - Check your OpenAI account for quota/rate limit issues
```

#### 4. Database Connection Issues

```bash
# Check database logs
docker-compose logs db

# Connect to the database directly
docker-compose exec db psql -U postgres -d aimpact

# If necessary, reset the database:
docker-compose down -v
docker-compose up -d
```

### Getting Support

If you encounter issues not covered in this guide:

- Check the [GitHub Issues](https://github.com/yourusername/aimpact/issues) for known problems and solutions
- Join our [Discord community](https://discord.gg/aimpact) for real-time support
- Contact support at support@aimpact.example.com

## Next Steps

Once you're comfortable with the basics, consider:

1. **Customizing agents** with specialized knowledge and capabilities
2. **Creating advanced workflows** for your specific use cases
3. **Integrating with your existing systems** using the API
4. **Contributing** to the platform by submitting PRs or reporting issues

Refer to the full [documentation](https://docs.aimpact.example.com) for more detailed information on each component and API reference.

