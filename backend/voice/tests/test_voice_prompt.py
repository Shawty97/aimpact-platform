"""
AImpact Voice Prompt API Test Suite

This module provides comprehensive tests for the Voice Prompt API, including:
- Voice prompt processing with different input methods (file uploads, base64 audio)
- Session management API tests
- Error handling and edge cases
- Schema extraction validation
"""

import os
import sys
import pytest
import json
import uuid
import base64
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
import numpy as np

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import API and models
from backend.voice.api.voice_prompt import (
    router, VoicePromptRequest, VoicePromptResponse, AudioSource, 
    AgentSchema, AgentProperty, AgentAction, get_redis_client
)
from backend.app.main import app

# Create test client
client = TestClient(app)

# Test data
SAMPLE_AUDIO_BASE64 = "UklGRiQEAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAEAABt7U3pTOZL40rhSeBI30fdRdtE2kPZQthB10DWP9U+1D3TO9I70jrROdE40TfRNtE20TbRNtE30TjROtE70TzTQdVD1kXYR9pJ3ErdTN9O4U/iUOJR4lDhT99N3UvbSNlF10TWQ9RC00HSQdFA0UDRQNFBAAAAvAE2AAAAAAAuAAAAAAAAAAQAAP//z//z/9f/3P/X/+P/3P/h/+H/3v/f/9z/4f/c/+H/4f/j/+P/4f/i/+X/5P/j/+j/6v/t//b/+f8FAAcADAAQABUAFgAcACMAKAAwAD4ARgBWAF8AagByAHsAgQCJAI0AkwCJAIcAfAB6AHEAZABcAFEASQA9ADQAKgAfABgADAADAPz/9P/o/9//1f/O/8f/v/+6/7P/rv+p/6b/o/+g/57/nf+b/5r/mf+Y/5f/l/+X/5b/lv+W/5b/lf+V/5X/lP+T/5L/kf+Q/4//jv+N/47/jf+M/4z/jf+O/4//kf+U/5f/m/+e/6P/qP+s/7H/tv+8/8H/xv/M/9L/1//c/+L/5//s//H/9v/7/wAABQAJAA4AEgAXABsAIAAlACkALQAyADUAOQA9AEAAQwBGAEkATABOAFEAUgBVAFYAWABaAFwAXgBfAGAAYQBjAGQAZQBmAGYAaABoAGkAagBrAGwAbQBuAG8AcABxAHIAcwBzAHQAdAB1AHUAdgB2AHcAdwB3AHcAdwB3AHcAdgB2AHUAdAB0AHMAcgBwAG8AbgBsAGoAaABmAGQAYgBgAF0AWwBYAFUAUgBQAE0ASgBHAEQAQQA+ADsAOAA1ADMAMAAuACsAKQAnACUAIwAhAB8AHgAcABsAGQAYABcAFgAVABQAFAAUABQAFAAUABUAFQAWABcAGAAZABsAHAAdAB8AIAAiACQAJgAnACkAKwAtAC8AMQAzADUANwA5ADoAPAA+AEAARQBKAFYAZwB4AIkAmwCtAMAAyADPANUA2wDgAOUA6QDtAPEA9AD4APsA/gABAQQBBgEJAQsBDgEQARIBFAEWARgBGgEcAR4BIAEiASQBJgEoASoBLAEuATABMgEzATUBNwE5ATsBPQE/AUEBQwFFAUcBSQFLAU0BTwFRAVMBVQFXAVkBWwFdAV8BYQFjAWUBZwFqAWwBbgFwAXIBdAF3AXkBewF9AYABggGEAYYBiQGLAY0BkAGSAZQBlwGZAZwBngGgAaMBpQGoAaoBrQGvAbIBtAG3AboB" 

# Helper functions
def generate_test_audio():
    """Generate a simple test audio file and return its path."""
    # Generate a simple sine wave
    duration = 2.0  # seconds
    sample_rate = 16000  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz tone
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create a temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    # Write WAV file using NumPy directly
    from scipy.io import wavfile
    wavfile.write(temp_file.name, sample_rate, audio_data)
    
    return temp_file.name

# Mock redis client for testing
class MockRedis:
    def __init__(self):
        self.data = {}
    
    def setex(self, key, ttl, value):
        self.data[key] = {"value": value, "ttl": ttl}
        return True
    
    def get(self, key):
        if key in self.data:
            return self.data[key]["value"]
        return None
    
    def delete(self, key):
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    def ttl(self, key):
        if key in self.data:
            return self.data[key]["ttl"]
        return -2  # Key doesn't exist


# Test classes

class TestVoicePromptAPI:
    """Test the Voice Prompt API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Create a test audio file
        self.test_audio_path = generate_test_audio()
        
        # Mock the speech engine
        self.speech_engine_patcher = patch('backend.voice.api.voice_prompt.SpeechEngine')
        self.mock_speech_engine = self.speech_engine_patcher.start()
        self.mock_speech_engine.return_value.transcribe_file.return_value = "This is a test transcript."
        
        # Mock the LLM for schema extraction
        self.llm_patcher = patch('backend.voice.api.voice_prompt.extract_agent_schema')
        self.mock_extract_schema = self.llm_patcher.start()
        self.mock_extract_schema.return_value = AgentSchema(
            name="TestAgent",
            description="A test agent schema",
            prompt_transcript="This is a test transcript.",
            properties=[
                AgentProperty(name="test_prop", type="string", description="A test property", required=True)
            ],
            actions=[
                AgentAction(name="test_action", description="A test action")
            ]
        )
        
        # Mock Redis
        self.redis_patcher = patch('backend.voice.api.voice_prompt.get_redis_client')
        self.mock_redis_func = self.redis_patcher.start()
        self.mock_redis = MockRedis()
        self.mock_redis_func.return_value = self.mock_redis
        
        yield
        
        # Clean up
        os.unlink(self.test_audio_path)
        self.speech_engine_patcher.stop()
        self.llm_patcher.stop()
        self.redis_patcher.stop()
    
    def test_process_voice_prompt_file_upload(self):
        """Test processing a voice prompt with file upload."""
        with open(self.test_audio_path, "rb") as f:
            response = client.post(
                "/voice-prompt",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={
                    "model_size": "base",
                    "extract_agent_schema": "true",
                    "store_in_redis": "true"
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["transcript"] == "This is a test transcript."
        assert data["agent_schema"]["name"] == "TestAgent"
        assert data["redis_key"] is not None
    
    def test_process_voice_prompt_base64(self):
        """Test processing a voice prompt with base64 encoded audio."""
        response = client.post(
            "/voice-prompt",
            json={
                "audio_source": "base64",
                "base64_audio": SAMPLE_AUDIO_BASE64,
                "model_size": "base",
                "extract_agent_schema": True,
                "store_in_redis": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["transcript"] == "This is a test transcript."
        assert data["agent_schema"]["name"] == "TestAgent"
        assert data["redis_key"] is not None
    
    def test_get_voice_prompt_result(self):
        """Test retrieving a previously processed voice prompt result."""
        # First, process a prompt to store in Redis
        with open(self.test_audio_path, "rb") as f:
            response = client.post(
                "/voice-prompt",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={
                    "model_size": "base",
                    "extract_agent_schema": "true",
                    "store_in_redis": "true"
                }
            )
        
        data = response.json()
        request_id = data["request_id"]
        
        # Now try to retrieve it
        response = client.get(f"/voice-prompt/{request_id}")
        
        assert response.status_code == status.HTTP_200_OK
        retrieved_data = response.json()
        assert retrieved_data["request_id"] == request_id
        assert retrieved_data["transcript"] == "This is a test transcript."
    
    def test_get_nonexistent_voice_prompt(self):
        """Test retrieving a non-existent voice prompt result."""
        response = client.get(f"/voice-prompt/{uuid.uuid4()}")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSessionManagementAPI:
    """Test the Voice Session Management API."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Create a test audio file
        self.test_audio_path = generate_test_audio()
        
        # Mock the speech engine
        self.speech_engine_patcher = patch('backend.voice.api.voice_prompt.SpeechEngine')
        self.mock_speech_engine = self.speech_engine_patcher.start()
        self.mock_speech_engine.return_value.transcribe_file.return_value = "This is a test transcript."
        
        # Mock Redis
        self.redis_patcher = patch('backend.voice.api.voice_prompt.get_redis_client')
        self.mock_redis_func = self.redis_patcher.start()
        self.mock_redis = MockRedis()
        self.mock_redis_func.return_value = self.mock_redis
        
        yield
        
        # Clean up
        os.unlink(self.test_audio_path)
        self.speech_engine_patcher.stop()
        self.redis_patcher.stop()
    
    def test_create_session(self):
        """Test creating a new voice session."""
        response = client.post(
            "/voice/sessions",
            json={
                "language": "en-US",
                "stt_engine": "whisper",
                "tts_engine": "openai",
                "voice_profile": "default"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "session_id" in data
        assert data["language"] == "en-US"
        assert data["stt_engine"] == "whisper"
        assert data["tts_engine"] == "openai"
        assert data["voice_profile"] == "default"
        assert data["created_at"] is not None
    
    def test_get_session(self):
        """Test retrieving a session by ID."""
        # First create a session
        create_response = client.post(
            "/voice/sessions",
            json={
                "language": "en-US",
                "stt_engine": "whisper"
            }
        )
        
        session_id = create_response.json()["session_id"]
        
        # Now retrieve it
        response = client.get(f"/voice/sessions/{session_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == session_id
        assert data["language"] == "en-US"
        assert data["stt_engine"] == "whisper"
        assert "created_at" in data
        assert "last_activity" in data
    
    def test_get_all_sessions(self):
        """Test retrieving all active sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            create_response = client.post(
                "/voice/sessions",
                json={
                    "language": f"en-US-{i}",
                    "stt_engine": "whisper",
                    "tts_engine": "openai"
                }
            )
            session_ids.append(create_response.json()["session_id"])
        
        # Retrieve all sessions
        response = client.get("/voice/sessions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3  # Could be more if other tests created sessions
        
        # Verify that our created sessions are in the list
        found_sessions = [session for session in data if session["session_id"] in session_ids]
        assert len(found_sessions) == 3
        
        # Verify session data is correct
        for i, session in enumerate(found_sessions):
            assert "language" in session
            assert "created_at" in session
            assert "last_activity" in session
    
    def test_update_session(self):
        """Test updating session settings."""
        # First create a session
        create_response = client.post(
            "/voice/sessions",
            json={
                "language": "en-US",
                "stt_engine": "whisper",
                "tts_engine": "openai",
                "voice_profile": "default"
            }
        )
        
        session_id = create_response.json()["session_id"]
        
        # Now update it
        update_response = client.put(
            f"/voice/sessions/{session_id}",
            json={
                "language": "es-ES",
                "tts_engine": "elevenlabs",
                "voice_profile": "custom"
            }
        )
        
        assert update_response.status_code == status.HTTP_200_OK
        update_data = update_response.json()
        assert update_data["session_id"] == session_id
        assert update_data["language"] == "es-ES"
        assert update_data["stt_engine"] == "whisper"  # Unchanged
        assert update_data["tts_engine"] == "elevenlabs"
        assert update_data["voice_profile"] == "custom"
        
        # Verify changes persisted with a get request
        get_response = client.get(f"/voice/sessions/{session_id}")
        get_data = get_response.json()
        assert get_data["language"] == "es-ES"
        assert get_data["tts_engine"] == "elevenlabs"
        assert get_data["voice_profile"] == "custom"
    
    def test_delete_session(self):
        """Test deleting a session."""
        # First create a session
        create_response = client.post(
            "/voice/sessions",
            json={
                "language": "en-US",
                "stt_engine": "whisper"
            }
        )
        
        session_id = create_response.json()["session_id"]
        
        # Verify the session exists
        get_response = client.get(f"/voice/sessions/{session_id}")
        assert get_response.status_code == status.HTTP_200_OK
        
        # Now delete it
        delete_response = client.delete(f"/voice/sessions/{session_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify it's gone
        get_response_after = client.get(f"/voice/sessions/{session_id}")
        assert get_response_after.status_code == status.HTTP_404_NOT_FOUND
    
    def test_session_expiration(self):
        """Test session expiration handling based on TTL."""
        # Override the MockRedis get method temporarily to simulate expiration
        original_get = self.mock_redis.get
        
        # First create a session
        create_response = client.post(
            "/voice/sessions",
            json={"language": "en-US", "stt_engine": "whisper"}
        )
        
        session_id = create_response.json()["session_id"]
        
        # Verify the session exists
        get_response = client.get(f"/voice/sessions/{session_id}")
        assert get_response.status_code == status.HTTP_200_OK
        
        # Replace the get method to simulate expiration
        def mock_expired_get(key):
            if key.endswith(session_id):
                return None  # Simulate expired/deleted key
            return original_get(key)
        
        self.mock_redis.get = mock_expired_get
        
        # Verify session is now considered expired
        get_response_expired = client.get(f"/voice/sessions/{session_id}")
        assert get_response_expired.status_code == status.HTTP_404_NOT_FOUND
        
        # Restore the original get method
        self.mock_redis.get = original_get
    
    def test_invalid_session_operations(self):
        """Test proper error handling for invalid session operations."""
        # Test get with invalid session ID
        invalid_uuid = str(uuid.uuid4())
        get_response = client.get(f"/voice/sessions/{invalid_uuid}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test update with invalid session ID
        update_response = client.put(
            f"/voice/sessions/{invalid_uuid}",
            json={"language": "fr-FR"}
        )
        assert update_response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test delete with invalid session ID
        delete_response = client.delete(f"/voice/sessions/{invalid_uuid}")
        assert delete_response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test create with invalid data
        invalid_create_response = client.post(
            "/voice/sessions",
            json={
                "language": "invalid-language-code-that-is-too-long",
                "stt_engine": None,
                "tts_engine": 12345  # Should be string
            }
        )
        assert invalid_create_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

