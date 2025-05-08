"""
AImpact Voice Prompt API Endpoint

This module provides a FastAPI endpoint for processing voice prompts that can be
used to extract agent schemas and create workflows through voice instructions.

The API supports:
- WAV file uploads
- Base64 encoded audio input
- Whisper-based transcription
- LLM processing for schema extraction
- Redis integration for result storage
- JSON response with the extracted agent schema
"""

import os
import time
import base64
import asyncio
import logging
import tempfile
import uuid
import json
from typing import Dict, List, Optional, Union, Any, BinaryIO
from enum import Enum
from datetime import datetime
from io import BytesIO

import redis
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Body, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import local modules
from backend.voice.core.speech_engine import SpeechEngine, SpeechProvider, RecognitionMode, AudioFormat
from backend.app.core.config import settings

# Configure logging
logger = logging.getLogger("aimpact.voice.api.voice_prompt")

# ----------------- Models -----------------

class AudioSource(str, Enum):
    """Source of audio data."""
    FILE = "file"
    BASE64 = "base64"


class VoicePromptRequest(BaseModel):
    """Request model for voice prompt processing."""
    audio_source: AudioSource = Field(AudioSource.FILE, description="Source of audio data")
    base64_audio: Optional[str] = Field(None, description="Base64 encoded audio data (when source is base64)")
    model_size: str = Field("large", description="Whisper model size to use (tiny, base, small, medium, large)")
    extract_agent_schema: bool = Field(True, description="Whether to extract agent schema from the prompt")
    store_in_redis: bool = Field(True, description="Whether to store the result in Redis")
    ttl: int = Field(3600, description="Time-to-live for Redis storage in seconds")
    
    @validator('base64_audio')
    def validate_base64(cls, v, values):
        if values.get('audio_source') == AudioSource.BASE64 and not v:
            raise ValueError('Base64 audio is required when audio_source is base64')
        return v


class AgentProperty(BaseModel):
    """Property of an agent schema."""
    name: str = Field(..., description="Name of the property")
    type: str = Field(..., description="Type of the property (string, number, boolean, etc.)")
    description: Optional[str] = Field(None, description="Description of the property")
    required: bool = Field(False, description="Whether the property is required")
    default: Optional[Any] = Field(None, description="Default value for the property")


class AgentAction(BaseModel):
    """Action that an agent can perform."""
    name: str = Field(..., description="Name of the action")
    description: Optional[str] = Field(None, description="Description of the action")
    parameters: List[AgentProperty] = Field(default_factory=list, description="Parameters for the action")
    returns: Optional[str] = Field(None, description="Return type of the action")


class AgentSchema(BaseModel):
    """Schema extracted from a voice prompt."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")
    properties: List[AgentProperty] = Field(default_factory=list, description="Properties of the agent")
    actions: List[AgentAction] = Field(default_factory=list, description="Actions the agent can perform")
    created_at: datetime = Field(default_factory=datetime.now)
    prompt_transcript: str = Field(..., description="Transcript of the original voice prompt")


class VoicePromptResponse(BaseModel):
    """Response model for voice prompt processing."""
    request_id: str = Field(..., description="Unique ID for this request")
    transcript: str = Field(..., description="Transcribed text from the voice prompt")
    processing_time: float = Field(..., description="Total processing time in seconds")
    transcription_time: float = Field(..., description="Time taken for transcription in seconds")
    schema_extraction_time: Optional[float] = Field(None, description="Time taken for schema extraction in seconds")
    agent_schema: Optional[AgentSchema] = Field(None, description="Extracted agent schema if requested")
    redis_key: Optional[str] = Field(None, description="Redis key where the result is stored")


# ----------------- Redis Client -----------------

def get_redis_client():
    """Get a Redis client instance."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=False,
        encoding="utf-8"
    )


# ----------------- Schema Extraction -----------------

SCHEMA_EXTRACTION_PROMPT = """
You are an expert AI system designer. Extract a structured agent schema from the following voice prompt.
The schema should include the agent's name, description, properties, and actions.

Voice Prompt Transcript:
{transcript}

Create a JSON schema with the following structure:
{
  "name": "Agent name",
  "description": "Detailed description of what the agent does",
  "properties": [
    {
      "name": "property_name",
      "type": "data_type",
      "description": "Property description",
      "required": true/false,
      "default": "default_value" (optional)
    }
  ],
  "actions": [
    {
      "name": "action_name",
      "description": "What the action does",
      "parameters": [
        {
          "name": "parameter_name",
          "type": "data_type",
          "description": "Parameter description",
          "required": true/false
        }
      ],
      "returns": "return_type"
    }
  ]
}

Only include properties and actions explicitly mentioned or strongly implied in the transcript.
If not enough information is provided, create a minimal viable schema with placeholder values.
"""


async def extract_agent_schema(transcript: str) -> AgentSchema:
    """Extract agent schema from a voice prompt transcript using LLM."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
        )
        
        # Process with LLM
        messages = [
            SystemMessage(content="You are an AI assistant that extracts structured schemas from voice prompts."),
            HumanMessage(content=SCHEMA_EXTRACTION_PROMPT.format(transcript=transcript))
        ]
        
        response = await llm.agenerate([messages])
        content = response.generations[0][0].text
        
        # Extract JSON from response
        try:
            schema_dict = json.loads(content)
            agent_schema = AgentSchema(prompt_transcript=transcript, **schema_dict)
            return agent_schema
        except json.JSONDecodeError:
            # If the response is not JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'```json\n(.*?)```', content, re.DOTALL)
            if json_match:
                schema_dict = json.loads(json_match.group(1))
                agent_schema = AgentSchema(prompt_transcript=transcript, **schema_dict)
                return agent_schema
            else:
                raise ValueError("Failed to extract JSON schema from LLM response")
    
    except Exception as e:
        logger.error(f"Error extracting agent schema: {str(e)}")
        # Return a minimal schema with error information
        return AgentSchema(
            name="Error Schema",
            description=f"Failed to extract schema: {str(e)}",
            prompt_transcript=transcript,
            properties=[],
            actions=[]
        )


# ----------------- API Router -----------------

router = APIRouter()


@router.post("/voice-prompt", response_model=VoicePromptResponse)
async def process_voice_prompt(
    background_tasks: BackgroundTasks,
    audio_file: Optional[UploadFile] = File(None),
    request_data: Optional[VoicePromptRequest] = None,
    base64_audio: Optional[str] = Form(None),
    model_size: str = Form("large"),
    extract_agent_schema: bool = Form(True),
    store_in_redis: bool = Form(True),
    ttl: int = Form(3600),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Process a voice prompt to extract agent schema.
    
    Supports both file upload and base64 encoded audio.
    Uses Whisper-large for transcription by default.
    Can extract agent schema from the transcript using LLM.
    Can store the results in Redis for later retrieval.
    
    Returns a JSON response with the transcript, schema (if requested), and timing information.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Validate request data
    if audio_file is None and base64_audio is None and (request_data is None or request_data.base64_audio is None):
        raise HTTPException(
            status_code=400, 
            detail="Either audio_file or base64_audio must be provided"
        )
    
    # If request_data is provided, use it to override form parameters
    if request_data:
        model_size = request_data.model_size
        extract_agent_schema = request_data.extract_agent_schema
        store_in_redis = request_data.store_in_redis
        ttl = request_data.ttl
        if request_data.audio_source == AudioSource.BASE64:
            base64_audio = request_data.base64_audio
    
    # Process audio data
    try:
        speech_engine = SpeechEngine()
        audio_data = None
        
        # Handle file upload
        if audio_file:
            contents = await audio_file.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            try:
                transcription_start = time.time()
                transcript = speech_engine.transcribe_file(
                    file_path=tmp_path,
                    provider=SpeechProvider.WHISPER,
                    model=model_size
                )
                transcription_time = time.time() - transcription_start
                
                # Clean up the temporary file
                os.unlink(tmp_path)
            except Exception as e:
                # Clean up on error
                os.unlink(tmp_path)
                raise e
                
        # Handle base64 input
        elif base64_audio:
            try:
                audio_bytes = base64.b64decode(base64_audio)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                transcription_start = time.time()
                transcript = speech_engine.transcribe_file(
                    file_path=tmp_path,
                    provider=SpeechProvider.WHISPER,
                    model=model_size
                )
                transcription_time = time.time() - transcription_start
                
                # Clean up the temporary file
                os.unlink(tmp_path)
            except Exception as e:
                # Clean up on error
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
                raise e
        
        # Extract agent schema if requested
        agent_schema = None
        schema_extraction_time = None
        
        if extract_agent_schema:
            schema_start = time.time()
            agent_schema = await extract_agent_schema(transcript)
            schema_extraction_time = time.time() - schema_start
        
        # Prepare response
        processing_time = time.time() - start_time
        redis_key = None
        
        # Store in Redis if requested
        if store_in_redis:
            redis_key = f"voice_prompt:{request_id}"
            response_data = {
                "request_id": request_id,
                "transcript": transcript,
                "processing_time": processing_time,
                "transcription_time": transcription_time,
                "schema_extraction_time": schema_extraction_time,
                "agent_schema": agent_schema.dict() if agent_schema else None,
                "created_at": datetime.now().isoformat()
            }
            
            redis_client.setex(
                redis_key,
                ttl,
                json.dumps(response_data)
            )
        
        # Create response
        response = VoicePromptResponse(
            request_id=request_id,
            transcript=transcript,
            processing_time=processing_time,
            transcription_time=transcription_time,
            schema_extraction_time=schema_extraction_time,
            agent_schema=agent_schema,
            redis_key=redis_key
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing voice prompt: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process voice prompt: {str(e)}"
        )


@router.get("/voice-prompt/{request_id}", response_model=VoicePromptResponse)
async def get_voice_prompt_result(
    request_id: str,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Retrieve a previously processed voice prompt result from Redis.
    
    The result is identified by its request_id, which is returned when
    the voice prompt is initially processed.
    """
    redis_key = f"voice_prompt:{request_id}"
    
    try:
        result = redis_client.get(redis_key)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Voice prompt result not found for request_id: {request_id}"
            )
        
        result_data = json.loads(result)
        
        # Convert the agent schema back to an AgentSchema model if it exists
        if result_data.get("agent_schema"):
            agent_schema = AgentSchema(**result_data["agent_schema"])
            result_data["agent_schema"] = agent_schema
        
        return VoicePromptResponse(**result_data)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error retrieving voice prompt result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve voice prompt result: {str(e)}"
        )

