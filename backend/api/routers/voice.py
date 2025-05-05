import os
import uuid
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Union, Literal, BinaryIO, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path, status
from fastapi import File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

import speech_recognition as sr

# Configure logging
logger = logging.getLogger("aimpact_api.voice")

# Initialize router
router = APIRouter()

# -------------------- Pydantic Models --------------------

class STTEngine(str, Enum):
    """Supported speech-to-text engines."""
    WHISPER = "whisper"
    GOOGLE = "google"
    SPHINX = "sphinx"
    AZURE = "azure"

class TTSEngine(str, Enum):
    """Supported text-to-speech engines."""
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"
    LOCAL = "local"

class TTSVoice(str, Enum):
    """Common voice options for TTS."""
    MALE_1 = "male_1"
    MALE_2 = "male_2"
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    NEUTRAL = "neutral"

class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"

class TranscriptionRequest(BaseModel):
    """Request for speech-to-text conversion."""
    engine: STTEngine = Field(default=STTEngine.WHISPER, description="Speech recognition engine to use")
    language: Optional[str] = Field(default="en-US", description="Language code (e.g., en-US, de-DE)")
    enhanced_model: bool = Field(default=False, description="Whether to use enhanced model for better accuracy")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV, description="Format of the audio data")

class TranscriptionResponse(BaseModel):
    """Response from speech-to-text conversion."""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0-1)")
    language: str = Field(..., description="Detected or specified language")
    processing_time: float = Field(..., description="Processing time in seconds")
    engine: STTEngine = Field(..., description="Engine used for transcription")

class TTSRequest(BaseModel):
    """Request for text-to-speech conversion."""
    text: str = Field(..., description="Text to convert to speech")
    engine: TTSEngine = Field(default=TTSEngine.OPENAI, description="TTS engine to use")
    voice: TTSVoice = Field(default=TTSVoice.NEUTRAL, description="Voice to use")
    language: str = Field(default="en-US", description="Language code")
    output_format: AudioFormat = Field(default=AudioFormat.WAV, description="Output audio format")
    speed: float = Field(default=1.0, description="Speech speed multiplier (0.5-2.0)")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('speed')
    def speed_must_be_in_range(cls, v):
        if v < 0.5 or v > 2.0:
            raise ValueError('Speed must be between 0.5 and 2.0')
        return v

class TTSResponse(BaseModel):
    """Metadata response for text-to-speech conversion."""
    request_id: str = Field(..., description="Unique ID for this request")
    text_length: int = Field(..., description="Length of input text in characters")
    processing_time: float = Field(..., description="Processing time in seconds")
    engine: TTSEngine = Field(..., description="Engine used for synthesis")
    output_format: AudioFormat = Field(..., description="Format of the generated audio")

class VoiceInteractionRequest(BaseModel):
    """Request for voice-based interaction with an agent."""
    agent_id: str = Field(..., description="ID of the agent to interact with")
    stt_engine: STTEngine = Field(default=STTEngine.WHISPER, description="Speech-to-text engine")
    tts_engine: TTSEngine = Field(default=TTSEngine.OPENAI, description="Text-to-speech engine")
    voice: TTSVoice = Field(default=TTSVoice.NEUTRAL, description="Voice for response")
    language: str = Field(default="en-US", description="Language code")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuing conversation")

class VoiceInteractionResponse(BaseModel):
    """Response from voice-based interaction with an agent."""
    interaction_id: str = Field(..., description="Unique ID for this interaction")
    session_id: str = Field(..., description="Session ID for the conversation")
    agent_id: str = Field(..., description="ID of the agent that responded")
    transcription: TranscriptionResponse = Field(..., description="Transcription of the input audio")
    response_text: str = Field(..., description="Text response from the agent")
    audio_available: bool = Field(..., description="Whether audio response is available")

class StreamingTranscriptionConfig(BaseModel):
    """Configuration for real-time streaming transcription."""
    engine: STTEngine = Field(default=STTEngine.WHISPER, description="Speech recognition engine to use")
    language: str = Field(default="en-US", description="Language code")
    interim_results: bool = Field(default=True, description="Whether to return interim results")
    vad_sensitivity: float = Field(default=0.5, description="Voice activity detection sensitivity (0-1)")

class StreamingTranscriptionResponse(BaseModel):
    """Response from streaming transcription."""
    text: str = Field(..., description="Transcribed text")
    is_final: bool = Field(..., description="Whether this is a final result")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1) for final results")

# -------------------- Mock Services --------------------

class SpeechRecognitionService:
    """Service for handling speech recognition."""
    
    @staticmethod
    def recognize_from_file(file: BinaryIO, engine: STTEngine, language: str) -> Dict[str, Any]:
        """
        Perform speech recognition on an audio file.
        
        This is a simplified implementation using the speech_recognition library.
        """
        start_time = datetime.now()
        
        recognizer = sr.Recognizer()
        
        # In a real implementation, we would handle different file formats properly
        audio_data = sr.AudioFile(file)
        
        try:
            with audio_data as source:
                audio = recognizer.record(source)
                
                result = {}
                
                if engine == STTEngine.GOOGLE:
                    # Simulate Google Speech Recognition
                    # In a real implementation: text = recognizer.recognize_google(audio, language=language)
                    text = "This is a simulated transcription from Google Speech Recognition."
                    confidence = 0.92
                
                elif engine == STTEngine.WHISPER:
                    # Simulate OpenAI Whisper
                    # In a real implementation: text = recognizer.recognize_whisper(audio, language=language)
                    text = "This is a simulated transcription from OpenAI Whisper."
                    confidence = 0.95
                
                elif engine == STTEngine.SPHINX:
                    # Simulate CMU Sphinx
                    # In a real implementation: text = recognizer.recognize_sphinx(audio, language=language)
                    text = "This is a simulated transcription from CMU Sphinx."
                    confidence = 0.85
                
                elif engine == STTEngine.AZURE:
                    # Simulate Microsoft Azure Speech
                    # In a real implementation: text = recognizer.recognize_azure(audio, language=language)
                    text = "This is a simulated transcription from Microsoft Azure Speech."
                    confidence = 0.93
                
                else:
                    raise ValueError(f"Unsupported STT engine: {engine}")
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                return {
                    "text": text,
                    "confidence": confidence,
                    "language": language,
                    "processing_time": processing_time,
                    "engine": engine
                }
        
        except Exception as e:
            logger.error(f"Speech recognition error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech recognition failed: {str(e)}"
            )

class TextToSpeechService:
    """Service for handling text-to-speech synthesis."""
    
    @staticmethod
    def synthesize_speech(request: TTSRequest) -> Tuple[bytes, float]:
        """
        Synthesize speech from text.
        
        This is a simplified implementation that would normally call external TTS services.
        For demonstration, it returns a simulated audio response.
        """
        start_time = datetime.now()
        
        try:
            # In a real implementation, this would call the appropriate TTS service
            # For this demo, we'll simulate TTS by returning a placeholder WAV file
            
            # Create a simple sine wave as placeholder audio
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple tone
            sample_rate = 22050
            duration = min(10, len(request.text) * 0.07)  # Roughly scale duration with text length
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Create a frequency that varies based on the voice selection
            if request.voice == TTSVoice.MALE_1:
                freq = 100
            elif request.voice == TTSVoice.MALE_2:
                freq = 120
            elif request.voice == TTSVoice.FEMALE_1:
                freq = 210
            elif request.voice == TTSVoice.FEMALE_2:
                freq = 230
            else:  # NEUTRAL
                freq = 165
            
            # Generate a simple sine wave
            audio_data = np.sin(2 * np.pi * freq * t) * 0.5
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save to BytesIO
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_data)
            buffer.seek(0)
            
            audio_bytes = buffer.getvalue()
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return audio_bytes, processing_time
        
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text-to-speech synthesis failed: {str(e)}"
            )

# -------------------- API Endpoints --------------------

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: TranscriptionRequest = Depends(),
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe speech from an audio file.
    
    Returns the transcribed text along with metadata.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate file extension
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in [fmt.value for fmt in AudioFormat]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {file_ext}"
        )
    
    # Process file with specified engine
    logger.info(f"Transcribing audio with {request.engine} engine")
    
    try:
        # Save file to temporary location for processing
        content = await file.read()
        temp_file = io.BytesIO(content)
        
        # Perform speech recognition
        result = SpeechRecognitionService.recognize_from_file(
            temp_file,
            request.engine,
            request.language
        )
        
        # Return response
        return TranscriptionResponse(**result)
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech.
    
    Returns metadata about the synthesized audio and streams the audio data.
    """
    logger.info(f"Synthesizing speech with {request.engine} engine")
    
    try:
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Synthesize speech
        audio_bytes, processing_time = TextToSpeechService.synthesize_speech(request)
        
        # Create metadata response
        response = TTSResponse(
            request_id=request_id,
            text_length=len(request.text),
            processing_time=processing_time,
            engine=request.engine,
            output_format=request.output_format
        )
        
        # Return streaming response with metadata in headers
        headers = {
            "X-Request-ID": request_id,
            "X-Processing-Time": str(processing_time),
            "X-Engine": request.engine.value,
        }
        
        content_type = f"audio/{request.output_format.value}"
        if request.output_format == AudioFormat.WAV:
            content_type = "audio/wav"
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers=headers
        )
    
    except Exception as e:
        logger.error(f"Speech synthesis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(e)}"
        )

@router.post("/interact", response_model=VoiceInteractionResponse)
async def voice_agent_interaction(
    request: VoiceInteractionRequest = Depends(),
    file: UploadFile = File(..., description="Audio file with user speech")
):
    """
    Interact with an AI agent using voice.
    
    Transcribes user speech, sends it to the specified agent,
    and returns the agent's response (text and synthesized speech).
    """
    # Validate agent ID
    # Note: In a real implementation, this would verify the agent exists
    agent_id = request.agent_id
    
    # Generate a session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    interaction_id = str(uuid.uuid4())
    
    try:
        # Step 1: Transcribe the audio
        content = await file.read()
        temp_file = io.BytesIO(content)
        
        transcription_result = SpeechRecognitionService.recognize_from_file(
            temp_file,
            request.stt_engine,
            request.language
        )
        
        transcription = TranscriptionResponse(**transcription_result)
        
        logger.info(f"Voice interaction transcription: '{transcription.text}'")
        
        # Step 2: Send the transcribed text to the agent
        # In a real implementation, this would use the agents router
        # For this demo, we'll simulate an agent response
        from .agents import MessageRole, Message
        
        # Create a simulated request to the agent
        agent_message = Message(
            role=MessageRole.USER,
            content=transcription.text
        )
        
        # For a real implementation, we would do:
        # from .agents import interact_with_agent
        # agent_response = await interact_with_agent(agent_id, [agent_message])
        
        # Simulate agent response
        agent_response_text = f"This is a simulated response from agent {agent_id} to: '{transcription.text}'"
        
        # Step 3: Synthesize speech from the agent's response
        tts_request = TTSRequest(
            text=agent_response_text,
            engine=request.tts_engine,
            voice=request.voice,
            language=request.language,
            output_format=AudioFormat.WAV
        )
        
        # Generate the audio
        audio_bytes, processing_time = TextToSpeechService.synthesize_speech(tts_request)
        
        # Store the audio for subsequent retrieval
        # In a real implementation, this would be stored in a database or cache
        # For this demo, we'll pretend it's stored
        audio_available = True
        
        # Step 4: Save the interaction history
        # In a real implementation, this would update a conversation history in a database
        # Here we'll just log it
        logger.info(f"Interaction {interaction_id}: User said '{transcription.text}', Agent responded '{agent_response_text}'")
        
        # Return the response
        return VoiceInteractionResponse(
            interaction_id=interaction_id,
            session_id=session_id,
            agent_id=agent_id,
            transcription=transcription,
            response_text=agent_response_text,
            audio_available=audio_available
        )
    
    except Exception as e:
        logger.error(f"Voice interaction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice interaction failed: {str(e)}"
        )

@router.get("/interact/{interaction_id}/audio")
async def get_interaction_audio(interaction_id: str = Path(..., description="The ID of the interaction")):
    """
    Retrieve the synthesized audio from a previous voice interaction.
    
    Returns the audio data as a streaming response.
    """
    try:
        # In a real implementation, this would retrieve the audio from a database or cache
        # For this demo, we'll simulate it by generating a new audio file
        
        # Simulate retrieving the interaction
        # In reality, this would come from a database
        agent_response_text = f"This is simulated audio for interaction {interaction_id}."
        
        # Generate audio on-the-fly (in a real system, this would be retrieved)
        tts_request = TTSRequest(
            text=agent_response_text,
            engine=TTSEngine.OPENAI,
            voice=TTSVoice.NEUTRAL,
            language="en-US",
            output_format=AudioFormat.WAV
        )
        
        # Generate the audio
        audio_bytes, _ = TextToSpeechService.synthesize_speech(tts_request)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"X-Interaction-ID": interaction_id}
        )
    
    except Exception as e:
        logger.error(f"Error retrieving interaction audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audio: {str(e)}"
        )

# -------------------- Session Management --------------------

# Simple in-memory session storage
# In a real implementation, this would be replaced with a database
voice_sessions: Dict[str, Dict[str, Any]] = {}

class VoiceSession(BaseModel):
    """Model for a voice interaction session."""
    session_id: str
    agent_id: str
    created_at: datetime
    last_active: datetime
    language: str = "en-US"
    stt_engine: STTEngine = STTEngine.WHISPER
    tts_engine: TTSEngine = TTSEngine.OPENAI
    voice: TTSVoice = TTSVoice.NEUTRAL
    interactions: List[str] = Field(default_factory=list)  # List of interaction IDs

@router.post("/sessions", response_model=VoiceSession)
async def create_voice_session(
    agent_id: str = Form(..., description="ID of the agent to interact with"),
    language: str = Form("en-US", description="Language code"),
    stt_engine: STTEngine = Form(STTEngine.WHISPER, description="Speech-to-text engine"),
    tts_engine: TTSEngine = Form(TTSEngine.OPENAI, description="Text-to-speech engine"),
    voice: TTSVoice = Form(TTSVoice.NEUTRAL, description="Voice for responses")
):
    """
    Create a new voice interaction session.
    
    Returns a session ID that can be used for subsequent interactions.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now()
    
    new_session = VoiceSession(
        session_id=session_id,
        agent_id=agent_id,
        created_at=now,
        last_active=now,
        language=language,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        voice=voice
    )
    
    # Store session
    voice_sessions[session_id] = new_session.dict()
    
    logger.info(f"Created new voice session: {session_id} for agent {agent_id}")
    
    return new_session

@router.get("/sessions/{session_id}", response_model=VoiceSession)
async def get_voice_session(session_id: str = Path(..., description="Session ID")):
    """Get details of a voice interaction session."""
    if session_id not in voice_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return voice_sessions[session_id]

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_voice_session(session_id: str = Path(..., description="Session ID")):
    """End a voice interaction session."""
    if session_id not in voice_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Delete session
    del voice_sessions[session_id]
    
    logger.info(f"Ended voice session: {session_id}")
    
    return None

# -------------------- WebSocket Support --------------------

class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        # Map of session ID to WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Add a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connection established for session {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket connection closed for session {session_id}")
    
    async def send_text(self, session_id: str, message: str):
        """Send a text message to a specific connection."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
    
    async def send_json(self, session_id: str, data: Dict[str, Any]):
        """Send a JSON message to a specific connection."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)
    
    async def send_bytes(self, session_id: str, data: bytes):
        """Send binary data to a specific connection."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_bytes(data)

# Initialize connection manager
manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Allows for continuous voice interaction with an agent.
    """
    # Check if session exists
    if session_id not in voice_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    # Get session
    session = voice_sessions[session_id]
    
    # Connect WebSocket
    await manager.connect(websocket, session_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_json(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to voice streaming service"
        })
        
        # Update session
        session["last_active"] = datetime.now()
        
        # Process incoming messages
        while True:
            # Receive message
            message = await websocket.receive()
            
            # Check message type
            if "text" in message:
                # Parse JSON commands
                try:
                    data = json.loads(message["text"])
                    command = data.get("command")
                    
                    if command == "start_streaming":
                        # Client is starting to stream audio
                        await manager.send_json(session_id, {
                            "type": "streaming_started",
                            "message": "Ready to receive audio"
                        })
                    
                    elif command == "end_streaming":
                        # Client has finished streaming audio
                        await manager.send_json(session_id, {
                            "type": "streaming_ended",
                            "message": "Audio stream ended"
                        })
                    
                    else:
                        await manager.send_json(session_id, {
                            "type": "error",
                            "message": f"Unknown command: {command}"
                        })
                
                except json.JSONDecodeError:
                    await manager.send_json(session_id, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
            
            elif "bytes" in message:
                # Process audio chunk
                audio_chunk = message["bytes"]
                
                # In a real implementation, this would:
                # 1. Buffer the audio chunks
                # 2. Perform streaming speech recognition
                # 3. When appropriate (e.g., end of utterance), send to agent
                # 4. Stream back the agent's response
                
                # For this demo, we'll just echo back a confirmation
                await manager.send_json(session_id, {
                    "type": "chunk_received",
                    "size": len(audio_chunk),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Simulate processing every few chunks (in reality would be VAD-based)
                # This is just a simplified example
                if len(audio_chunk) > 1000:  # Arbitrary threshold
                    # Simulate transcription
                    transcription = "This is a simulated real-time transcription."
                    
                    # Send transcription to client
                    await manager.send_json(session_id, {
                        "type": "transcription",
                        "text": transcription,
                        "is_final": True
                    })
                    
                    # Simulate agent response
                    agent_response = f"This is a simulated agent response to: '{transcription}'"
                    
                    # Send agent response to client
                    await manager.send_json(session_id, {
                        "type": "agent_response",
                        "text": agent_response
                    })
                    
                    # Simulate TTS (in a real implementation, this would stream audio back)
                    await manager.send_json(session_id, {
                        "type": "tts_complete",
                        "message": "TTS response ready"
                    })
    
    except WebSocketDisconnect:
        # Handle disconnection
        manager.disconnect(session_id)
    
    except Exception as e:
        # Handle other errors
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await manager.send_json(session_id, {
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        finally:
            manager.disconnect(session_id)

# Add JSON import at the top
import json

