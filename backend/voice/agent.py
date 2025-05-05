"""
AImpact Advanced Voice Agent

This module provides a high-level orchestration layer for voice-based AI agents,
coordinating speech recognition, emotion detection, voice cloning, and language models
into a unified system with advanced capabilities.

The VoiceAgent acts as the central coordinator for all voice processing components,
implementing context-aware conversations, emotional response matching, dynamic voice
adaptation, and multi-modal processing.

Features:
- Seamless integration of all voice processing components
- Context-aware conversation management
- Emotional intelligence and response matching
- Dynamic voice adaptation based on context
- Multi-modal input processing (audio, text, etc.)
- Workflow creation and management through voice
- Real-time interaction with low latency
- Advanced customization and adaptation
"""

import os
import time
import asyncio
import logging
import uuid
import json
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, BinaryIO, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io
from contextlib import asynccontextmanager

import numpy as np
from pydantic import BaseModel, Field
from fastapi import WebSocket

# Import local modules
from backend.voice.core.speech_engine import (
    SpeechEngine, SpeechProvider, RecognitionMode, RecognitionQuality,
    SynthesisVoiceStyle, AudioFormat
)
from backend.voice.core.emotion_detector import (
    EmotionDetector, EmotionDetectionResult, BasicEmotion, 
    AdvancedEmotion, EmotionIntensity, CulturalContext
)
from backend.voice.core.voice_cloner import (
    VoiceCloner, VoiceModel, VoicePersonality, VoiceAge,
    VoiceGender, AccentType, VoiceEmotionParams, CloneQuality
)

# Configure logging
logger = logging.getLogger("aimpact.voice.agent")

# ----------------- Enums and Models -----------------

class VoiceAgentMode(str, Enum):
    """Operating modes for voice agents."""
    STANDARD = "standard"  # Basic mode with standard features
    HIGH_QUALITY = "high_quality"  # High-quality voice with higher latency
    LOW_LATENCY = "low_latency"  # Optimized for real-time with lower quality
    ADAPTIVE = "adaptive"  # Dynamically adapts based on context
    CUSTOM = "custom"  # Custom configuration


class VoiceAgentCapability(str, Enum):
    """Capabilities that can be enabled for voice agents."""
    SPEECH_RECOGNITION = "speech_recognition"
    SPEECH_SYNTHESIS = "speech_synthesis"
    EMOTION_DETECTION = "emotion_detection"
    EMOTION_RESPONSE = "emotion_response"
    VOICE_CLONING = "voice_cloning"
    VOICE_ADAPTATION = "voice_adaptation"
    MULTILINGUAL = "multilingual"
    NOISE_REDUCTION = "noise_reduction"
    CONTEXT_AWARENESS = "context_awareness"
    ACCENT_ADAPTATION = "accent_adaptation"
    MULTI_SPEAKER = "multi_speaker"
    VOICE_COMMANDS = "voice_commands"
    WORKFLOW_CREATION = "workflow_creation"
    STREAMING = "streaming"


class InputModality(str, Enum):
    """Input modalities supported by the voice agent."""
    AUDIO = "audio"  # Raw audio input
    TEXT = "text"  # Text input
    MULTIMODAL = "multimodal"  # Combined audio and text


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    speaker: str = Field(..., description="Speaker identifier (user or agent)")
    input_modality: InputModality = Field(InputModality.AUDIO, description="Input modality")
    text: str = Field(..., description="Text content of the turn")
    audio_data: Optional[bytes] = Field(None, description="Audio data if available")
    emotion: Optional[EmotionDetectionResult] = Field(None, description="Detected emotion")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    language: Optional[str] = Field(None, description="Detected or specified language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationContext(BaseModel):
    """Context for a conversation session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    turns: List[ConversationTurn] = Field(default_factory=list)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    agent_profile: Dict[str, Any] = Field(default_factory=dict)
    language: str = Field("en-US", description="Primary language")
    cultural_context: CulturalContext = Field(CulturalContext.GLOBAL)
    current_emotion: Optional[EmotionDetectionResult] = Field(None)
    current_workflow: Optional[str] = Field(None)
    current_topic: Optional[str] = Field(None)
    current_voice_model: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the context."""
        self.turns.append(turn)
        self.last_active = datetime.now()
        
        # Update current emotion if available
        if turn.emotion:
            self.current_emotion = turn.emotion
            
    def get_conversation_history(self, max_turns: int = None) -> List[Dict[str, Any]]:
        """Get conversation history formatted for LLM context."""
        history = []
        turns = self.turns[-max_turns:] if max_turns else self.turns
        
        for turn in turns:
            role = "user" if turn.speaker == "user" else "assistant"
            history.append({
                "role": role,
                "content": turn.text
            })
            
        return history


class VoiceAgentConfig(BaseModel):
    """Configuration for a voice agent."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="User-friendly name")
    description: Optional[str] = Field(None, description="Description of the agent")
    mode: VoiceAgentMode = Field(VoiceAgentMode.ADAPTIVE, description="Operating mode")
    enabled_capabilities: List[VoiceAgentCapability] = Field(
        default_factory=lambda: list(VoiceAgentCapability),
        description="Enabled capabilities"
    )
    default_language: str = Field("en-US", description="Default language")
    supported_languages: List[str] = Field(default_factory=list)
    default_voice_model: Optional[str] = Field(None, description="Default voice model ID")
    default_emotion_response: bool = Field(True, description="Whether to match emotions by default")
    speech_recognition_provider: SpeechProvider = Field(
        SpeechProvider.WHISPER,
        description="Default speech recognition provider"
    )
    speech_synthesis_provider: SpeechProvider = Field(
        SpeechProvider.OPENAI,
        description="Default speech synthesis provider"
    )
    context_window_size: int = Field(
        10,
        description="Number of conversation turns to keep in context"
    )
    streaming_enabled: bool = Field(
        True,
        description="Whether streaming processing is enabled"
    )
    system_prompt: str = Field(
        "You are a helpful voice assistant that responds naturally in conversation.",
        description="System prompt for the language model"
    )
    log_level: str = Field(
        "INFO",
        description="Logging level"
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom settings for the agent"
    )


# ----------------- Voice Agent Implementation -----------------

class VoiceAgent:
    """
    Advanced Voice Agent that coordinates speech recognition, emotion detection,
    voice cloning, and LLM capabilities with context-awareness and real-time processing.
    """
    
    def __init__(self, config: VoiceAgentConfig):
        """Initialize the voice agent with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"aimpact.voice.agent.{config.agent_id}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Initialize component engines
        self.speech_engine = SpeechEngine()
        self.emotion_detector = EmotionDetector()
        self.voice_cloner = VoiceCloner()
        
        # Session and context management
        self.active_sessions: Dict[str, ConversationContext] = {}
        self.voice_models: Dict[str, VoiceModel] = {}
        
        # Initialize the agent
        self._initialize()
        
        self.logger.info(f"Voice Agent '{config.name}' ({config.agent_id}) initialized")
    
    def _initialize(self):
        """Initialize the agent components based on configuration."""
        # Configure capabilities based on mode
        if self.config.mode == VoiceAgentMode.HIGH_QUALITY:
            # Optimize for quality
            self.recognition_mode = RecognitionMode.HIGH_ACCURACY
            self.recognition_quality = RecognitionQuality.HIGH
            self.clone_quality = CloneQuality.HIGH_QUALITY
        elif self.config.mode == VoiceAgentMode.LOW_LATENCY:
            # Optimize for latency
            self.recognition_mode = RecognitionMode.REALTIME
            self.recognition_quality = RecognitionQuality.LOW
            self.clone_quality = CloneQuality.ULTRA_LOW_LATENCY
        else:
            # Balanced or adaptive mode
            self.recognition_mode = RecognitionMode.STANDARD
            self.recognition_quality = RecognitionQuality.MEDIUM
            self.clone_quality = CloneQuality.BALANCED
        
        # Load default voice model if specified
        if self.config.default_voice_model:
            self._load_voice_model(self.config.default_voice_model)
    
    async def process_audio(self, audio_data: Union[bytes, BinaryIO], session_id: str = None,
                           language: str = None) -> dict:
        """
        Process audio input and generate a response with the appropriate voice.
        
        Args:
            audio_data: Audio data as bytes or file-like object
            session_id: Session ID for context, creates new if None
            language: Language code, uses default if None
            
        Returns:
            Dictionary with the response and metadata
        """
        start_time = time.time()
        session = self._get_or_create_session(session_id)
        language = language or self.config.default_language
        
        try:
            # Step 1: Speech Recognition
            recognition_result = await self.speech_engine.recognize(
                audio_data,
                language=language,
                mode=self.recognition_mode,
                quality=self.recognition_quality
            )
            
            transcribed_text = recognition_result.get("text", "")
            confidence = recognition_result.get("confidence", 0.0)
            
            if not transcribed_text:
                self.logger.warning("No speech detected in the audio")
                return {
                    "success": False,
                    "error": "No speech detected",
                    "session_id": session.session_id
                }
            
            # Step 2: Emotion Detection (if enabled)
            emotion_result = None
            if VoiceAgentCapability.EMOTION_DETECTION in self.config.enabled_capabilities:
                emotion_result = await self.emotion_detector.detect_emotion(
                    audio_data=audio_data,
                    text=transcribed_text,
                    language=language,
                    cultural_context=session.cultural_context
                )
            
            # Step 3: Create user turn and add to context
            user_turn = ConversationTurn(
                speaker="user",
                input_modality=InputModality.AUDIO,
                text=transcribed_text,
                audio_data=audio_data if isinstance(audio_data, bytes) else None,
                emotion=emotion_result,
                duration=recognition_result.get("audio_duration"),
                language=language,
                metadata={
                    "confidence": confidence,
                    "recognition_provider": recognition_result.get("provider", "unknown")
                }
            )
            session.add_turn(user_turn)
            
            # Step 4: Generate response using conversation context
            response_text = await self._generate_response(session)
            
            # Step 5: Prepare voice for response (emotion matching if enabled)
            voice_style = SynthesisVoiceStyle.NEUTRAL
            voice_emotion_params = None
            
            if (VoiceAgentCapability.EMOTION_RESPONSE in self.config.enabled_capabilities and
                emotion_result and self.config.default_emotion_response):
                # Match response emotion to detected user emotion
                primary_emotion = emotion_result.primary_emotion.emotion
                if isinstance(primary_emotion, BasicEmotion):
                    voice_emotion_params = VoiceEmotionParams.for_emotion(
                        primary_emotion.value,
                        intensity=emotion_result.primary_emotion.intensity or 0.5
                    )
                    
                    # Map basic emotions to voice styles
                    emotion_to_style = {
                        BasicEmotion.HAPPY: SynthesisVoiceStyle.CHEERFUL,
                        BasicEmotion.SAD: SynthesisVoiceStyle.SAD,
                        BasicEmotion.ANGRY: SynthesisVoiceStyle.ANGRY,
                        BasicEmotion.FEARFUL: SynthesisVoiceStyle.WHISPERED,
                        BasicEmotion.DISGUSTED: SynthesisVoiceStyle.SERIOUS,
                        BasicEmotion.SURPRISED: SynthesisVoiceStyle.EXCITED,
                        BasicEmotion.CALM: SynthesisVoiceStyle.NEUTRAL
                    }
                    voice_style = emotion_to_style.get(
                        primary_emotion, 
                        SynthesisVoiceStyle.NEUTRAL
                    )
            
            # Step 6: Synthesize speech response with appropriate voice
            synthesis_result = await self.speech_engine.synthesize(
                text=response_text,
                language=language,
                voice_style=voice_style,
                output_format=AudioFormat.WAV,
                voice_params=voice_emotion_params
            )
            
            response_audio = synthesis_result[0]  # Audio data
            synthesis_metadata = synthesis_result[1]  # Metadata
            
            # Step 7: Create agent turn and add to context
            agent_turn = ConversationTurn(
                speaker="agent",
                input_modality=InputModality.TEXT,
                text=response_text,
                audio_data=response_audio,
                duration=synthesis_metadata.get("duration"),
                language=language,
                metadata={
                    "synthesis_provider": synthesis_metadata.get("provider", "unknown"),
                    "voice_style": voice_style.value,
                    "processing_time": time.time() - start_time
                }
            )
            session.add_turn(agent_turn)
            
            # Step 8: Return response with all relevant data
            return {
                "success": True,
                "session_id": session.session_id,
                "input": {
                    "text": transcribed_text,
                    "emotion": emotion_result.to_dict() if emotion_result else None,
                    "language": language,
                    "confidence": confidence
                },
                "response": {
                    "text": response_text,
                    "audio": response_audio,  # In a real API, this would be a URL or base64 encoded
                    "duration": synthesis_metadata.get("duration"),
                    "voice_style": voice_style.value
                },
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "session_id": session.session_id if session else None
            }
    
    async def process_text(self, text: str, session_id: str = None,
                          language: str = None, audio_response: bool = True) -> dict:
        """
        Process text input and generate a response.
        
        Args:
            text: Text input from the user
            session_id: Session ID for context, creates new if None
            language: Language code, uses default if None
            audio_response: Whether to generate audio for the response
            
        Returns:
            Dictionary with the response and metadata
        """
        start_time = time.time()
        session = self._get_or_create_session(session_id)
        language = language or self.config.default_language
        
        try:
            # Step 1: Create user turn and add to context
            user_turn = ConversationTurn(
                speaker="user",
                input_modality=InputModality.TEXT,
                text=text,
                language=language
            )
            session.add_turn(user_turn)
            
            # Step 2: Generate response using conversation context
            response_text = await self._generate_response(session)
            
            # Step 3: Generate audio if requested
            response_audio = None
            synthesis_metadata = {}
            
            if audio_response:
                voice_style = SynthesisVoiceStyle.NEUTRAL
                synthesis_result = await self.speech_engine.synthesize(
                    text=response_text,
                    language=language,
                    voice_style=voice_style,
                    output_format=AudioFormat.WAV
                )
                response_audio = synthesis_result[0]
                synthesis_metadata = synthesis_result[1]
            
            # Step 4: Create agent turn and add to context
            agent_turn = ConversationTurn(
                speaker="agent",
                input_modality=InputModality.TEXT,
                text=response_text,
                audio_data=response_audio,
                duration=synthesis_metadata.get("duration"),
                language=language,
                metadata={
                    "synthesis_provider": synthesis_metadata.get("provider", "unknown") if audio_response else None,
                    "processing_time": time.time() - start_time
                }
            )
            session.add_turn(agent_turn)
            
            # Step 5: Return response
            return {
                "success": True,
                "session_id": session.session_id,
                "input": {
                    "text": text,
                    "language": language
                },
                "response": {
                    "text": response_text,
                    "audio": response_audio if audio_response else None,
                    "duration": synthesis_metadata.get("duration") if audio_response else None
                },
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "session_id": session.session_id if session else None
            }
    
    async def _generate_response(self, session: ConversationContext) -> str:
        """
        Generate a response using the LLM based on conversation context.
        
        In a real implementation, this would call an LLM through a separate service.
        For this demo, we'll generate a simple response.
        """
        # Get conversation history
        history = session.get_conversation_history(self.config.context_window_size)
        
        # Check for command patterns in the last user message
        if history and history[-1]["role"] == "user":
            last_message = history[-1]["content"].strip().lower()
            
            # Check for possible voice commands if that capability is enabled
            if VoiceAgentCapability.VOICE_COMMANDS in self.config.enabled_capabilities:
                command_response = self._check_for_commands(last_message, session)
                if command_response:
                    return command_response
        
        # In a real implementation, this would call the LLM with the history
        # For this demo, generate a simple response
        if not history:
            return "Hello! How can I assist you today?"
        
        last_user_message = "No message"
        for msg in reversed(history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        # Simple response generation
        return f"I understand you said: '{last_user_message}'. How can I help with that?"
    
    def _check_for_commands(self, message: str, session: ConversationContext) -> Optional[str]:
        """Check if a message contains voice commands and process them."""
        # Command patterns
        create_workflow_patterns = ["create workflow", "new workflow", "make a workflow"]
        voice_model_patterns = ["change voice", "use voice", "switch voice"]
        agent_config_patterns = ["configure agent", "agent settings", "update agent"]
        
        # Check for workflow creation commands
        if any(pattern in message for pattern in create_workflow_patterns) and \
           VoiceAgentCapability.WORKFLOW_CREATION in self.config.enabled_capabilities:
            # Extract workflow name (if any)
            workflow_name = None
            for pattern in ["called", "named", "with name"]:
                if pattern in message:
                    parts = message.split(pattern, 1)
                    if len(parts) > 1:
                        workflow_name = parts[1].strip().strip('"\'')
                        break
            
            workflow_name = workflow_name or f"Workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # In a real implementation, this would create a workflow
            session.current_workflow = workflow_name
            return f"I've created a new workflow called '{workflow_name}'. What would you like to add to it?"
        
        # Check for voice model commands
        elif any(pattern in message for pattern in voice_model_patterns) and \
             VoiceAgentCapability.VOICE_ADAPTATION in self.config.enabled_capabilities:
            # Extract voice characteristics
            voice_gender = None
            voice_age = None
            voice_personality = None
            
            # Check for gender
            for gender in ["male", "female", "neutral"]:
                if gender in message:
                    voice_gender = gender
                    break
            
            # Check for age
            for age in ["young", "adult", "mature", "senior", "child"]:
                if age in message:
                    voice_age = age
                    break
            
            # Check for personality
            for personality in ["professional", "friendly", "serious", "casual", "energetic"]:
                if personality in message:
                    voice_personality = personality
                    break
            
            # Generate response based on what was specified
            response = "I've updated my voice settings"
            if voice_gender or voice_age or voice_personality:
                response += " to be more"
                if voice_gender:
                    response += f" {voice_gender}"
                if voice_age:
                    response += f" {voice_age}"
                if voice_personality:
                    response += f" and {voice_personality}"
            
            return response + ". How does this sound?"
        
        # Check for agent configuration commands
        elif any(pattern in message for pattern in agent_config_patterns):
            # In a real implementation, this would update agent settings
            return "I can help you configure my settings. What would you like to change?"
        
        # No command detected
        return None
    
    # ----------------- Streaming Methods -----------------
    
    async def process_streaming_audio(self, websocket: WebSocket, session_id: str = None,
                                    language: str = None):
        """
        Process streaming audio through WebSocket and provide real-time responses.
        
        Args:
            websocket: The WebSocket connection
            session_id: Session ID for context, creates new if None
            language: Language code, uses default if None
        """
        if not self.config.streaming_enabled:
            await websocket.close(code=1000, reason="Streaming is not enabled for this agent")
            return
        
        session = self._get_or_create_session(session_id)
        language = language or self.config.default_language
        
        # Initialize streaming components
        streaming_buffer = io.BytesIO()
        streaming_active = True
        transcription_buffer = ""
        last_transcription_time = time.time()
        chunk_count = 0
        
        try:
            # Send initial connection message
            await websocket.send_json({
                "type": "connection_established",
                "session_id": session.session_id,
                "message": "Connected to voice agent streaming service"
            })
            
            while streaming_active:
                # Receive message from WebSocket
                message = await websocket.receive()
                
                if "bytes" in message:
                    # Process audio chunk
                    audio_chunk = message["bytes"]
                    streaming_buffer.write(audio_chunk)
                    chunk_count += 1
                    
                    # Process after accumulating enough chunks or if enough time has passed
                    current_time = time.time()
                    buffer_size = streaming_buffer.tell()
                    
                    if (buffer_size > 16000 * 2 * 0.5 or  # ~0.5 seconds of audio at 16kHz
                        (current_time - last_transcription_time > 0.5 and buffer_size > 0)):
                        
                        # Process the accumulated audio
                        streaming_buffer.seek(0)
                        chunk_audio = streaming_buffer.getvalue()
                        
                        # Perform speech recognition on the chunk
                        recognition_result = await self.speech_engine.recognize(
                            chunk_audio,
                            language=language,
                            mode=RecognitionMode.REALTIME,
                            quality=RecognitionQuality.LOW
                        )
                        
                        # Get transcribed text
                        chunk_text = recognition_result.get("text", "").strip()
                        
                        if chunk_text:
                            # Accumulate transcription
                            transcription_buffer += " " + chunk_text
                            transcription_buffer = transcription_buffer.strip()
                            
                            # Send interim transcription to client
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription_buffer,
                                "is_final": False
                            })
                            
                            # Check for end of speech or commands
                            if self._check_for_end_marker(chunk_text):
                                # Generate response for the accumulated transcription
                                response_text = await self._generate_response(session)
                                
                                # Send the response
                                await websocket.send_json({
                                    "type": "agent_response",
                                    "text": response_text
                                })
                                
                                # Synthesize speech if needed
                                synthesis_result = await self.speech_engine.synthesize(
                                    text=response_text,
                                    language=language,
                                    voice_style=SynthesisVoiceStyle.NEUTRAL,
                                    output_format=AudioFormat.WAV
                                )
                                
                                response_audio = synthesis_result[0]
                                
                                # Stream the audio back in chunks
                                chunk_size = 4096  # Send in 4KB chunks
                                for i in range(0, len(response_audio), chunk_size):
                                    await websocket.send_bytes(response_audio[i:i+chunk_size])
                                
                                # Signal end of audio
                                await websocket.send_json({
                                    "type": "audio_complete"
                                })
                                
                                # Add the conversation turns to the session
                                user_turn = ConversationTurn(
                                    speaker="user",
                                    input_modality=InputModality.AUDIO,
                                    text=transcription_buffer,
                                    language=language
                                )
                                
                                agent_turn = ConversationTurn(
                                    speaker="agent",
                                    input_modality=InputModality.TEXT,
                                    text=response_text,
                                    audio_data=response_audio,
                                    language=language
                                )
                                
                                session.add_turn(user_turn)
                                session.add_turn(agent_turn)
                                
                                # Reset buffers
                                transcription_buffer = ""
                                streaming_buffer = io.BytesIO()
                            
                        # Reset for next chunk
                        last_transcription_time = current_time
                        streaming_buffer = io.BytesIO()
                
                elif "text" in message:
                    # Process text commands
                    try:
                        command = json.loads(message["text"])
                        
                        if command.get("type") == "end_session":
                            # End the streaming session
                            await websocket.send_json({
                                "type": "session_ended",
                                "session_id": session.session_id,
                                "message": "Session ended by client"
                            })
                            streaming_active = False
                        
                        elif command.get("type") == "text_input":
                            # Process direct text input
                            text_input = command.get("text", "")
                            
                            if text_input:
                                # Add user turn
                                user_turn = ConversationTurn(
                                    speaker="user",
                                    input_modality=InputModality.TEXT,
                                    text=text_input,
                                    language=language
                                )
                                session.add_turn(user_turn)
                                
                                # Generate response
                                response_text = await self._generate_response(session)
                                
                                # Send the response
                                await websocket.send_json({
                                    "type": "agent_response",
                                    "text": response_text
                                })
                                
                                # Synthesize speech if needed
                                synthesis_result = await self.speech_engine.synthesize(
                                    text=response_text,
                                    language=language,
                                    voice_style=SynthesisVoiceStyle.NEUTRAL,
                                    output_format=AudioFormat.WAV
                                )
                                
                                response_audio = synthesis_result[0]
                                
                                # Stream the audio back in chunks
                                chunk_size = 4096  # Send in 4KB chunks
                                for i in range(0, len(response_audio), chunk_size):
                                    await websocket.send_bytes(response_audio[i:i+chunk_size])
                                
                                # Signal end of audio
                                await websocket.send_json({
                                    "type": "audio_complete"
                                })
                                
                                # Add the agent turn to the session
                                agent_turn = ConversationTurn(
                                    speaker="agent",
                                    input_modality=InputModality.TEXT,
                                    text=response_text,
                                    audio_data=response_audio,
                                    language=language
                                )
                                session.add_turn(agent_turn)
                        
                    except json.JSONDecodeError:
                        # Handle invalid JSON
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
                
                # Handle disconnection or close
                elif "type" in message and message["type"] == "websocket.disconnect":
                    streaming_active = False
        
        except WebSocketDisconnect:
            # Handle WebSocket disconnection
            self.logger.info(f"WebSocket disconnected for session {session.session_id}")
            streaming_active = False
        
        except Exception as e:
            # Handle any other exceptions
            self.logger.error(f"Error in streaming process: {str(e)}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
            except:
                pass  # Connection might be closed already
            streaming_active = False
        
        finally:
            # Clean up
            if streaming_active:
                try:
                    await websocket.close()
                except:
                    pass
    
    def _check_for_end_marker(self, text: str) -> bool:
        """
        Check if the text contains markers indicating the end of speech.
        
        Args:
            text: Text to check for end markers
            
        Returns:
            True if end markers are detected, False otherwise
        """
        # Common end of speech patterns
        end_markers = [
            ".", "?", "!", "thank you", "that's all", "that is all",
            "goodbye", "bye", "end", "stop", "finish", "done"
        ]
        
        # Advanced: also detect longer pauses as end markers
        # In a real implementation, this would be determined by silence detection in the audio
        
        # Check for any end markers
        text = text.lower().strip()
        for marker in end_markers:
            if marker in text or text.endswith(marker):
                return True
        
        # Default: no end marker detected
        return False
    
    # ----------------- Session Management Methods -----------------
    
    def _get_or_create_session(self, session_id: str = None) -> ConversationContext:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Session ID to retrieve, or None to create a new session
            
        Returns:
            ConversationContext for the session
        """
        if session_id and session_id in self.active_sessions:
            # Update last active time
            session = self.active_sessions[session_id]
            session.last_active = datetime.now()
            return session
        
        # Create a new session
        new_session = ConversationContext(
            session_id=session_id or str(uuid.uuid4()),
            language=self.config.default_language,
            agent_profile={
                "agent_id": self.config.agent_id,
                "name": self.config.name,
                "voice_model": self.config.default_voice_model
            }
        )
        
        # Store in active sessions
        self.active_sessions[new_session.session_id] = new_session
        self.logger.info(f"Created new session: {new_session.session_id}")
        
        return new_session
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get an existing session by ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            ConversationContext for the session or None if not found
        """
        if session_id in self.active_sessions:
            # Update last active time
            session = self.active_sessions[session_id]
            session.last_active = datetime.now()
            return session
        
        return None
    
    def end_session(self, session_id: str) -> bool:
        """
        End and remove a session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            True if session was found and removed, False otherwise
        """
        if session_id in self.active_sessions:
            # Remove session
            del self.active_sessions[session_id]
            self.logger.info(f"Ended session: {session_id}")
            return True
        
        return False
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired sessions.
        
        Args:
            max_age_hours: Maximum age in hours for sessions to keep
            
        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        expired_sessions = []
        
        # Find expired sessions
        for session_id, session in self.active_sessions.items():
            age = now - session.last_active
            if age.total_seconds() > max_age_hours * 3600:
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def _validate_session(self, session_id: str) -> bool:
        """
        Validate if a session exists and is active.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.active_sessions:
            return False
        
        # Check if session is expired (24 hours)
        session = self.active_sessions[session_id]
        age = datetime.now() - session.last_active
        if age.total_seconds() > 24 * 3600:
            # Remove expired session
            del self.active_sessions[session_id]
            return False
        
        return True
    
    # ----------------- Voice Model Management Methods -----------------
    
    async def create_voice_model(self, name: str, audio_data: Union[bytes, BinaryIO],
                               description: str = None, gender: VoiceGender = VoiceGender.NEUTRAL,
                               age: VoiceAge = VoiceAge.ADULT,
                               personality: VoicePersonality = VoicePersonality.NEUTRAL,
                               accent: AccentType = AccentType.NEUTRAL) -> VoiceModel:
        """
        Create a voice model from audio data.
        
        Args:
            name: User-friendly name for the model
            audio_data: Audio data for voice analysis
            description: Optional description
            gender: Voice gender
            age: Voice age category
            personality: Voice personality
            accent: Voice accent
            
        Returns:
            Created voice model
        """
        # Check if VoiceCloning capability is enabled
        if VoiceAgentCapability.VOICE_CLONING not in self.config.enabled_capabilities:
            raise ValueError("Voice cloning capability is not enabled for this agent")
        
        try:
            # Create a unique ID for the model
            model_id = str(uuid.uuid4())
            
            # Create a new voice model with voice cloner
            voice_model = await self.voice_cloner.create_voice_model(
                audio_data=audio_data,
                model_id=model_id,
                name=name,
                description=description,
                gender=gender,
                age=age,
                personality=personality,
                accent=accent
            )
            
            # Store the model
            self.voice_models[model_id] = voice_model
            self.logger.info(f"Created new voice model: {model_id} ({name})")
            
            return voice_model
        
        except Exception as e:
            self.logger.error(f"Error creating voice model: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create voice model: {str(e)}")
    
    async def update_voice_model(self, model_id: str, name: str = None, 
                               description: str = None, audio_data: Union[bytes, BinaryIO] = None,
                               gender: VoiceGender = None,
                               age: VoiceAge = None,
                               personality: VoicePersonality = None,
                               accent: AccentType = None) -> VoiceModel:
        """
        Update an existing voice model.
        
        Args:
            model_id: ID of the model to update
            name: New name (optional)
            description: New description (optional)
            audio_data: New audio data (optional)
            gender: New gender (optional)
            age: New age category (optional)
            personality: New personality (optional)
            accent: New accent (optional)
            
        Returns:
            Updated voice model
        """
        # Check if VoiceCloning capability is enabled
        if VoiceAgentCapability.VOICE_CLONING not in self.config.enabled_capabilities:
            raise ValueError("Voice cloning capability is not enabled for this agent")
        
        # Check if model exists
        if model_id not in self.voice_models:
            raise ValueError(f"Voice model {model_id} not found")
        
        try:
            # Get existing model
            voice_model = self.voice_models[model_id]
            
            # Update model with voice cloner
            updated_model = await self.voice_cloner.update_voice_model(
                model_id=model_id,
                name=name,
                description=description,
                audio_data=audio_data,
                gender=gender,
                age=age,
                personality=personality,
                accent=accent
            )
            
            # Store updated model
            self.voice_models[model_id] = updated_model
            self.logger.info(f"Updated voice model: {model_id}")
            
            return updated_model
        
        except Exception as e:
            self.logger.error(f"Error updating voice model: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to update voice model: {str(e)}")
    
    def delete_voice_model(self, model_id: str) -> bool:
        """
        Delete a voice model.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if model was deleted, False if not found
        """
        if model_id not in self.voice_models:
            return False
        
        # Remove model
        del self.voice_models[model_id]
        self.logger.info(f"Deleted voice model: {model_id}")
        
        return True
    
    def list_voice_models(self) -> List[VoiceModel]:
        """
        List all available voice models.
        
        Returns:
            List of voice models
        """
        return list(self.voice_models.values())
    
    def _load_voice_model(self, model_id: str) -> Optional[VoiceModel]:
        """
        Load a voice model by ID.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded voice model or None if not found
        """
        # Check if model is already loaded
        if model_id in self.voice_models:
            return self.voice_models[model_id]
        
        try:
            # In a real implementation, this would load the model from storage
            # For this demo, return None if not found in memory
            self.logger.warning(f"Voice model {model_id} not found in memory")
            return None
        
        except Exception as e:
            self.logger.error(f"Error loading voice model: {str(e)}", exc_info=True)
            return None
