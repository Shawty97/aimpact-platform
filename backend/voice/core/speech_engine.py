"""
AImpact Advanced Speech Engine

This module provides a high-performance, multi-provider speech recognition and synthesis engine
with advanced features like streaming, real-time processing, automatic language detection,
noise reduction, and dynamic provider selection.

The engine supports multiple providers (Whisper, Google, Azure) with automatic fallback
and performance-based routing to ensure optimal quality and reliability.
"""

import os
import time
import asyncio
import logging
import json
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, BinaryIO, Callable, Set, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import lru_cache
import tempfile
import io
import wave
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
import speech_recognition as sr
from fastapi import WebSocket
from langchain.cache import InMemoryCache

# Type aliases for improved readability
AudioData = Union[bytes, BinaryIO, np.ndarray]
ProviderResult = Dict[str, Any]
T = TypeVar('T')

# Configure logging
logger = logging.getLogger("aimpact.voice.speech_engine")

# ----------------- Enums and Models -----------------

class SpeechProvider(str, Enum):
    """Supported speech recognition and synthesis providers."""
    WHISPER = "whisper"
    GOOGLE = "google"
    AZURE = "azure"
    AMAZON = "amazon"
    LOCAL = "local"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    PCM = "pcm"
    WEBM = "webm"


class RecognitionMode(str, Enum):
    """Speech recognition processing modes."""
    STANDARD = "standard"      # Standard processing with all features
    REALTIME = "realtime"      # Optimized for real-time low-latency
    HIGH_ACCURACY = "high_accuracy"  # Optimized for maximum accuracy
    LOW_RESOURCE = "low_resource"    # Optimized for low resource usage


class SynthesisVoiceStyle(str, Enum):
    """Voice styles for text-to-speech synthesis."""
    NEUTRAL = "neutral"
    FORMAL = "formal"
    CASUAL = "casual"
    CHEERFUL = "cheerful"
    EMPATHETIC = "empathetic"
    ANGRY = "angry"
    SAD = "sad"
    EXCITED = "excited"
    WHISPERED = "whispered"
    PROFESSIONAL = "professional"


class RecognitionQuality(str, Enum):
    """Quality settings for speech recognition."""
    LOW = "low"          # Faster but less accurate
    MEDIUM = "medium"    # Balanced performance
    HIGH = "high"        # Highest accuracy, slower
    ADAPTIVE = "adaptive"  # Dynamically adjusts based on content


class LanguageConfig(BaseModel):
    """Configuration for language processing."""
    language_code: str = Field(..., description="BCP-47 language code (e.g., 'en-US')")
    dialect: Optional[str] = Field(None, description="Specific dialect if applicable")
    accent: Optional[str] = Field(None, description="Specific accent if applicable")
    auto_detect: bool = Field(False, description="Automatically detect language")
    confidence_threshold: float = Field(0.7, description="Minimum confidence for language detection")


@dataclass
class SpeechMetrics:
    """Metrics for speech processing performance and quality."""
    provider: SpeechProvider
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    latency: Optional[float] = None
    confidence: Optional[float] = None
    word_count: Optional[int] = None
    error_count: Optional[int] = None
    quality_score: Optional[float] = None
    cost: Optional[float] = None
    token_count: Optional[int] = None
    audio_duration: Optional[float] = None
    
    def complete(self, confidence: float = None, word_count: int = None, 
                error_count: int = 0, quality_score: float = None,
                cost: float = 0.0, token_count: int = None,
                audio_duration: float = None):
        """Complete the metrics with results."""
        self.end_time = time.time()
        self.latency = self.end_time - self.start_time
        self.confidence = confidence
        self.word_count = word_count
        self.error_count = error_count
        self.quality_score = quality_score
        self.cost = cost
        self.token_count = token_count
        self.audio_duration = audio_duration
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "provider": self.provider.value,
            "latency": self.latency,
            "confidence": self.confidence,
            "word_count": self.word_count,
            "error_count": self.error_count,
            "quality_score": self.quality_score,
            "cost": self.cost,
            "token_count": self.token_count,
            "audio_duration": self.audio_duration,
            "processing_time": self.latency
        }


class ProviderConfig(BaseModel):
    """Configuration for a speech provider."""
    enabled: bool = Field(True, description="Whether this provider is enabled")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    region: Optional[str] = Field(None, description="Region for the provider if applicable")
    endpoint: Optional[str] = Field(None, description="Custom endpoint URL if applicable")
    timeout: float = Field(30.0, description="Timeout in seconds for API calls")
    retry_count: int = Field(3, description="Number of retries for failed API calls")
    model: Optional[str] = Field(None, description="Specific model to use")
    priority: int = Field(1, description="Priority (lower is higher priority)")
    cost_per_minute: float = Field(0.0, description="Cost per minute of audio")
    supports_streaming: bool = Field(False, description="Whether provider supports streaming")
    supports_realtime: bool = Field(False, description="Whether provider supports real-time processing")
    supported_languages: List[str] = Field(default_factory=list, description="Supported language codes")
    supported_features: List[str] = Field(default_factory=list, description="Supported features")


class SpeechEngineConfig(BaseModel):
    """Configuration for the speech engine."""
    providers: Dict[SpeechProvider, ProviderConfig] = Field(
        default_factory=dict, 
        description="Configuration for each provider"
    )
    default_provider: SpeechProvider = Field(
        SpeechProvider.WHISPER, 
        description="Default provider to use"
    )
    fallback_providers: List[SpeechProvider] = Field(
        default_factory=list,
        description="Providers to try if the primary provider fails, in order"
    )
    auto_select_provider: bool = Field(
        True, 
        description="Automatically select the best provider based on the request"
    )
    cache_enabled: bool = Field(
        True, 
        description="Enable caching of results"
    )
    cache_ttl: int = Field(
        3600, 
        description="Time-to-live for cached results in seconds"
    )
    noise_reduction_enabled: bool = Field(
        True, 
        description="Enable noise reduction and audio enhancement"
    )
    collect_metrics: bool = Field(
        True, 
        description="Collect and store performance metrics"
    )
    log_level: str = Field(
        "INFO", 
        description="Logging level"
    )


# ----------------- Provider Abstract Base Classes -----------------

class SpeechRecognitionProvider(ABC):
    """Base class for speech recognition providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = "base_provider"
        self._initialize()
        
    def _initialize(self):
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    async def recognize(self, audio_data: AudioData, language: str = "en-US", 
                       mode: RecognitionMode = RecognitionMode.STANDARD,
                       quality: RecognitionQuality = RecognitionQuality.MEDIUM) -> ProviderResult:
        """Recognize speech in audio data and return text and metadata."""
        pass
    
    @abstractmethod
    async def recognize_stream(self, audio_stream, language: str = "en-US",
                              callback: Callable[[str, bool], None] = None) -> ProviderResult:
        """Process streaming audio and return results through callback."""
        pass
    
    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Check if the provider supports the given language."""
        pass
    
    @abstractmethod
    def get_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass
    
    def prepare_audio(self, audio_data: AudioData) -> Tuple[AudioData, AudioFormat]:
        """Prepare audio data for the provider, converting if necessary."""
        # Default implementation, override as needed
        return audio_data, AudioFormat.WAV
    
    def cleanup(self):
        """Clean up any resources used by the provider."""
        pass
    
    def estimate_cost(self, audio_duration: float) -> float:
        """Estimate the cost of processing audio of the given duration."""
        return self.config.cost_per_minute * (audio_duration / 60.0)


class SpeechSynthesisProvider(ABC):
    """Base class for speech synthesis providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = "base_synthesis_provider"
        self._initialize()
        
    def _initialize(self):
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, language: str = "en-US", 
                        voice_style: SynthesisVoiceStyle = SynthesisVoiceStyle.NEUTRAL,
                        output_format: AudioFormat = AudioFormat.WAV) -> Tuple[bytes, ProviderResult]:
        """Synthesize speech from text and return audio data and metadata."""
        pass
    
    @abstractmethod
    async def synthesize_stream(self, text: str, language: str = "en-US",
                               voice_style: SynthesisVoiceStyle = SynthesisVoiceStyle.NEUTRAL,
                               callback: Callable[[bytes, bool], None] = None) -> ProviderResult:
        """Synthesize speech and stream the audio chunks through callback."""
        pass
    
    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Check if the provider supports the given language."""
        pass
    
    @abstractmethod
    def get_voices(self, language_code: str = None) -> List[Dict[str, Any]]:
        """Get list of available voices, optionally filtered by language."""
        pass
    
    def cleanup(self):
        """Clean up any resources used by the provider."""
        pass
    
    def estimate_cost(self, text_length: int) -> float:
        """Estimate the cost of synthesizing text of the given length."""
        # Default implementation estimates based on character count
        # Providers can override with more accurate cost models
        words = text_length / 5  # Rough estimate of words
        minutes = words / 150  # Rough estimate of speaking rate (words per minute)
        return self.config.cost_per_minute * minutes


# ----------------- Provider Implementations -----------------

class WhisperRecognitionProvider(SpeechRecognitionProvider):
    """OpenAI Whisper speech recognition provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.name = "whisper"
        
    def _initialize(self):
        """Initialize Whisper with API key and configuration."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key)
            self.model = self.config.model or "whisper-1"
            logger.info(f"Initialized Whisper provider with model {self.model}")
        except ImportError:
            logger.error("OpenAI package not installed. Please install with 'pip install openai'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Whisper provider: {str(e)}")
            raise
    
    async def recognize(self, audio_data: AudioData, language: str = "en-US", 
                       mode: RecognitionMode = RecognitionMode.STANDARD,
                       quality: RecognitionQuality = RecognitionQuality.MEDIUM) -> ProviderResult:
        """Recognize speech using OpenAI Whisper API."""
        metrics = SpeechMetrics(provider=SpeechProvider.WHISPER)
        
        try:
            # Handle different types of audio data
            if isinstance(audio_data, np.ndarray):
                # Convert numpy array to bytes
                with io.BytesIO() as buf:
                    # Normalize to 16-bit PCM
                    audio_data = (audio_data * 32767).astype(np.int16)
                    # Write as WAV
                    with wave.open(buf, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_data.tobytes())
                    file_data = buf.getvalue()
                    
                    # Save to temporary file for OpenAI API
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(file_data)
                        temp_file_path = temp_file.name
            elif isinstance(audio_data, bytes):
                # Save bytes directly to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
            elif hasattr(audio_data, 'read'):
                # If it's a file-like object, read it and save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data.read())
                    temp_file_path = temp_file.name
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            # Set quality based on mode and quality settings
            if mode == RecognitionMode.HIGH_ACCURACY or quality == RecognitionQuality.HIGH:
                temperature = 0.0  # Lower temperature for higher accuracy
                model = self.model
            elif mode == RecognitionMode.REALTIME:
                temperature = 0.5  # Balance between speed and accuracy
                model = "whisper-1"  # Use fastest model for real-time
            elif mode == RecognitionMode.LOW_RESOURCE:
                temperature = 0.7  # Faster processing
                model = "whisper-1"  # Use fastest model for low resource
            else:  # STANDARD or default
                temperature = 0.3
                model = self.model
            
            # Detect language if not specified
            language_param = None if language.lower() == "auto" else language
            
            # Process with OpenAI Whisper API
            with open(temp_file_path, "rb") as audio_file:
                # Call the API with appropriate parameters
                response = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    file=audio_file,
                    model=model,
                    language=language_param,
                    temperature=temperature,
                    response_format="verbose_json"
                )
            
            # Extract results
            if hasattr(response, 'text'):
                text = response.text
                language_code = getattr(response, 'language', language)
                confidence = getattr(response, 'confidence', 0.9)  # Default if not provided
                segments = getattr(response, 'segments', [])
            else:
                # Handle dictionary-like response
                response_dict = response
                text = response_dict.get('text', '')
                language_code = response_dict.get('language', language)
                confidence = response_dict.get('confidence', 0.9)
                segments = response_dict.get('segments', [])
            
            # Calculate word count
            word_count = len(text.split())
            
            # Get audio duration if available
            audio_duration = 0.0
            if segments and len(segments) > 0:
                last_segment = segments[-1]
                if isinstance(last_segment, dict) and 'end' in last_segment:
                    audio_duration = last_segment['end']
                elif hasattr(last_segment, 'end'):
                    audio_duration = last_segment.end
            
            # Complete metrics
            metrics.complete(
                confidence=confidence,
                word_count=word_count,
                quality_score=confidence,
                audio_duration=audio_duration,
                cost=self.estimate_cost(audio_duration)
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            # Return structured result
            return {
                "text": text,
                "confidence": confidence,
                "language": language_code,
                "word_count": word_count,
                "audio_duration": audio_duration,
                "metrics": metrics.to_dict(),
                "segments": segments
            }
        
        except Exception as e:
            logger.error(f"Error in Whisper recognition: {str(e)}")
            # Clean up temp file in case of error
            try:
                os.unlink(temp_file_path)
            except:
                pass
            raise
    
    async def recognize_stream(self, audio_stream, language: str = "en-US",
                              callback: Callable[[str, bool], None] = None) -> ProviderResult:
        """Process streaming audio using Whisper."""
        # Whisper doesn't natively support streaming, so we'll implement chunking
        metrics = SpeechMetrics(provider=SpeechProvider.WHISPER)
        buffer = io.BytesIO()
        final_text = ""
        chunk_duration = 3.0  # Process in 3-second chunks
        chunk_size = int(16000 * 2 * chunk_duration)  # assuming 16kHz, 16-bit audio
        last_chunk_time = time.time()
        final_result = {}
        
        try:
            # Process audio chunks as they come in
            async for chunk in audio_stream:
                buffer.write(chunk)
                current_time = time.time()
                
                # Process when buffer reaches threshold or enough time has passed
                if buffer.tell() >= chunk_size or (current_time - last_chunk_time) > chunk_duration:
                    buffer.seek(0)
                    chunk_audio = buffer.read()
                    
                    # Process this chunk
                    result = await self.recognize(chunk_audio, language, 
                                                RecognitionMode.REALTIME, 
                                                RecognitionQuality.LOW)
                    
                    # Update final text
                    if result["text"]:
                        partial_text = result["text"]
                        final_text += " " + partial_text
                        final_text = final_text.strip()
                        
                        # Send partial result through callback if provided
                        if callback:
                            await callback(partial_text, False)
                    
                    # Clear buffer for next chunk
                    buffer = io.BytesIO()
                    last_chunk_time = current_time
                    
                    # Keep track of the latest result for metadata
                    final_result = result
            
            # Process any remaining audio
            if buffer.tell() > 0:
                buffer.seek(0)
                chunk_audio = buffer.read()
                
                result = await self.recognize(chunk_audio, language, 
                                            RecognitionMode.STANDARD, 
                                            RecognitionQuality.MEDIUM)
                
                if result["text"]:
                    final_text += " " + result["text"]
                    final_text = final_text.strip()
                    
                    # Update final result
                    final_result = result
            
            # Send final result through callback
            if callback:
                await callback(final_text, True)
            
            # Create final combined result
            combined_result = {
                "text": final_text,
                "confidence": final_result.get("confidence", 0.8),
                "language": final_result.get("language", language),
                "is_final": True,
                "metrics": metrics.to_dict()
            }
            
            return combined_result
        
        except Exception as e:
            logger.error(f"Error in Whisper streaming recognition: {str(e)}")
            if callback:
                await callback(f"Error: {str(e)}", True)
            raise
    
    def supports_language(self, language_code: str) -> bool:
        """Check if Whisper supports the specified language."""
        # Whisper supports 100+ languages, but check if it's in the supported list
        supported_languages = self.get_languages()
        return language_code in supported_languages or language_code.split('-')[0] in supported_languages
    
    def get_languages(self) -> List[str]:
        """Get list of languages supported by Whisper."""
        # Return list of BCP-47 language codes supported by Whisper
        # This is a partial list; Whisper supports many more languages
        return [
            "en", "en-US", "en-GB", "es", "es-ES", "fr", "fr-FR", "de", "de-DE",
            "it", "it-IT", "pt", "pt-BR", "nl", "nl-NL", "ja", "ja-JP",
            "ko", "ko-KR", "zh", "zh-CN", "zh-TW", "ar", "ru", "ru-RU",
            "hi", "hi-IN", "tr", "tr-TR", "pl", "pl-PL", "vi", "vi-VN",
            "sv", "sv-SE", "fi", "fi-FI", "no", "no-NO", "da", "da-DK",
            "cs", "cs-CZ", "hu", "hu-HU", "el", "el-GR", "ro", "ro-RO",
            "id", "id-ID", "ms", "ms-MY", "th", "th-TH", "he", "he-IL",
            "bg", "bg-BG", "uk", "uk-UA", "hr", "hr-HR", "sk", "sk-SK",
            "lt", "lt-LT", "lv", "lv-LV", "et", "et-EE", "ca", "ca-ES"
        ]


class GoogleRecognitionProvider(SpeechRecognitionProvider):
    """Google Cloud Speech-to-Text provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.name = "google"
        
    def _initialize(self):
        """Initialize Google Speech client with API key and configuration."""
        try:
            # In a production system, we would use the Google Cloud client library
            # For this example, we'll use a simulated implementation
            self.api_key = self.config.api_key
            logger.info("Initialized Google Speech-to-Text provider")
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {str(e)}")
            raise
    
    async def recognize(self, audio_data: AudioData, language: str = "en-US", 
                       mode: RecognitionMode = RecognitionMode.STANDARD,
                       quality: RecognitionQuality = RecognitionQuality.MEDIUM) -> ProviderResult:
        """Recognize speech using Google Cloud Speech-to-Text."""
        metrics = SpeechMetrics(provider=SpeechProvider.GOOGLE)
        
        try:
            # In a production implementation, we would use the Google Cloud SDK
            # For this example, we'll use the speech_recognition library
            recognizer = sr.Recognizer()
            
            # Prepare audio data for Google
            audio_file = sr.AudioFile(io.BytesIO(audio_data))
            with audio_file as source:
                audio = recognizer.record(source)
            
            # Set recognition parameters based on mode and quality
            show_all = (mode == RecognitionMode.HIGH_ACCURACY)
            
            # Perform recognition
            result = await asyncio.to_thread(
                recognizer.recognize_google, 
                audio, 
                language=language,
                show_all=show_all
            )
            
            # Parse result
            if show_all and isinstance(result, dict):
                # Parse detailed response
                text = result.get("alternative", [{}])[0].get("transcript", "")
                confidence = result.get("alternative", [{}])[0].get("confidence", 0.9)
            else:
                # Simple text result
                text = result
                confidence = 0.9  # Default confidence
            
            # Calculate word count
            word_count = len(text.split())
            
            # Get audio duration (estimate from audio data)
            audio_duration = len(audio_data) / (16000 * 2)  # Assuming 16kHz, 16-bit audio
            
            # Complete metrics
            metrics.complete(
                confidence=confidence,
                word_count=word_count,
                quality_score=confidence,
                audio_duration=audio_duration,
                cost=self.estimate_cost(audio_duration)
            )
            
            # Return structured result
            return {
                "text": text,
                "confidence": confidence,
                "language": language,
                "word_count": word_count,
                "audio_duration": audio_duration,
                "metrics": metrics.to_dict()
            }
        
        except Exception as e:
            logger.error(f"Error in Google recognition: {str(e)}")
            raise
    
    async def recognize_stream(self, audio_stream, language: str = "en-US",
                              callback: Callable[[str, bool], None] = None) -> ProviderResult:
        """Process streaming audio using Google Cloud Speech-to-Text."""
        # Implement streaming recognition using Google's streaming API
        # In a real implementation, we would use the Google Cloud streaming API
        # For this example, we'll buffer chunks and process them
        
        # Similar implementation to WhisperRecognitionProvider's streaming method
        metrics = SpeechMetrics(provider=SpeechProvider.GOOGLE)
        buffer = io.BytesIO()
        final_text = ""
        
        # Process streaming chunks similar to the Whisper implementation
        # but using Google's API
        # For brevity, this implementation is simplified
        
        return {
            "text": "Google streaming transcription",
            "confidence": 0.9,
            "language": language,
            "is_final": True,
            "metrics": metrics.to_dict()
        }
    
    def supports_language(self, language_code: str) -> bool:
        """Check if Google supports the specified language."""
        supported_languages = self.get_languages()
        return language_code in supported_languages
    
    def get_languages(self) -> List[str]:
        """Get list of languages supported by Google Speech-to-Text."""
        # Return a subset of supported languages
        return [
            "en-US", "en-GB", "es-ES", "es-MX", "fr-FR", "fr-CA", "de-DE", 
            "it-IT", "pt-BR", "pt-PT", "nl-NL", "ja-JP", "ko-KR", "zh-CN", 
            "zh-TW", "ru-RU", "hi-IN", "ar-SA", "pl-PL"
        ]


# ----------------- Main Speech Engine Class -----------------

class SpeechEngine:
    """
    Advanced speech processing engine with multi-provider support, fallback mechanisms,
    automatic language detection, and dynamic provider selection.
    """
    
    def __init__(self, config: SpeechEngineConfig = None):
        """Initialize the speech engine with configuration."""
        self.config = config or SpeechEngineConfig()
        self.recognition_providers: Dict[SpeechProvider, SpeechRecognitionProvider] = {}
        self.synthesis_providers: Dict[SpeechProvider, SpeechSynthesisProvider] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.cache = InMemoryCache(ttl_seconds=self.config.cache_ttl if self.config.cache_enabled else 0)
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured speech providers."""
        # Initialize recognition providers
        for provider_type, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
                
            try:
                if provider_type == SpeechProvider.WHISPER:
                    self.recognition_providers[provider_type] = WhisperRecognitionProvider(provider_config)
                elif provider_type == SpeechProvider.GOOGLE:
                    self.recognition_providers[provider_type] = GoogleRecognitionProvider(provider_config)
                # Add more providers as implemented

