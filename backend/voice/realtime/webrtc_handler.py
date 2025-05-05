"""
WebRTC Handler for Real-time Voice Processing

Provides WebRTC support for high-quality, low-latency voice communication
with advanced features like echo cancellation, noise suppression, and
automatic gain control.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("aimpact.voice.realtime.webrtc")

class WebRTCConfig(BaseModel):
    """Configuration for WebRTC connections."""
    ice_servers: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"}
        ],
        description="ICE servers for WebRTC connection establishment"
    )
    enable_echo_cancellation: bool = Field(
        True, 
        description="Enable acoustic echo cancellation"
    )
    enable_noise_suppression: bool = Field(
        True, 
        description="Enable noise suppression"
    )
    enable_auto_gain_control: bool = Field(
        True, 
        description="Enable automatic gain control"
    )
    audio_bandwidth: str = Field(
        "high", 
        description="Audio bandwidth preference: 'narrow', 'medium', or 'high'"
    )

class WebRTCConnection(BaseModel):
    """Information about a WebRTC connection."""
    connection_id: str
    peer_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebRTCSignalMessage(BaseModel):
    """WebRTC signaling message."""
    type: str
    data: Dict[str, Any]
    peer_id: str

class WebRTCHandler:
    """Handler for WebRTC connections and real-time voice processing."""
    
    def __init__(self, config: Optional[WebRTCConfig] = None):
        """Initialize the WebRTC handler."""
        self.config = config or WebRTCConfig()
        self.connections: Dict[str, WebRTCConnection] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, Callable] = {}
        
    async def handle_connection(self, websocket: WebSocket, client_id: str = None):
        """
        Handle a new WebRTC connection.
        
        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID, generated if not provided
        """
        if not client_id:
            client_id = str(uuid.uuid4())
            
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Create a new connection
        connection = WebRTCConnection(
            connection_id=str(uuid.uuid4()),
            peer_id=client_id
        )
        
        # Store the connection and websocket
        self.connections[connection.connection_id] = connection
        self.websockets[connection.connection_id] = websocket
        
        # Send WebRTC configuration
        await websocket.send_json({
            "type": "config",
            "data": {
                "ice_servers": self.config.ice_servers,
                "echo_cancellation": self.config.enable_echo_cancellation,
                "noise_suppression": self.config.enable_noise_suppression,
                "auto_gain_control": self.config.enable_auto_gain_control,
                "audio_bandwidth": self.config.audio_bandwidth
            }
        })
        
        try:
            # Handle incoming messages
            while True:
                # Receive a message
                message = await websocket.receive_json()
                
                # Update last activity
                connection.last_activity = datetime.now()
                
                # Handle message based on type
                if message.get("type") == "offer":
                    # Handle SDP offer
                    await self._handle_offer(connection, message)
                    
                elif message.get("type") == "answer":
                    # Handle SDP answer
                    await self._handle_answer(connection, message)
                    
                elif message.get("type") == "ice_candidate":
                    # Handle ICE candidate
                    await self._handle_ice_candidate(connection, message)
                    
                elif message.get("type") == "audio_data":
                    # Handle audio data
                    await self._handle_audio_data(connection, message)
                    
                elif message.get("type") == "disconnect":
                    # Handle disconnect request
                    await self._handle_disconnect(connection)
                    break
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error in WebRTC connection: {str(e)}")
            
        finally:
            # Clean up connection
            self._cleanup_connection(connection.connection_id)
    
    async def _handle_offer(self, connection: WebRTCConnection, message: Dict[str, Any]):
        """
        Handle an SDP offer.
        
        Args:
            connection: The WebRTC connection
            message: The SDP offer message
        """
        logger.debug(f"Received SDP offer from {connection.peer_id}")
        
        # Get the websocket
        websocket = self.websockets.get(connection.connection_id)
        if not websocket:
            logger.error(f"WebSocket not found for connection {connection.connection_id}")
            return
        
        # Just echo the offer for now (in a real implementation, this would process the offer)
        await websocket.send_json({
            "type": "offer_received",
            "data": {"sdp": message.get("data", {}).get("sdp")}
        })
    
    async def _handle_answer(self, connection: WebRTCConnection, message: Dict[str, Any]):
        """
        Handle an SDP answer.
        
        Args:
            connection: The WebRTC connection
            message: The SDP answer message
        """
        logger.debug(f"Received SDP answer from {connection.peer_id}")
        
        # Get the websocket
        websocket = self.websockets.get(connection.connection_id)
        if not websocket:
            logger.error(f"WebSocket not found for connection {connection.connection_id}")
            return
        
        # Just echo the answer for now
        await websocket.send_json({
            "type": "answer_received",
            "data": {"sdp": message.get("data", {}).get("sdp")}
        })
    
    async def _handle_ice_candidate(self, connection: WebRTCConnection, message: Dict[str, Any]):
        """
        Handle an ICE candidate.
        
        Args:
            connection: The WebRTC connection
            message: The ICE candidate message
        """
        logger.debug(f"Received ICE candidate from {connection.peer_id}")
        
        # Get the websocket
        websocket = self.websockets.get(connection.connection_id)
        if not websocket:
            logger.error(f"WebSocket not found for connection {connection.connection_id}")
            return
        
        # Just echo the ICE candidate for now
        await websocket.send_json({
            "type": "ice_candidate_received",
            "data": {"candidate": message.get("data", {}).get("candidate")}
        })
    
    async def _handle_audio_data(self, connection: WebRTCConnection, message: Dict[str, Any]):
        """
        Handle incoming audio data.
        
        Args:
            connection: The WebRTC connection
            message: The audio data message
        """
        logger.debug(f"Received audio data from {connection.peer_id}")
        
        # Get the websocket
        websocket = self.websockets.get(connection.connection_id)
        if not websocket:
            logger.error(f"WebSocket not found for connection {connection.connection_id}")
            return
        
        # Get the audio processor for this connection
        processor = self.audio_processors.get(connection.connection_id)
        
        # Process the audio data if a processor is registered
        if processor:
            try:
                # Get the audio data
                audio_data = message.get("data", {}).get("audio")
                
                # Process the audio data
                processed_data = await processor(audio_data)
                
                # Send the processed data back
                await websocket.send_json({
                    "type": "processed_audio",
                    "data": {"audio": processed_data}
                })
                
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}")
                
                # Send an error message
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Audio processing error: {str(e)}"}
                })
    
    async def _handle_disconnect(self, connection: WebRTCConnection):
        """
        Handle a disconnect request.
        
        Args:
            connection: The WebRTC connection
        """
        logger.info(f"Received disconnect request from {connection.peer_id}")
        
        # Get the websocket
        websocket = self.websockets.get(connection.connection_id)
        if not websocket:
            logger.error(f"WebSocket not found for connection {connection.connection_id}")
            return
        
        # Send a disconnect acknowledgement
        await websocket.send_json({
            "type": "disconnect_ack",
            "data": {"message": "Disconnected successfully"}
        })
        
        # Clean up the connection
        self._cleanup_connection(connection.connection_id)
    
    def _cleanup_connection(self, connection_id: str):
        """
        Clean up a WebRTC connection.
        
        Args:
            connection_id: The ID of the connection to clean up
        """
        # Remove the connection
        self.connections.pop(connection_id, None)
        
        # Remove the websocket
        self.websockets.pop(connection_id, None)
        
        # Remove the audio processor
        self.audio_processors.pop(connection_id, None)
        
        logger.info(f"Cleaned up connection {connection_id}")
    
    def register_audio_processor(self, connection_id: str, processor: Callable):
        """
        Register an audio processor for a connection.
        
        Args:
            connection_id: The ID of the connection
            processor: A callable that processes audio data
        """
        self.audio_processors[connection_id] = processor
        logger.info(f"Registered audio processor for connection {connection_id}")

