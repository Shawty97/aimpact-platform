"""
Optimization Service for AImpact platform.

This module provides agent optimization capabilities using PPO
(Proximal Policy Optimization) for self-learning based on feedback.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np

from ..memory.service import MemoryService
from ..memory.models import Memory, MemoryType, MemoryMetadata
from .models import (
    AgentFeedback, FeedbackType, FeedbackValue, OptimizationJob,
    OptimizationStatus, OptimizationTarget, AgentModelVersion,
    TrainingConfig
)

logger = logging.getLogger(__name__)


class OptimizerService:
    """
    Service for optimizing agents using PPO based on feedback.
    
    This service:
    - Collects and processes feedback
    - Trains agents using PPO
    - Manages model versions
    - Provides optimization APIs
    """
    
    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        model_storage_path: str = "/data/models",
        enable_background_processing: bool = True,
        background_check_interval: int = 300,  # 5 minutes
        default_config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize the optimizer service.
        
        Args:
            memory_service: Memory service for storing optimization data
            model_storage_path: Path to store model versions
            enable_background_processing: Whether to enable background job processing
            background_check_interval: Interval to check for pending jobs (seconds)
            default_config: Default training configuration
        """
        self.memory_service = memory_service
        self.model_storage_path = model_storage_path
        self.enable_background_processing = enable_background_processing
        self.background_check_interval = background_check_interval
        self.default_config = default_config or TrainingConfig()
        
        # In-memory storage (replace with database in production)
        self._feedback_store: Dict[str, AgentFeedback] = {}
        self._job_store: Dict[str, OptimizationJob] = {}
        self._model_version_store: Dict[str, AgentModelVersion] = {}
        
        # Active background tasks
        self._background_tasks = {}
        
        # Start background processor if enabled
        if enable_background_processing:
            self._background_task = asyncio.create_task(self._background_processor())
        
    async def _background_processor(self) -> None:
        """Background task to process pending optimization jobs."""
        while True:
            try:
                # Find pending jobs
                pending_jobs = [
                    job for job in self._job_store.values()
                    if job.status == OptimizationStatus.PENDING
                ]
                
                # Process up to 3 jobs concurrently
                if pending_jobs:
                    # Sort by creation time (oldest first)
                    pending_jobs.sort(key=lambda j: j.created_at)
                    
                    # Process up to 3 jobs
                    for job in pending_jobs[:3]:
                        if job.id not in self._background_tasks:
                            task = asyncio.create_task(self._process_job(job))
                            self._background_tasks[job.id] = task
                
                # Clean up completed tasks
                completed_jobs = []
                for job_id, task in list(self._background_tasks.items()):
                    if task.done():
                        completed_jobs.append(job_id)
                        
                        # Handle exceptions
                        if task.exception():
                            logger.error(f"Error in job {job_id}: {task.exception()}")
                
                for job_id in completed_jobs:
                    self._background_tasks.pop(job_id, None)
                    
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
            
            # Sleep before next check
            await asyncio.sleep(self.background_check_interval)
    
    async def save_feedback(self, feedback: AgentFeedback) -> AgentFeedback:
        """
        Save feedback for an agent.
        
        Args:
            feedback: Feedback to save
            
        Returns:
            The saved feedback
        """
        # Generate ID if not provided
        if not feedback.id:
            feedback.id = str(uuid.uuid4())
        
        # Save to memory service if available
        if self.memory_service:
            metadata = MemoryMetadata(
                session_id=feedback.session_id,
                user_id=feedback.user_id,
                tags=["feedback", feedback.feedback_type],
                source="user_feedback",
                importance=(3 if feedback.feedback_type in [FeedbackType.HELPFULNESS, FeedbackType.CORRECTNESS] else 2),
                custom_metadata={
                    "feedback_id": feedback.id,
                    "interaction_id": feedback.interaction_id
                }
            )
            
            content = {
                "feedback_type": feedback.feedback_type,
                "value": feedback.value.dict(),
                "context": feedback.context
            }
            
            await self.memory_service.save_memory(
                agent_id=feedback.agent_id,
                memory_type=MemoryType.FEEDBACK,
                content=content,
                metadata=metadata
            )
        
        # Store in memory
        self._feedback_store[feedback.id] = feedback
        
        # Log feedback
        logger.info(f"Saved feedback {feedback.id} for agent {feedback.agent_id}")
        
        return feedback
    
    async def get_feedback(self, feedback_id: str) -> Optional[AgentFeedback]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: ID of the feedback to retrieve
            
        Returns:
            The feedback if found, None otherwise
        """
        # Check in-memory store
        if feedback_id in self._feedback_store:
            return self._feedback_store[feedback_id]
        
        # If using memory service, try to retrieve from there
        if self.memory_service:
            # Search for the feedback memory
            query = {
                "agent_id": "*",  # Any agent
                "query": f"feedback_id:{feedback_id}",
                "memory_types": [MemoryType.FEEDBACK],
                "tags": ["feedback"],
                "min_score": 0.0  # Exact match
            }
            
            results = await self.memory_service.search_memories(query)
            if results and len(results) > 0:
                memory = results[0].memory
                content = memory.content
                
                # Reconstruct feedback object
                feedback = AgentFeedback(
                    id=feedback_id,
                    agent_id=memory.agent_id,
                    session_id=memory.metadata.session_id,
                    user_id=memory.metadata.user_id,
                    interaction_id=memory.metadata.custom_metadata.get("interaction_id"),
                    feedback_type=content["feedback_type"],
                    value=FeedbackValue(**content["value"]),
                    context=content["context"],
                    created_at=memory.created_at
                )
                
                # Cache in memory
                self._feedback_store[feedback_id] = feedback
                
                return feedback
        
        return None
    
    async def get_agent_feedback(
        self,
        agent_id: str,
        feedback_type: Optional[FeedbackType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AgentFeedback]:
        """
        Get feedback for a specific agent.
        
        Args:
            agent_id: ID of

