"""
Recommendation Service for AImpact platform.

This service provides intelligent recommendations for workflow improvement,
prompt optimization, and module suggestions.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set

import numpy as np

from ..memory.service import MemoryService
from ..memory.models import Memory, MemoryType, MemoryMetadata
from .models import (
    Recommendation, RecommendationType, RecommendationPriority, RecommendationImpact,
    ModuleSuggestion, PromptEnhancement, WorkflowOptimization,
    RecommendationRequest, RecommendationResponse
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service for generating intelligent recommendations for workflows.
    
    This service:
    - Analyzes existing workflows
    - Suggests optimizations using LLM
    - Recommends compatible modules
    - Provides prompt enhancement suggestions
    """
    
    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        llm_provider: Optional[Any] = None,
        module_registry: Optional[Any] = None,
        workflow_service: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        cache_ttl: int = 3600,  # 1 hour cache TTL
    ):
        """
        Initialize the recommendation service.
        
        Args:
            memory_service: Memory service for retrieving workflow data
            llm_provider: Provider for LLM-based recommendations
            module_registry: Registry of available modules
            workflow_service: Service for accessing workflows
            embedding_provider: Provider for generating embeddings
            cache_ttl: Cache time-to-live in seconds
        """
        self.memory_service = memory_service
        self.llm_provider = llm_provider
        self.module_registry = module_registry
        self.workflow_service = workflow_service
        self.embedding_provider = embedding_provider
        self.cache_ttl = cache_ttl
        
        # In-memory storage (replace with database in production)
        self._recommendation_store: Dict[str, Recommendation] = {}
        self._workflow_recommendation_cache: Dict[str, RecommendationResponse] = {}
        self._workflow_cache: Dict[str, Any] = {}
        self._module_compatibility_cache: Dict[str, Dict[str, float]] = {}
        
    async def generate_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Generate recommendations for a workflow.
        
        Args:
            request: Recommendation request parameters
            
        Returns:
            Response containing recommendations
        """
        workflow_id = request.workflow_id
        
        # Check cache first
        cache_key = f"{workflow_id}_{request.max_suggestions}_{request.min_confidence}"
        if cache_key in self._workflow_recommendation_cache:
            cached = self._workflow_recommendation_cache[cache_key]
            cache_age = datetime.utcnow() - cached.timestamp
            
            # Return from cache if still valid
            if cache_age < timedelta(seconds=self.cache_ttl):
                return cached
        
        # Get workflow
        workflow = await self._get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Generate recommendations
        recommendations = []
        
        # Determine which types of recommendations to generate
        focus_areas = request.focus_areas or list(RecommendationType)
        
        # Generate recommendations by type
        if RecommendationType.WORKFLOW_OPTIMIZATION in focus_areas:
            workflow_optimizations = await self._generate_workflow_optimizations(
                workflow,
                request.node_ids,
                min_confidence=request.min_confidence
            )
            recommendations.extend(workflow_optimizations)
        
        if RecommendationType.PROMPT_ENHANCEMENT in focus_areas:
            prompt_enhancements = await self._generate_prompt_enhancements(
                workflow,
                request.node_ids,
                min_confidence=request.min_confidence
            )
            recommendations.extend(prompt_enhancements)
        
        if RecommendationType.MODULE_SUGGESTION in focus_areas:
            module_suggestions = await self._generate_module_suggestions(
                workflow,
                request.node_ids,
                min_confidence=request.min_confidence
            )
            recommendations.extend(module_suggestions)
            
        if RecommendationType.ERROR_PREVENTION in focus_areas:
            error_preventions = await self._generate_error_preventions(
                workflow,
                request.node_ids,
                min_confidence=request.min_confidence
            )
            recommendations.extend(error_preventions)
            
        if RecommendationType.PERFORMANCE_IMPROVEMENT in focus_areas:
            performance_improvements = await self._generate_performance_improvements(
                workflow,
                request.node_ids,
                min_confidence=request.min_confidence
            )
            recommendations.extend(performance_improvements)
        
        # Sort by priority and confidence
        recommendations.sort(
            key=lambda r: (r.priority.value, r.confidence),
            reverse=True
        )
        
        # Limit to max suggestions
        recommendations = recommendations[:request.max_suggestions]
        
        # Generate analysis summary
        analysis_summary = await self._generate_analysis_summary(workflow, recommendations)
        
        # Store recommendations
        for recommendation in recommendations:
            self._recommendation_store[recommendation.id] = recommendation
        
        # Create response
        response = RecommendationResponse(
            workflow_id=workflow_id,
            recommendations=recommendations,
            analysis_summary=analysis_summary,
            timestamp=datetime.utcnow()
        )
        
        # Cache response
        self._workflow_recommendation_cache[cache_key] = response
        
        return response
        
    async def get_recommendation(self, recommendation_id: str) -> Optional[Recommendation]:
        """
        Get a recommendation by ID.
        
        Args:
            recommendation_id: ID of the recommendation to retrieve
            
        Returns:
            Recommendation if found, None otherwise
        """
        return self._recommendation_store.get(recommendation_id)
        
    async def apply_recommendation(
        self,
        recommendation_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Recommendation]:
        """
        Mark a recommendation as applied.
        
        Args:
            recommendation_id: ID of the recommendation to apply
            user_id: ID of the user applying the recommendation
            
        Returns:
            Updated recommendation if found, None otherwise
        """
        recommendation = self._recommendation_store.get(recommendation_id)
        if not recommendation:
            return None
            
        # Update recommendation
        recommendation.applied = True
        recommendation.applied_at = datetime.utcnow()
        
        # Store updated recommendation
        self._recommendation_store[recommendation_id] = recommendation
        
        # Clear cache for this workflow
        workflow_id = recommendation.workflow_id
        cache_keys_to_remove = [
            key for key in self._workflow_recommendation_cache.keys()
            if key.startswith(f"{workflow_id}_")
        ]
        for key in cache_keys_to_remove:
            self._workflow_recommendation_cache.pop(key, None)
            
        return recommendation
    
    async def _get_workflow(self, workflow_id: str) -> Optional[Any]:
        """Get workflow by ID from workflow service or cache."""
        if workflow_id in self._workflow_cache:
            return self._workflow_cache[workflow_id]
            
        if self.workflow_service:
            workflow = await self.workflow_service.get_workflow(workflow_id)
            if workflow:
                self._workflow_cache[workflow_id] = workflow
                return workflow
                
        return None
    
    async def _generate_workflow_optimizations(
        self,
        workflow: Any,
        node_ids: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Recommendation]:
        """Generate workflow structure optimization recommendations."""
        # Get workflow structure
        nodes = workflow.nodes
        edges = workflow.edges
        
        recommendations = []
        
        # Filter nodes if specific ones requested
        if node_ids:
            nodes = [node for node in nodes if node.id in node_ids]
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for generating workflow optimizations")
            return recommendations
        
        try:
            # Analyze workflow structure using LLM
            workflow_json = json.dumps({"nodes": [n.dict() for n in nodes], "edges": [e.dict() for e in edges]})
            
            prompt = f"""
            Analyze the following workflow structure and suggest optimizations.
            
            Workflow: {workflow.name}
            Description: {workflow.description or 'No description provided'}
            
            Structure:
            {workflow_json}
            
            Provide optimizations that could improve the workflow's efficiency, reliability, or performance.
            Focus on structural changes, node ordering, parallel execution opportunities, and error handling.
            
            Format your response as a JSON array of optimization suggestions, where each suggestion has:
            - optimization_type: Type of optimization (e.g., "parallelization", "error_handling", "node_reordering")
            - affected_nodes: Array of node IDs that would be affected
            - description: Detailed description of the optimization
            - expected_benefits: Array of expected benefits
            - implementation_complexity: One of "easy", "medium", "hard"
            - reasoning: Explanation of why this optimization would help
            """
            
            # Get LLM response
            optimization_json = await self.llm_provider.generate(prompt, json_mode=True)
            optimization_suggestions = json.loads(optimization_json)
            
            # Convert suggestions to recommendations
            for i, suggestion in enumerate(optimization_suggestions):
                # Generate unique ID
                rec_id = str(uuid.uuid4())
                
                # Determine priority based on complexity and benefits
                priority = RecommendationPriority.MEDIUM
                if suggestion["implementation_complexity"] == "easy" and len(suggestion["expected_benefits"]) >= 2:
                    priority = RecommendationPriority.HIGH
                elif suggestion["implementation_complexity"] == "hard":
                    priority = RecommendationPriority.LOW
                
                # Determine impact
                impact = RecommendationImpact.MODERATE
                benefit_text = " ".join(suggestion["expected_benefits"]).lower()
                if "significant" in benefit_text or "major" in benefit_text:
                    impact = RecommendationImpact.SIGNIFICANT
                elif "minor" in benefit_text:
                    impact = RecommendationImpact.MINOR
                
                # Calculate confidence based on reasoning
                confidence = 0.7 + (len(suggestion["reasoning"]) / 1000) * 0.2  # More detailed reasoning = higher confidence
                confidence = min(0.95, max(min_confidence, confidence))  # Clamp between min_confidence and 0.95
                
                # Create workflow optimization
                workflow_opt = WorkflowOptimization(
                    optimization_type=suggestion["optimization_type"],
                    affected_nodes=suggestion["affected_nodes"],
                    description=suggestion["description"],
                    expected_benefits=suggestion["expected_benefits"],
                    implementation_complexity=suggestion["implementation_complexity"],
                    reasoning=suggestion["reasoning"]
                )
                
                # Create recommendation
                recommendation = Recommendation(
                    id=rec_id,
                    workflow_id=workflow.id,
                    type=RecommendationType.WORKFLOW_OPTIMIZATION,
                    title=f"Workflow Optimization: {suggestion['optimization_type'].replace('_', ' ').title()}",
                    description=suggestion["description"],
                    priority=priority,
                    impact=impact,
                    workflow_optimization=workflow_opt,
                    confidence=confidence,
                    created_at=datetime.utcnow()
                )
                
                recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Error generating workflow optimizations: {e}")
        
        return recommendations
        
    async def _generate_prompt_enhancements(
        self,
        workflow: Any,
        node_ids: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Recommendation]:
        """Generate prompt enhancement recommendations."""
        recommendations = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for generating prompt enhancements")
            return recommendations
        
        try:
            # Find nodes with prompts
            prompt_nodes = []
            for node in workflow.nodes:
                # Customize this based on your workflow node structure
                if hasattr(node, 'type') and node.type in ["llm", "prompt", "text_generation"]:
                    if hasattr(node, 'config') and 'prompt' in node.config:
                        prompt_nodes.append(node)
            
            # Filter nodes if specific ones requested
            if node_ids:
                prompt_nodes = [node for node in prompt_nodes if node.id in node_ids]
            
            # Process each prompt node
            for node in prompt_nodes:
                original_prompt = node.config['prompt']
                
                # Skip if prompt is too short
                if len(original_prompt) < 10:
                    continue
                
                prompt = f"""
                Analyze and enhance the following prompt for an AI agent.
                
                Original Prompt:
                {original_prompt}
                
                Provide an enhanced version that improves:
                1. Clarity of instructions
                2. Specificity of requirements
                3. Context provided to the AI
                4. Output formatting instructions
                5. Error handling guidance
                
                Format your response as a JSON object with:
                - enhanced_prompt: The improved prompt text
                - improvements: Array of specific improvements made
                - expected_benefits: Array of expected benefits from these changes
                - reasoning: Explanation of why these improvements will help
                """
                
                # Get LLM response
                enhancement_json = await self.llm_provider.generate(prompt, json_mode=True)
                enhancement = json.loads(enhancement_json)
                
                # Generate unique ID
                rec_id = str(uuid.uuid4())
                
                # Determine confidence based on difference between original and enhanced
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, original_prompt, enhancement["enhanced_prompt"]).ratio()
                change_amount = 1 - similarity  # How much was changed
                
                # More changes generally indicate more confident recommendations
                # but we want some middle ground - too many or too few changes might indicate issues
                confidence = 0.5 + change_amount * 0.5
                confidence = min(0.95, max(min_confidence, confidence))
                
                # Create prompt enhancement
                prompt_enh = PromptEnhancement(
                    node_id=node.id,
                    original_prompt=original_prompt,
                    enhanced_prompt=enhancement["enhanced_prompt"],
                    improvements=enhancement["improvements"],
                    expected_benefits=enhancement["expected_benefits"],
                    reasoning=enhancement["reasoning"]
                )
                
                # Determine priority based on improvements
                priority = RecommendationPriority.MEDIUM
                if len(enhancement["improvements"]) >= 3:
                    priority = RecommendationPriority.HIGH
                
                # Determine impact
                impact = RecommendationImpact.MODERATE
                if change_amount > 0.4:  # Significant changes
                    impact = RecommendationImpact.SIGNIFICANT
                elif change_amount < 0.1:  # Minor changes
                    impact = RecommendationImpact.MINOR
                
                # Create recommendation
                recommendation = Recommendation(
                    id=rec_id,
                    workflow_id=workflow.id,
                    type=RecommendationType.PROMPT_ENHANCEMENT,
                    title=f"Prompt Enhancement for '{node.name}'",
                    description=f"Optimize the prompt in node '{node.name}' to improve {', '.join(enhancement['improvements'][:2])}",
                    priority=priority,
                    impact=impact,
                    prompt_enhancement=prompt_enh,
                    confidence=confidence,
                    created_at=datetime.utcnow()
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.error(f"Error generating prompt enhancements: {e}")
        
        return recommendations
        
    async def _generate_module_suggestions(
        self,
        workflow: Any,
        node_ids: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Recommendation]:
        """Generate module suggestion recommendations."""
        recommendations = []
        
        if not self.llm_provider or not self.module_registry:
            logger.warning("LLM provider or module registry not available for generating module suggestions")
            return recommendations
        
        try:
            # Get available modules from registry
            available_modules = await self.module_registry.list_modules()
            
            if not available_modules:
                return recommendations
                
            # Analyze workflow for gaps and opportunities
            workflow_json = json.dumps({
                "id": workflow.id,
                "name": workflow.name, 
                "description": workflow.description,
                "nodes": [{"id": n.id, "name": n.name, "type": n.type} for n in workflow.nodes],
                "edges": [{"source": e.source_id, "target": e.target_id} for e in workflow.edges]
            })
            
            modules_json = json.dumps([{
                "id": m.id,
                "name": m.name,
                "type": m.type,
                "description": m.description,
                "capabilities": m.capabilities
            } for m in available_modules])
            
            prompt = f"""
            Analyze this workflow and suggest compatible modules that could enhance its functionality.
            
            Workflow:
            {workflow_json}
            
            Available Modules:
            {modules_json}
            
            Identify modules that would complement the workflow or fill functionality gaps.
            Consider where these modules would fit in the workflow structure.
            
            Format your response as a JSON array of module suggestions, where each suggestion has:
            - module_id: ID of the suggested module
            - insertion_point: Node ID after which this module would be inserted (or null if it's a general suggestion)
            - description: Description of how this module would enhance the workflow
            - compatibility_score: A value between 0 and 1 indicating compatibility (higher is better)
            - configuration: Object with suggested configuration values
            - reasoning: Explanation of why this module would be beneficial
            """
            
            # Get LLM response
            suggestions_json = await self.llm_provider.generate(prompt, json_mode=True)
            module_suggestions = json.loads(suggestions_json)
            
            # Convert suggestions to recommendations
            for suggestion in module_suggestions:
                # Skip low compatibility suggestions
                if suggestion["compatibility_score"] < min_confidence:
                    continue
                
                # Find module details
                module = next((m for m in available_modules if m.id == suggestion["module_id"]), None)
                if not module:
                    continue
                
                # Generate unique ID
                rec_id = str(uuid.uuid4())
                
                # Create module suggestion
                module_sugg = ModuleSuggestion(
                    module_id=suggestion["module_id"],
                    module_name=module.name,
                    module_type=module.type,
                    description=suggestion["description"],
                    insertion_point=suggestion["insertion_point"],
                    compatibility_score=suggestion["compatibility_score"],
                    configuration=suggestion["configuration"],
                    reasoning=suggestion["reasoning"]
                )
                
                # Determine priority based on compatibility
                priority = RecommendationPriority.MEDIUM
                if suggestion["compatibility_score"] > 0.8:
                    priority = RecommendationPriority.HIGH
                elif suggestion["compatibility_score"] < 0.6:
                    priority = RecommendationPriority.LOW
                
                # Determine impact based on module capabilities
                impact = RecommendationImpact.MODERATE
                if len(module.capabilities) > 3:  # Module with many capabilities
                    impact = RecommendationImpact.SIGNIFICANT
                
                # Create recommendation
                recommendation = Recommendation(
                    id=rec_id,
                    workflow_id=workflow.id,
                    type=RecommendationType.MODULE_SUGGESTION,
                    title=f"Add '{module.name}' Module",
                    description=suggestion["description"],
                    priority=priority,
                    impact=impact,
                    module_suggestion=module_sugg,
                    confidence=suggestion["compatibility_score"],
                    created_at=datetime.utcnow()
                )
                
                recommendations.append(recommendation)
                
        except Exception as e:
            logger.error(f"Error generating module suggestions: {e}")
        
        return recommendations
    
    async def _generate_error_preventions(
        self,
        workflow: Any,
        node_ids: Optional[List[str]] = None,
        min_confidence: float = 0.7
    ) -> List[Recommendation]:
        """Generate error prevention recommendations."""
        recommendations = []
        
        if not self.llm_provider:
            logger.warning("No LLM provider available for generating error prevention suggestions")
            return recommendations
        
        try:
            # Get execution history if available
            execution_history = []
            if hasattr(workflow, 'executions') and workflow.executions:
                execution_history = workflow.executions
            
            # Find error-prone areas in workflow
            workflow_json = json.dumps({
                "nodes": [n.dict() for n in workflow.nodes],
                "edges": [e.dict() for e in workflow.edges],

