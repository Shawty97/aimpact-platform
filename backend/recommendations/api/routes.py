"""
API routes for the Recommendations service.

This module defines the FastAPI routes for workflow recommendations,
including getting recommendations, applying them, and providing feedback.
"""

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse

from ..service import RecommendationService
from ..models import (
    Recommendation, RecommendationType, RecommendationRequest, RecommendationResponse
)
from .models import (
    GetRecommendationsRequest, ApplyRecommendationRequest,
    RecommendationPreviewResponse, RecommendationFeedbackRequest
)
from ...app.api.deps import get_current_user, get_recommendation_service, get_workflow_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/workflow/{workflow_id}", 
    response_model=RecommendationResponse,
    summary="Get recommendations for a workflow"
)
async def get_workflow_recommendations(
    workflow_id: str = Path(..., description="ID of the workflow to get recommendations for"),
    params: GetRecommendationsRequest = Depends(),
    current_user = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Get intelligent recommendations for improving a workflow.
    
    This endpoint analyzes a workflow and provides recommendations for:
    - Structure optimizations
    - Prompt enhancements
    - Module suggestions
    - Error prevention measures
    - Performance improvements
    
    The recommendations are ranked by priority and confidence.
    """
    try:
        # Check if user has access to this workflow
        # In a real implementation, this would be handled by a permission check
        
        # Create request
        request = RecommendationRequest(
            workflow_id=workflow_id,
            focus_areas=params.focus_areas,
            node_ids=params.node_ids,
            max_suggestions=params.max_suggestions,
            min_confidence=params.min_confidence,
            include_reasoning=params.include_reasoning,
            user_id=current_user.id
        )
        
        # Generate recommendations
        response = await recommendation_service.generate_recommendations(request)
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.post(
    "/apply/{recommendation_id}",
    response_model=Dict[str, Any],
    summary="Apply a recommendation"
)
async def apply_recommendation(
    recommendation_id: str = Path(..., description="ID of the recommendation to apply"),
    apply_request: ApplyRecommendationRequest = Body(...),
    current_user = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    workflow_service = Depends(get_workflow_service)
):
    """
    Apply a recommendation to a workflow.
    
    This endpoint:
    1. Retrieves the recommendation
    2. Applies the changes to the workflow
    3. Optionally creates a new workflow version
    4. Updates the recommendation status
    
    Returns the updated workflow information.
    """
    try:
        # Get the recommendation
        recommendation = await recommendation_service.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {recommendation_id} not found"
            )
        
        # Check if user has access to the workflow
        # In a real implementation, this would be handled by a permission check
        
        # Get the workflow
        workflow_id = recommendation.workflow_id
        workflow = await workflow_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Apply the recommendation
        result = await _apply_recommendation_to_workflow(
            recommendation=recommendation,
            workflow=workflow,
            customizations=apply_request.customizations,
            workflow_service=workflow_service,
            save_as_version=apply_request.save_as_version,
            version_name=apply_request.version_name
        )
        
        # Mark the recommendation as applied
        updated_recommendation = await recommendation_service.apply_recommendation(
            recommendation_id=recommendation_id,
            user_id=current_user.id
        )
        
        # Return the result
        return {
            "workflow_id": workflow_id,
            "recommendation_id": recommendation_id,
            "applied": True,
            "new_version_id": result.get("new_version_id"),
            "changes": result.get("changes"),
            "timestamp": updated_recommendation.applied_at
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error applying recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply recommendation"
        )


@router.get(
    "/preview/{recommendation_id}",
    response_model=RecommendationPreviewResponse,
    summary="Preview a recommendation before applying it"
)
async def preview_recommendation(
    recommendation_id: str = Path(..., description="ID of the recommendation to preview"),
    customizations: Optional[Dict[str, Any]] = Query(None, description="Custom modifications to the recommendation"),
    current_user = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    workflow_service = Depends(get_workflow_service)
):
    """
    Preview the changes that would be made by applying a recommendation.
    
    This endpoint:
    1. Retrieves the recommendation
    2. Generates a before/after preview of the changes
    3. Identifies potential issues
    
    It does NOT make any actual changes to the workflow.
    """
    try:
        # Get the recommendation
        recommendation = await recommendation_service.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {recommendation_id} not found"
            )
        
        # Check if user has access to the workflow
        # In a real implementation, this would be handled by a permission check
        
        # Get the workflow
        workflow_id = recommendation.workflow_id
        workflow = await workflow_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Generate preview
        preview = await _generate_recommendation_preview(
            recommendation=recommendation,
            workflow=workflow,
            customizations=customizations
        )
        
        return preview
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error previewing recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preview recommendation"
        )


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit feedback on a recommendation"
)
async def submit_recommendation_feedback(
    feedback: RecommendationFeedbackRequest,
    current_user = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Submit feedback on a recommendation.
    
    This feedback is used to improve the quality of future recommendations.
    """
    try:
        # Get the recommendation
        recommendation = await recommendation_service.get_recommendation(feedback.recommendation_id)
        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {feedback.recommendation_id} not found"
            )
        
        # Store the feedback
        # In a real implementation, this would be stored in a database
        
        # For now, just log it
        logger.info(
            f"Received feedback for recommendation {recommendation.id}: "
            f"useful={feedback.useful}, applied={feedback.applied}, "
            f"rating={feedback.rating}, comments={feedback.comments}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"message": "Feedback received successfully"}
        )
        
    except Exception as e:
        logger.exception(f"Error processing feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process feedback"
        )


# Helper functions

async def _apply_recommendation_to_workflow(
    recommendation: Recommendation,
    workflow: Any,
    customizations: Optional[Dict[str, Any]],
    workflow_service: Any,
    save_as_version: bool = False,
    version_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply a recommendation to a workflow.
    
    Args:
        recommendation: The recommendation to apply
        workflow: The workflow to modify
        customizations: Custom modifications to the recommendation
        workflow_service: Service for manipulating workflows
        save_as_version: Whether to save as a new version
        version_name: Name for the new version
        
    Returns:
        Dictionary with the result of the operation
    """
    result = {
        "changes": [],
        "new_version_id": None
    }
    
    # Create a working copy of the workflow
    workflow_copy = workflow.copy() if hasattr(workflow, "copy") else workflow
    
    # Apply changes based on recommendation type
    if recommendation.type == RecommendationType.WORKFLOW_OPTIMIZATION:
        _apply_workflow_optimization(workflow_copy, recommendation.workflow_optimization, customizations)
        result["changes"].append("Workflow structure optimized")
        
    elif recommendation.type == RecommendationType.PROMPT_ENHANCEMENT:
        _apply_prompt_enhancement(workflow_copy, recommendation.prompt_enhancement, customizations)
        result["changes"].append(f"Enhanced prompt in node {recommendation.prompt_enhancement.node_id}")
        
    elif recommendation.type == RecommendationType.MODULE_SUGGESTION:
        module_id = _apply_module_suggestion(workflow_copy, recommendation.module_suggestion, customizations)
        result["changes"].append(f"Added module '{recommendation.module_suggestion.module_name}'")
        
    # Save the modified workflow
    if save_as_version:
        # Create a new version
        version_name = version_name or f"Recommendation {recommendation.id[:8]} applied"
        new_version = await workflow_service.create_workflow_version(
            workflow_id=workflow.id,
            workflow_data=workflow_copy,
            version_name=version_name
        )
        result["new_version_id"] = new_version.id
    else:
        # Update the existing workflow
        await workflow_service.update_workflow(
            workflow_id=workflow.id,
            workflow_data=workflow_copy
        )
    
    return result


async def _generate_recommendation_preview(
    recommendation: Recommendation,
    workflow: Any,
    customizations: Optional[Dict[str, Any]] = None
) -> RecommendationPreviewResponse:
    """
    Generate a preview of applying a recommendation.
    
    Args:
        recommendation: The recommendation to preview
        workflow: The workflow to preview changes to
        customizations: Custom modifications to the recommendation
        
    Returns:
        Preview of the changes
    """
    # Create a simplified version of the workflow for "before"
    before = _get_simplified_workflow(workflow)
    
    # Create a working copy of the workflow
    workflow_copy = workflow.copy() if hasattr(workflow, "copy") else workflow
    
    # Apply changes based on recommendation type
    changes_summary = ""
    potential_issues = []
    
    try:
        if recommendation.type == RecommendationType.WORKFLOW_OPTIMIZATION:
            opt = recommendation.workflow_optimization
            _apply_workflow_optimization(workflow_copy, opt, customizations)
            changes_summary = f"Optimize workflow structure: {opt.optimization_type}"
            
        elif recommendation.type == RecommendationType.PROMPT_ENHANCEMENT:
            enh = recommendation.prompt_enhancement
            _apply_prompt_enhancement(workflow_copy, enh, customizations)
            changes_summary = f"Enhance prompt in node '{workflow.nodes[enh.node_id].name}'"
            
        elif recommendation.type == RecommendationType.MODULE_SUGGESTION:
            su

