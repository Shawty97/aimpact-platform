import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path, status, BackgroundTasks
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger("aimpact_api.workflows")

# Initialize router
router = APIRouter()

# -------------------- Pydantic Models --------------------

class NodeType(str, Enum):
    """Types of nodes in a workflow."""
    START = "start"
    END = "end"
    AGENT = "agent"
    DECISION = "decision"
    TRANSFORMATION = "transformation"
    API_CALL = "api_call"
    USER_INPUT = "user_input"
    DATA_SOURCE = "data_source"

class EdgeType(str, Enum):
    """Types of connections between nodes."""
    DEFAULT = "default"
    CONDITION = "condition"
    ERROR = "error"
    SUCCESS = "success"

class WorkflowState(str, Enum):
    """Possible states of a workflow execution."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class NodeState(str, Enum):
    """Possible states of a node in a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class NodeDefinition(BaseModel):
    """Definition of a node in a workflow."""
    id: str
    type: NodeType
    name: str
    description: Optional[str] = None
    agent_id: Optional[str] = None  # Reference to an agent if type is AGENT
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, float] = Field(default_factory=dict)  # For UI positioning

class EdgeDefinition(BaseModel):
    """Definition of an edge connecting nodes in a workflow."""
    id: str
    source_id: str
    target_id: str
    type: EdgeType = EdgeType.DEFAULT
    condition: Optional[str] = None  # Expression for conditional branching
    name: Optional[str] = None

class WorkflowCreateUpdate(BaseModel):
    """Schema for creating or updating a workflow."""
    name: str
    description: Optional[str] = None
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Name must be at least 3 characters')
        return v.strip()
    
    @validator('nodes')
    def must_have_start_and_end(cls, v):
        node_types = [node.type for node in v]
        if NodeType.START not in node_types:
            raise ValueError('Workflow must have a START node')
        if NodeType.END not in node_types:
            raise ValueError('Workflow must have an END node')
        return v

class Workflow(BaseModel):
    """Schema for a workflow."""
    id: str
    name: str
    description: Optional[str] = None
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    created_at: datetime
    updated_at: datetime

class NodeExecution(BaseModel):
    """Schema for the execution state of a node."""
    node_id: str
    state: NodeState
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class WorkflowExecution(BaseModel):
    """Schema for a workflow execution."""
    id: str
    workflow_id: str
    state: WorkflowState
    node_executions: Dict[str, NodeExecution] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: Optional[datetime] = None

class WorkflowExecutionRequest(BaseModel):
    """Schema for requesting a workflow execution."""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = Field(default=True, description="Whether to run the workflow asynchronously")

class WorkflowExecutionResponse(BaseModel):
    """Schema for a workflow execution response."""
    execution_id: str
    workflow_id: str
    state: WorkflowState
    message: str

# -------------------- Mock Database --------------------
# In a real implementation, this would be replaced with database connections

# Simple in-memory storage for demonstration
workflow_db: Dict[str, Workflow] = {}
execution_db: Dict[str, WorkflowExecution] = {}

# -------------------- Helper Functions --------------------

def get_workflow_by_id(workflow_id: str) -> Workflow:
    """Retrieve a workflow by its ID."""
    if workflow_id not in workflow_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    return workflow_db[workflow_id]

def get_execution_by_id(execution_id: str) -> WorkflowExecution:
    """Retrieve a workflow execution by its ID."""
    if execution_id not in execution_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow execution with ID {execution_id} not found"
        )
    return execution_db[execution_id]

async def execute_workflow(workflow_id: str, execution_id: str, inputs: Dict[str, Any]):
    """
    Asynchronously execute a workflow.
    
    This is a simplified implementation that simulates workflow execution.
    In a real system, this would involve complex orchestration logic.
    """
    workflow = get_workflow_by_id(workflow_id)
    execution = get_execution_by_id(execution_id)
    
    # Update execution state to running
    execution.state = WorkflowState.RUNNING
    
    try:
        # Find start node
        start_node = next(node for node in workflow.nodes if node.type == NodeType.START)
        current_node_id = start_node.id
        
        # Initialize all nodes as pending
        for node in workflow.nodes:
            execution.node_executions[node.id] = NodeExecution(
                node_id=node.id,
                state=NodeState.PENDING
            )
        
        # Process nodes until we reach the end
        while current_node_id:
            # Get current node
            current_node = next(node for node in workflow.nodes if node.id == current_node_id)
            node_execution = execution.node_executions[current_node_id]
            
            # Update node state to running
            node_execution.state = NodeState.RUNNING
            node_execution.start_time = datetime.now()
            
            # Simulate node execution
            logger.info(f"Executing node {current_node.name} (ID: {current_node_id})")
            await asyncio.sleep(1)  # Simulate processing time
            
            # Process node based on type
            try:
                if current_node.type == NodeType.END:
                    # End node reached, workflow is complete
                    node_execution.state = NodeState.COMPLETED
                    node_execution.end_time = datetime.now()
                    execution.state = WorkflowState.COMPLETED
                    execution.completed_at = datetime.now()
                    current_node_id = None  # Stop execution
                    continue
                
                if current_node.type == NodeType.AGENT:
                    # Simulate agent processing
                    node_execution.result = {
                        "message": f"Agent {current_node.agent_id} processed the input",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Mark node as completed
                node_execution.state = NodeState.COMPLETED
                node_execution.end_time = datetime.now()
                
                # Find next node
                next_edges = [edge for edge in workflow.edges if edge.source_id == current_node_id]
                
                if not next_edges:
                    # No outgoing edges, end execution
                    execution.state = WorkflowState.FAILED
                    execution.completed_at = datetime.now()
                    logger.error(f"Workflow execution failed: no outgoing edges from node {current_node_id}")
                    break
                
                # In a real implementation, evaluate conditions for DECISION nodes
                # For simplicity, we'll just take the first edge
                current_node_id = next_edges[0].target_id
                
            except Exception as e:
                # Handle node execution error
                node_execution.state = NodeState.FAILED
                node_execution.end_time = datetime.now()
                node_execution.error = str(e)
                
                # Find error edge if available
                error_edges = [edge for edge in workflow.edges 
                               if edge.source_id == current_node_id and edge.type == EdgeType.ERROR]
                
                if error_edges:
                    # Follow error path
                    current_node_id = error_edges[0].target_id
                else:
                    # No error handling, workflow fails
                    execution.state = WorkflowState.FAILED
                    execution.completed_at = datetime.now()
                    logger.error(f"Workflow execution failed at node {current_node_id}: {str(e)}")
                    break
        
        # Record outputs from the last completed nodes
        completed_nodes = [node_exec for node_exec in execution.node_executions.values() 
                          if node_exec.state == NodeState.COMPLETED and node_exec.result]
        
        if completed_nodes:
            for node in completed_nodes:
                if node.result:
                    execution.outputs[node.node_id] = node.result
    
    except Exception as e:
        # Handle overall workflow error
        execution.state = WorkflowState.FAILED
        execution.completed_at = datetime.now()
        logger.error(f"Workflow execution failed: {str(e)}")
    
    logger.info(f"Workflow execution completed: {execution_id} (State: {execution.state})")

# -------------------- API Endpoints --------------------

@router.post("/", response_model=Workflow, status_code=status.HTTP_201_CREATED)
async def create_workflow(workflow_data: WorkflowCreateUpdate):
    """
    Create a new workflow.
    
    Returns the created workflow with its generated ID.
    """
    workflow_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Generate IDs for nodes and edges if not provided
    for node in workflow_data.nodes:
        if not node.id:
            node.id = str(uuid.uuid4())
    
    for edge in workflow_data.edges:
        if not edge.id:
            edge.id = str(uuid.uuid4())
    
    new_workflow = Workflow(
        id=workflow_id,
        name=workflow_data.name,
        description=workflow_data.description,
        nodes=workflow_data.nodes,
        edges=workflow_data.edges,
        created_at=now,
        updated_at=now
    )
    
    workflow_db[workflow_id] = new_workflow
    logger.info(f"Created new workflow: {workflow_id} ({workflow_data.name})")
    
    return new_workflow

@router.get("/", response_model=List[Workflow])
async def list_workflows():
    """
    List all available workflows.
    """
    return list(workflow_db.values())

@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: str = Path(..., description="The ID of the workflow to retrieve")):
    """
    Retrieve a specific workflow by ID.
    """
    return get_workflow_by_id(workflow_id)

@router.put("/{workflow_id}", response_model=Workflow)
async def update_workflow(
    workflow_data: WorkflowCreateUpdate,
    workflow_id: str = Path(..., description="The ID of the workflow to update")
):
    """
    Update an existing workflow.
    """
    # Check if workflow exists
    get_workflow_by_id(workflow_id)
    
    now = datetime.now()
    
    # Generate IDs for new nodes and edges if not provided
    for node in workflow_data.nodes:
        if not node.id:
            node.id = str(uuid.uuid4())
    
    for edge in workflow_data.edges:
        if not edge.id:
            edge.id = str(uuid.uuid4())
    
    updated_workflow = Workflow(
        id=workflow_id,
        name=workflow_data.name,
        description=workflow_data.description,
        nodes=workflow_data.nodes,
        edges=workflow_data.edges,
        created_at=workflow_db[workflow_id].created_at,  # Preserve creation time
        updated_at=now
    )
    
    workflow_db[workflow_id] = updated_workflow
    logger.info(f"Updated workflow: {workflow_id}")
    
    return updated_workflow

@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(workflow_id: str = Path(..., description="The ID of the workflow to delete")):
    """
    Delete a workflow by ID.
    """
    get_workflow_by_id(workflow_id)  # Check if workflow exists
    del workflow_db[workflow_id]
    logger.info(f"Deleted workflow: {workflow_id}")
    
    # Also delete any executions of this workflow
    for exec_id, execution in list(execution_db.items()):
        if execution.workflow_id == workflow_id:
            del execution_db[exec_id]
    
    return None

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow_endpoint(
    execution_request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    workflow_id: str = Path(..., description="The ID of the workflow to execute")
):
    """
    Execute a workflow.
    
    Starts a new workflow execution with the provided inputs.
    """
    workflow = get_workflow_by_id(workflow_id)
    execution_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Create new execution
    new_execution = WorkflowExecution(
        id=execution_id,
        workflow_id=workflow_id,
        state=WorkflowState.CREATED,
        inputs=execution_request.inputs,
        started_at=now
    )
    
    execution_db[execution_id] = new_execution
    
    if execution_request.async_execution:
        # Start workflow execution in the background
        background_tasks.add_task(
            execute_workflow, 
            workflow_id=workflow_id, 
            execution_id=execution_id, 
            inputs=execution_request.inputs
        )
        
        message = f"Workflow execution started asynchronously. Monitor progress at /workflows/executions/{execution_id}"
    else:
        # Execute workflow synchronously (blocking)
        await execute_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            inputs=execution_request.inputs
        )
        
        message = "Workflow execution completed"
    
    # Get updated state for response
    current_state = execution_db[execution_id].state
    
    return WorkflowExecutionResponse(
        execution_id=execution_id,
        workflow_id=workflow_id,
        state=current_state,
        message=message
    )

@router.get("/executions/", response_model=List[WorkflowExecution])
async def list_executions(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    state: Optional[WorkflowState] = Query(None, description="Filter by execution state")
):
    """
    List workflow executions, with optional filtering.
    """
    executions = list(execution_db.values())
    
    if workflow_id:
        executions = [exec for exec in executions if exec.workflow_id == workflow_id]
    
    if state:
        executions = [exec for exec in executions if exec.state == state]
    
    return executions

@router.get("/executions/{execution_id}", response_model=WorkflowExecution)
async def get_execution(execution_id: str = Path(..., description="The ID of the execution to retrieve")):
    """
    Retrieve a specific workflow execution by ID.
    """
    return get_execution_by_id(execution_id)

@router.post("/executions/{execution_id}/pause", response_model=WorkflowExecutionResponse)
async def pause_execution(execution_id: str = Path(..., description="The ID of the execution to pause")):
    """
    Pause a running workflow execution.
    """
    execution = get_execution_by_id(execution_id)
    
    if execution.state != WorkflowState.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause execution in state {execution.state}"
        )
    
    execution.state = WorkflowState.PAUSED
    logger.info(f"Paused workflow execution: {execution_id}")
    
    return WorkflowExecutionResponse(
        execution_id=execution_id,
        workflow_id=execution.workflow_id,
        state=execution.state,
        message="Workflow execution paused"
    )

@router.post("/executions/{execution_id}/resume", response_model=WorkflowExecutionResponse)
async def resume_execution(
    background_tasks: BackgroundTasks,
    execution_id: str = Path(..., description="The ID of the execution to resume")
):
    """
    Resume a paused workflow execution.
    """
    execution = get_execution_by_id(execution_id)
    
    if execution.state != WorkflowState.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume execution in state {execution.state}"
        )
    
    execution.state = WorkflowState.RUNNING
    logger.info(f"Resumed workflow execution: {execution_id}")
    
    # In a real implementation, we would resume from the last executed node
    # For this demo, we'll just restart the execution
    background_tasks.add_task(
        execute_workflow,
        workflow_id=execution.workflow_id,
        execution_id=execution_id,
        inputs=execution.inputs
    )
    
    return WorkflowExecutionResponse(
        execution_id=execution_id,
        workflow_id=execution.workflow_id,
        state=execution.state,
        message="Workflow execution resumed"
    )

@router.post("/executions/{execution_id}/cancel", response_model=WorkflowExecutionResponse)
async def cancel_execution(execution_id: str = Path(..., description="The ID of the execution to cancel")):
    """
    Cancel a workflow execution.
    """
    execution = get_execution_by_id(execution_id)
    
    if execution.state in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel execution in state {execution.state}"
        )
    
    execution.state = WorkflowState.FAILED
    execution.completed_at = datetime.now()
    logger.info(f"Cancelled workflow execution: {execution_id}")
    
    return WorkflowExecutionResponse(
        execution_id=execution_id,
        workflow_id=execution.workflow_id,
        state=execution.state,
        message="Workflow execution cancelled"
    )

