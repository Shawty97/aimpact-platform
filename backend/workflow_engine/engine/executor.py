"""
Workflow Executor

The main orchestrator for workflow execution. Manages the overall execution
process, dispatches nodes to appropriate handlers, and tracks execution state.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

from ..models.workflow import Workflow, Node, Edge, NodeType, EdgeType
from ..models.execution import (
    WorkflowExecution, NodeExecution, EdgeExecution, 
    ExecutionStatus, NodeExecutionStatus
)
from .state_manager import WorkflowStateManager
from .node_handlers import get_node_handler
from .experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Main orchestrator for workflow execution."""
    
    def __init__(self):
        """Initialize the workflow executor."""
        self.state_manager = WorkflowStateManager()
        self.experiment_manager = ExperimentManager()
        self._active_executions: Dict[str, asyncio.Task] = {}
    
    async def start_execution(
        self, 
        workflow: Workflow, 
        input_data: Dict[str, Any] = None,
        variables: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """
        Start executing a workflow.
        
        Args:
            workflow: The workflow to execute
            input_data: Input data for the workflow
            variables: Initial variables for the workflow
            
        Returns:
            The created workflow execution
        """
        # Create workflow execution
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(),
            input_data=input_data or {},
            variables=variables or {},
            created_by=workflow.created_by
        )
        
        # Save execution
        await self.state_manager.save_workflow_execution(execution)
        
        # Start execution in background
        task = asyncio.create_task(self._execute_workflow(workflow, execution))
        self._active_executions[execution.id] = task
        
        return execution
    
    async def _execute_workflow(
        self, 
        workflow: Workflow, 
        execution: WorkflowExecution
    ) -> None:
        """
        Execute a workflow.
        
        Args:
            workflow: The workflow to execute
            execution: The workflow execution
        """
        start_time = time.time()
        
        try:
            # Find start node
            start_node = next(node for node in workflow.nodes if node.type == NodeType.START)
            
            # Execute start node
            await self._execute_node(workflow, execution, start_node)
            
            # Mark execution as completed
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.id}: {str(e)}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            execution.execution_time_ms = int((time.time() - start_time) * 1000)
        
        finally:
            # Save execution state
            await self.state_manager.save_workflow_execution(execution)
            
            # Remove from active executions
            self._active_executions.pop(execution.id, None)
    
    async def _execute_node(
        self, 
        workflow: Workflow, 
        execution: WorkflowExecution, 
        node: Node,
        input_data: Dict[str, Any] = None
    ) -> Optional[NodeExecution]:
        """
        Execute a single node in the workflow.
        
        Args:
            workflow: The workflow being executed
            execution: The workflow execution
            node: The node to execute
            input_data: Input data for the node
            
        Returns:
            The node execution or None if the node is an end node
        """
        logger.info(f"Executing node {node.id} ({node.name}) of type {node.type}")
        
        # Check if it's an end node
        if node.type == NodeType.END:
            logger.info(f"Reached end node {node.id}")
            return None
        
        # Create node execution
        node_execution = NodeExecution(
            workflow_execution_id=execution.id,
            node_id=node.id,
            node_type=node.type,
            status=NodeExecutionStatus.RUNNING,
            started_at=datetime.now(),
            input_data=input_data or {}
        )
        
        # Add to current executions
        execution.current_node_executions.append(node_execution.id)
        execution.node_executions.append(node_execution)
        
        # Save state
        await self.state_manager.save_workflow_execution(execution)
        
        start_time = time.time()
        
        try:
            # Get appropriate handler for node type
            handler = get_node_handler(node.type)
            
            # Execute the node with the handler
            result = await handler.execute(node, node_execution, execution, self)
            
            # Update node execution
            node_execution.status = NodeExecutionStatus.COMPLETED
            node_execution.completed_at = datetime.now()
            node_execution.execution_time_ms = int((time.time() - start_time) * 1000)
            node_execution.output_data = result or {}
            
            # Handle special node types
            if node.type == NodeType.EXPERIMENT:
                await self._handle_experiment_node(workflow, execution, node, node_execution)
            else:
                # Find and follow outgoing edges
                await self._follow_edges(workflow, execution, node, node_execution)
                
        except Exception as e:
            logger.error(f"Error executing node {node.id}: {str(e)}")
            node_execution.status = NodeExecutionStatus.FAILED
            node_execution.error_message = str(e)
            node_execution.completed_at = datetime.now()
            node_execution.execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Find and follow error edges
            await self._follow_error_edges(workflow, execution, node, node_execution)
        
        finally:
            # Remove from current executions
            if node_execution.id in execution.current_node_executions:
                execution.current_node_executions.remove(node_execution.id)
            
            # Save state
            await self.state_manager.save_workflow_execution(execution)
        
        return node_execution
    
    async def _handle_experiment_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        node: Node,
        node_execution: NodeExecution
    ) -> None:
        """
        Handle experiment (A/B testing) node.
        
        Args:
            workflow: The workflow being executed
            execution: The workflow execution
            node: The experiment node
            node_execution: The node execution
        """
        # Determine which variant to use
        variant, experiment_execution = await self.experiment_manager.assign_variant(
            workflow, execution, node, node_execution
        )
        
        # Add experiment execution to workflow execution
        execution.experiment_executions.append(experiment_execution)
        
        # Find edges for the selected variant
        if variant == "a":
            edge_type = EdgeType.EXPERIMENT_A
        else:
            edge_type = EdgeType.EXPERIMENT_B
        
        # Find and follow variant-specific edges
        variant_edges = [
            edge for edge in workflow.edges 
            if edge.source_id == node.id and edge.type == edge_type
        ]
        
        if not variant_edges:
            logger.warning(f"No edges found for experiment variant {variant}")
            return
        
        # Follow each edge
        for edge in variant_edges:
            await self._follow_edge(workflow, execution, edge, node_execution)
    
    async def _follow_edges(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        node: Node,
        node_execution: NodeExecution
    ) -> None:
        """
        Find and follow outgoing edges from a node.
        
        Args:
            workflow: The workflow being executed
            execution: The workflow execution
            node: The source node
            node_execution: The source node execution
        """
        # Find all outgoing edges
        edges = [edge for edge in workflow.edges if edge.source_id == node.id]
        
        if not edges:
            logger.warning(f"No outgoing edges found for node {node.id}")
            return
        
        # Process conditional edges
        conditional_edges = [edge for edge in edges if edge.type == EdgeType.CONDITIONAL]
        default_edges = [edge for edge in edges if edge.type == EdgeType.DEFAULT]
        standard_edges = [edge for edge in edges if edge.type == EdgeType.STANDARD]
        
        # If there are conditional edges, evaluate them
        if conditional_edges:
            # Track if any condition was met
            condition_met = False
            
            for edge in conditional_edges:
                # Evaluate condition (in real implementation, this would use a condition evaluator)
                condition_result = True  # Placeholder for condition evaluation
                
                if condition_result:
                    condition_met = True
                    await self._follow_edge(workflow, execution, edge, node_execution, condition_result)
            
            # If no condition was met, follow default edges
            if not condition_met and default_edges:
                for edge in default_edges:
                    await self._follow_edge(workflow, execution, edge, node_execution)
                    
        # If there are no conditional edges, follow all standard edges
        elif standard_edges:
            for edge in standard_edges:
                await self._follow_edge(workflow, execution, edge, node_execution)
    
    async def _follow_error_edges(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        node: Node,
        node_execution: NodeExecution
    ) -> None:
        """
        Find and follow error edges from a node.
        
        Args:
            workflow: The workflow being executed
            execution: The workflow execution
            node: The source node
            node_execution: The source node execution
        """
        # Find error edges
        error_edges = [
            edge for edge in workflow.edges 
            if edge.source_id == node.id and edge.type == EdgeType.ERROR
        ]
        
        if not error_edges:
            logger.warning(f"No error edges found for node {node.id}")
            return
        
        # Follow each error edge
        for edge in error_edges:
            await self._follow_edge(workflow, execution, edge, node_execution)
    
    async def _follow_edge(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        edge: Edge,
        source_node_execution: NodeExecution,
        condition_result: Optional[bool] = None
    ) -> None:
        """
        Follow an edge to its target node.
        
        Args:
            workflow: The workflow being executed
            execution: The workflow execution
            edge: The edge to follow
            source_node_execution: The source node execution
            condition_result: The result of condition evaluation for conditional edges
        """
        # Find target node
        target_node = next((node for node in workflow.nodes if node.id == edge.target_id), None)
        
        if not target_node:
            logger.error(f"Target node {edge.target_id} not found")
            return
        
        # Create edge execution
        edge_execution = EdgeExecution(
            workflow_execution_id=execution.id,
            edge_id=edge.id,
            source_node_execution_id=source_node_execution.id,
            target_node_execution_id="pending",  # Will be updated after target node execution
            traversed_at=datetime.now(),
            condition_result=condition_result
        )
        
        # Add to executions
        execution.edge_executions.append(edge_execution)
        
        # Save state
        await self.state_manager.save_workflow_execution(execution)
        
        # Execute target node
        target_node_execution = await self._execute_node(
            workflow, execution, target_node, source_node_execution.output_data
        )
        
        # Update edge execution with target node execution ID
        if target_node_execution:
            edge_execution.target_node_execution_id = target_node_execution.id
            await self.state_manager.save_workflow_execution(execution)
    
    async def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running workflow execution.
        
        Args:
            execution_id: ID of the workflow execution to pause
            
        Returns:
            True if successfully paused, False otherwise
        """
        # Load execution
        execution = await self.state_manager.get_workflow_execution(execution_id)
        
        if not execution:
            logger.error(f"Execution {execution_id} not found")
            return False
        
        if execution.status != ExecutionStatus.RUNNING:
            logger.warning(f"Cannot pause execution {execution_id} with status {execution.status}")
            return False
        
        # Update status
        execution.status = ExecutionStatus.PAUSED
        await self.state_manager.save_workflow_execution(execution)
        
        # Cancel task if exists
        task = self._active_executions.pop(execution_id, None)
        if task:
            task.cancel()
        
        return True
    
    async def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused workflow execution.
        
        Args:
            execution_id: ID of the workflow execution to resume
            
        Returns:
            True if successfully resumed, False otherwise
        """
        # Load execution
        execution = await self.state_manager.get_workflow_execution(execution_id)
        
        if not execution:
            logger.error(f"Execution {execution_id} not found")
            return False
        
        if execution.status != ExecutionStatus.PAUSED:
            logger.warning(f"Cannot resume execution {execution_id} with status {execution.status}")
            return False
        
        # Load workflow
        workflow = await self.state_manager.get_workflow(execution.workflow_id)
        
        if not workflow:
            logger.error(f"Workflow {execution.workflow_id} not found")

