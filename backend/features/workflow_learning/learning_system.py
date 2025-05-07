"""
Workflow Learning System Core Module

This module provides the main functionality for analyzing workflow execution patterns,
identifying optimization opportunities, and suggesting improvements.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field

from .models import WorkflowPattern, PatternMetric, LearningStrategy
from .optimizers import PatternOptimizer, PerformancePredictor
from ...workflow_engine.models.workflow import Workflow
from ...workflow_engine.models.execution import WorkflowExecution, ExecutionStatus
from ...workflow_engine.engine.executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class WorkflowLearningSystem:
    """
    Advanced Workflow Learning System that analyzes execution patterns and
    suggests optimizations based on historical workflow execution data.
    
    Key capabilities:
    1. Pattern recognition: Identifies common workflow execution patterns
    2. Bottleneck detection: Locates performance bottlenecks and inefficiencies
    3. Optimization suggestions: Recommends changes to improve workflow performance
    4. Self-improvement: Learns from applied optimizations to improve future suggestions
    """
    
    def __init__(
        self,
        learning_strategy: LearningStrategy = LearningStrategy.BALANCED,
        min_executions_for_learning: int = 5,
        analysis_window_days: int = 30,
        confidence_threshold: float = 0.75,
        enable_auto_optimization: bool = False
    ):
        """
        Initialize the Workflow Learning System.
        
        Args:
            learning_strategy: The strategy to use for learning and optimization
            min_executions_for_learning: Minimum number of executions needed before learning
            analysis_window_days: How many days of past executions to analyze
            confidence_threshold: Minimum confidence required for optimization suggestions
            enable_auto_optimization: Whether to automatically apply optimizations
        """
        self.learning_strategy = learning_strategy
        self.min_executions_for_learning = min_executions_for_learning
        self.analysis_window_days = analysis_window_days
        self.confidence_threshold = confidence_threshold
        self.enable_auto_optimization = enable_auto_optimization
        
        # Initialize optimizers and predictors
        self.pattern_optimizer = PatternOptimizer()
        self.performance_predictor = PerformancePredictor()
        
        # Cache for recent analyses
        self._analysis_cache: Dict[str, Tuple[datetime, Any]] = {}
        
        logger.info(f"Initialized WorkflowLearningSystem with {learning_strategy} strategy")
    
    async def analyze_workflow(self, workflow_id: str, state_manager) -> Dict[str, Any]:
        """
        Analyze a workflow's execution history and identify optimization opportunities.
        
        Args:
            workflow_id: ID of the workflow to analyze
            state_manager: The state manager to use for retrieving execution data
            
        Returns:
            Analysis results containing patterns, bottlenecks, and optimization suggestions
        """
        logger.info(f"Analyzing workflow {workflow_id}")
        
        # Check cache for recent analysis
        if workflow_id in self._analysis_cache:
            cache_time, cache_result = self._analysis_cache[workflow_id]
            if datetime.now() - cache_time < timedelta(hours=1):
                logger.debug(f"Using cached analysis for workflow {workflow_id}")
                return cache_result
        
        # Load workflow
        workflow = await state_manager.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow {workflow_id} not found")
            return {"error": "Workflow not found"}
        
        # Load execution history
        cutoff_date = datetime.now() - timedelta(days=self.analysis_window_days)
        executions = await state_manager.get_workflow_executions(
            workflow_id=workflow_id,
            status=ExecutionStatus.COMPLETED,
            after_date=cutoff_date
        )
        
        if len(executions) < self.min_executions_for_learning:
            logger.warning(f"Insufficient execution data for workflow {workflow_id}. "
                          f"Need at least {self.min_executions_for_learning}, got {len(executions)}.")
            return {
                "workflow_id": workflow_id,
                "status": "insufficient_data",
                "executions_analyzed": len(executions),
                "min_required": self.min_executions_for_learning
            }
        
        # Extract execution patterns
        patterns = self._extract_patterns(workflow, executions)
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(workflow, executions)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            workflow, executions, patterns, bottlenecks
        )
        
        # Create analysis result
        analysis_result = {
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "status": "completed",
            "executions_analyzed": len(executions),
            "analysis_timestamp": datetime.now().isoformat(),
            "patterns": patterns,
            "bottlenecks": bottlenecks,
            "optimization_suggestions": optimization_suggestions,
            "prediction": {
                "current_avg_execution_time_ms": self._calculate_avg_execution_time(executions),
                "estimated_optimized_time_ms": self._estimate_optimized_time(
                    executions, optimization_suggestions
                )
            }
        }
        
        # Cache result
        self._analysis_cache[workflow_id] = (datetime.now(), analysis_result)
        
        return analysis_result
    
    def _extract_patterns(
        self, 
        workflow: Workflow, 
        executions: List[WorkflowExecution]
    ) -> List[WorkflowPattern]:
        """Extract common execution patterns from workflow execution history."""
        logger.debug(f"Extracting patterns from {len(executions)} executions")
        
        patterns = []
        
        # Extract node execution sequences
        sequences = []
        for execution in executions:
            # Sort node executions by timestamp
            sorted_nodes = sorted(
                execution.node_executions, 
                key=lambda n: n.started_at if n.started_at else datetime.min
            )
            
            # Create sequence of node IDs
            sequence = [node.node_id for node in sorted_nodes]
            sequences.append((sequence, execution))
        
        # Group similar sequences
        if sequences:
            # Convert sequences to feature vectors (this is simplified)
            # In a real implementation, more sophisticated techniques would be used
            node_ids = set()
            for seq, _ in sequences:
                node_ids.update(seq)
            
            # Create feature vectors (presence/absence of nodes)
            node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
            
            vectors = []
            for seq, _ in sequences:
                vector = [0] * len(node_id_to_idx)
                for node_id in seq:
                    if node_id in node_id_to_idx:
                        vector[node_id_to_idx[node_id]] = 1
                vectors.append(vector)
            
            # Cluster sequences
            if vectors:
                clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors)
                
                # Group by cluster
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(sequences[i])
                
                # Create pattern for each significant cluster
                for label, cluster_sequences in clusters.items():
                    if label != -1 and len(cluster_sequences) >= 2:  # Ignore noise and small clusters
                        # Calculate metrics for this pattern
                        execution_times = []
                        success_rate = 0
                        
                        for _, execution in cluster_sequences:
                            if execution.execution_time_ms:
                                execution_times.append(execution.execution_time_ms)
                            if execution.status == ExecutionStatus.COMPLETED:
                                success_rate += 1
                        
                        if execution_times:
                            avg_time = sum(execution_times) / len(execution_times)
                            success_rate = success_rate / len(cluster_sequences)
                            
                            # Create pattern
                            pattern = WorkflowPattern(
                                id=f"pattern_{workflow.id}_{label}",
                                workflow_id=workflow.id,
                                frequency=len(cluster_sequences) / len(executions),
                                metrics={
                                    "avg_execution_time_ms": avg_time,
                                    "success_rate": success_rate,
                                },
                                sample_execution_ids=[e.id for _, e in cluster_sequences[:3]]
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_bottlenecks(
        self, 
        workflow: Workflow, 
        executions: List[WorkflowExecution]
    ) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks in workflow executions."""
        logger.debug(f"Detecting bottlenecks for workflow {workflow.id}")
        
        bottlenecks = []
        
        # Aggregate node execution times
        node_times = {}
        for execution in executions:
            for node_execution in execution.node_executions:
                if node_execution.execution_time_ms:
                    if node_execution.node_id not in node_times:
                        node_times[node_execution.node_id] = []
                    node_times[node_execution.node_id].append(node_execution.execution_time_ms)
        
        # Identify nodes with high execution times
        for node_id, times in node_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                
                # Find node details
                node = next((n for n in workflow.nodes if n.id == node_id), None)
                node_name = node.name if node else "Unknown"
                node_type = node.type if node else "Unknown"
                
                # Determine if this is a bottleneck
                # In a real implementation, more sophisticated detection would be used
                if avg_time > 1000:  # More than 1 second on average
                    bottleneck = {
                        "node_id": node_id,
                        "node_name": node_name,
                        "node_type": node_type,
                        "avg_execution_time_ms": avg_time,
                        "max_execution_time_ms": max_time,
                        "severity": "high" if avg_time > 5000 else "medium",
                        "frequency": len(times) / len(executions),
                        "impact_percentage": 0.0  # Will be calculated below
                    }
                    
                    # Calculate impact as percentage of overall execution time
                    total_execution_times = [e.execution_time_ms for e in executions if e.execution_time_ms]
                    if total_execution_times:
                        avg_total_time = sum(total_execution_times) / len(total_execution_times)
                        bottleneck["impact_percentage"] = (avg_time / avg_total_time) * 100
                    
                    bottlenecks.append(bottleneck)
        
        # Sort bottlenecks by impact
        bottlenecks.sort(key=lambda b: b["impact_percentage"], reverse=True)
        
        return bottlenecks
    
    def _generate_optimization_suggestions(
        self,
        workflow: Workflow,
        executions: List[WorkflowExecution],
        patterns: List[WorkflowPattern],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for workflow optimization."""
        logger.debug(f"Generating optimization suggestions for workflow {workflow.id}")
        
        suggestions = []
        
        # Bottleneck-based suggestions
        for bottleneck in bottlenecks:
            node_id = bottleneck["node_id"]
            node = next((n for n in workflow.nodes if n.id == node_id), None)
            
            if node:
                if bottleneck["impact_percentage"] > 20:  # High impact
                    suggestion = {
                        "type": "bottleneck_resolution",
                        "target": {
                            "node_id": node_id,
                            "node_name": bottleneck["node_name"],
                            "node_type": bottleneck["node_type"]
                        },
                        "description": f"Optimize high-impact node '{bottleneck['node_name']}' "
                                      f"which accounts for {bottleneck['impact_percentage']:.1f}% "
                                      f"of total execution time",
                        "recommendation": self._get_node_optimization_recommendation(node),
                        "estimated_improvement": {
                            "execution_time_reduction_ms": bottleneck["avg_execution_time_ms"] * 0.5,  # Estimate 50% improvement
                            "confidence": 0.8
                        }
                    }
                    suggestions.append(suggestion)
        
        # Pattern-based suggestions
        if patterns:
            # Find slow patterns
            slow_patterns = sorted(
                patterns, 
                key=lambda p: p.metrics.get("avg_execution_time_ms", 0), 
                reverse=True
            )
            
            if slow_patterns:
                slowest = slow_patterns[0]
                
                # Get sample executions
                sample

