"""
AImpact AutoPilot System

This module provides automated workflow optimization, performance monitoring,
and continuous improvement capabilities for the AImpact platform.

Features:
- Real-time performance monitoring
- Automated workflow optimization
- A/B testing management
- Cost optimization
- Learning from successful patterns
- Anomaly detection
- Reinforcement learning
"""

__version__ = "0.1.0"

from .models import OptimizationMetric, OptimizationStrategy, AutoPilotConfig
from .monitor import PerformanceMonitor
from .optimizer import WorkflowOptimizer
from .learner import PatternLearner
from .experiment_manager import ExperimentManager
from .autopilot import AutoPilot

__all__ = [
    "OptimizationMetric",
    "OptimizationStrategy",
    "AutoPilotConfig",
    "PerformanceMonitor",
    "WorkflowOptimizer",
    "PatternLearner",
    "ExperimentManager",
    "AutoPilot"
]

