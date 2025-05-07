"""
Workflow Learning System

An advanced system that analyzes workflow execution patterns and automatically
suggests improvements based on historical performance data.
"""

from .learning_system import WorkflowLearningSystem
from .models import WorkflowPattern, PatternMetric, LearningStrategy
from .optimizers import PatternOptimizer, PerformancePredictor

__all__ = [
    'WorkflowLearningSystem',
    'WorkflowPattern',
    'PatternMetric',
    'LearningStrategy',
    'PatternOptimizer',
    'PerformancePredictor',
]

