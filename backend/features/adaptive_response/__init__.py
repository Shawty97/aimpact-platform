"""
Adaptive Response Optimization System

This module provides capabilities for dynamically adjusting AI responses
based on user interaction patterns, feedback, and contextual information.
It integrates with the workflow engine and cross-modal intelligence to
provide highly personalized and effective responses.
"""

from .optimizer import AdaptiveResponseOptimizer
from .models import (
    UserFeedback,
    ResponseAdjustment,
    InteractionPattern,
    PersonalizationProfile,
    OptimizationMetrics
)

__all__ = [
    'AdaptiveResponseOptimizer',
    'UserFeedback',
    'ResponseAdjustment',
    'InteractionPattern',
    'PersonalizationProfile',
    'OptimizationMetrics'
]

