# Optimizer Service

The Optimizer Service enables self-learning capabilities for agents within the AImPact platform.

## Overview

The Optimizer implements a feedback-driven learning system that continuously improves agent performance based on user interactions and explicit feedback.

## Architecture

The Optimizer uses a Proximal Policy Optimization (PPO) approach to fine-tune language models based on collected feedback.

## Core Components

### FeedbackCollector

Collects and processes user feedback on agent interactions.

```python
class FeedbackCollector:
    def record_feedback(tenant_id, session_id, interaction_id, feedback_type, score, comments=None)
    def get_feedback_stats(tenant_id, agent_id, time_period=None)
    def export_feedback_dataset(tenant_id, agent_id, format="jsonl")
```

### ModelOptimizer

Handles the training process to improve agent performance.

```python
class ModelOptimizer:
    def schedule_optimization(tenant_id, agent_id, config={})
    def get_optimization_status(tenant_id, job_id)
    def cancel_optimization(tenant_id, job_id)
```

### VersionManager

Manages multiple versions of optimized agents.

```python
class VersionManager:
    def create_version(tenant_id, agent_id, model_data, metadata)
    def list_versions(tenant_id, agent_id)
    def activate_version(tenant_id, agent_id, version_id)
    def rollback_version(tenant_id, agent_id, version_id)
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/optimizer/feedback` | POST | Record user feedback |
| `/api/optimizer/feedback/{agent_id}` | GET | Get feedback statistics |
| `/api/optimizer/jobs` | POST | Schedule optimization job |
| `/api/optimizer/jobs/{job_id}` | GET | Get job status |
| `/api/optimizer/jobs/{job_id}` | DELETE | Cancel job |
| `/api/optimizer/versions/{agent_id}` | GET | List agent versions |
| `/api/optimizer/versions/{agent_id}/{version_id}/activate` | POST | Activate version |

## Configuration

The Optimizer Service can be configured via environment variables:

```
OPTIMIZER_WORKER_COUNT=2
OPTIMIZER_BATCH_SIZE=16
OPTIMIZER_LEARNING_RATE=2e-5
MIN_FEEDBACK_THRESHOLD=100
OPTIMIZATION_TIMEOUT=3600
```

## Integration Points

- Integrated with Agent Service for model versioning
- Uses Memory Service to retrieve historical contexts
- Provides data to the Recommendation Engine for improvement suggestions

## Performance Considerations

Optimization is a resource-intensive process. The service implements:
- Job queuing for high-demand periods
- Resource limiting per tenant
- Incremental training to build on previous optimizations

