"""
Prometheus Metrics for AImpact Platform

This module defines centralized Prometheus metrics used across all services
in the AImpact platform. It provides:

1. HTTP request metrics (counts, latencies, errors)
2. Workflow execution metrics
3. Agent performance metrics
4. Voice processing metrics

Usage:
    from backend.monitoring.metrics import REGISTRY, HTTP_REQUESTS_TOTAL
    
    # Increment counter
    HTTP_REQUESTS_TOTAL.labels(method="GET", path="/api/v1/status", status="200").inc()
"""
import time
from typing import Callable, Dict, List, Optional, Union
import logging

from prometheus_client import Counter, Histogram, Gauge, Summary, Info, REGISTRY
from prometheus_client import start_http_server, multiprocess, CollectorRegistry

# Configure logging
logger = logging.getLogger("backend.monitoring.metrics")

# Constants
NAMESPACE = "aimpact"  # Global namespace for all metrics
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0)
DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)

# ---------------------- HTTP Metrics ----------------------

# Count of HTTP requests
HTTP_REQUESTS_TOTAL = Counter(
    name=f"{NAMESPACE}_http_requests_total",
    documentation="Total number of HTTP requests by method, path, and status",
    labelnames=["method", "path", "status", "tenant"]
)

# HTTP request latency
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_http_request_duration_seconds",
    documentation="HTTP request latency in seconds by method and path",
    labelnames=["method", "path", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Current in-flight HTTP requests
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    name=f"{NAMESPACE}_http_requests_in_progress",
    documentation="Current number of HTTP requests in progress by method and path",
    labelnames=["method", "path", "tenant"]
)

# HTTP error rates
HTTP_REQUEST_ERRORS_TOTAL = Counter(
    name=f"{NAMESPACE}_http_request_errors_total",
    documentation="Total number of HTTP request errors by method, path, and error type",
    labelnames=["method", "path", "error", "tenant"]
)

# ---------------------- Workflow Metrics ----------------------

# Workflow executions
WORKFLOW_EXECUTIONS_TOTAL = Counter(
    name=f"{NAMESPACE}_workflow_executions_total",
    documentation="Total number of workflow executions by workflow_id and status",
    labelnames=["workflow_id", "status", "tenant"]
)

# Workflow execution time
WORKFLOW_EXECUTION_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_workflow_execution_duration_seconds",
    documentation="Workflow execution time in seconds by workflow_id",
    labelnames=["workflow_id", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Workflow step executions
WORKFLOW_STEP_EXECUTIONS_TOTAL = Counter(
    name=f"{NAMESPACE}_workflow_step_executions_total",
    documentation="Total number of workflow step executions by workflow_id, step_id, and status",
    labelnames=["workflow_id", "step_id", "status", "tenant"]
)

# Workflow step execution time
WORKFLOW_STEP_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_workflow_step_duration_seconds",
    documentation="Workflow step execution time in seconds by workflow_id and step_id",
    labelnames=["workflow_id", "step_id", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Current in-progress workflows
WORKFLOWS_IN_PROGRESS = Gauge(
    name=f"{NAMESPACE}_workflows_in_progress",
    documentation="Current number of workflows in progress by status",
    labelnames=["status", "tenant"]
)

# ---------------------- Agent Metrics ----------------------

# Agent executions
AGENT_EXECUTIONS_TOTAL = Counter(
    name=f"{NAMESPACE}_agent_executions_total",
    documentation="Total number of agent executions by agent_id and status",
    labelnames=["agent_id", "status", "tenant"]
)

# Agent execution time
AGENT_EXECUTION_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_agent_execution_duration_seconds",
    documentation="Agent execution time in seconds by agent_id",
    labelnames=["agent_id", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Agent errors
AGENT_ERRORS_TOTAL = Counter(
    name=f"{NAMESPACE}_agent_errors_total",
    documentation="Total number of agent errors by agent_id and error type",
    labelnames=["agent_id", "error", "tenant"]
)

# Agent token usage
AGENT_TOKEN_USAGE_TOTAL = Counter(
    name=f"{NAMESPACE}_agent_token_usage_total",
    documentation="Total number of tokens used by agents",
    labelnames=["agent_id", "model", "tenant"]
)

# ---------------------- Voice Processing Metrics ----------------------

# Voice transcription requests
VOICE_TRANSCRIPTIONS_TOTAL = Counter(
    name=f"{NAMESPACE}_voice_transcriptions_total",
    documentation="Total number of voice transcription requests by provider and status",
    labelnames=["provider", "status", "tenant"]
)

# Voice transcription duration
VOICE_TRANSCRIPTION_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_voice_transcription_duration_seconds",
    documentation="Voice transcription processing time in seconds by provider",
    labelnames=["provider", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Voice synthesis requests
VOICE_SYNTHESIS_TOTAL = Counter(
    name=f"{NAMESPACE}_voice_synthesis_total",
    documentation="Total number of voice synthesis requests by provider and status",
    labelnames=["provider", "status", "tenant"]
)

# Voice synthesis duration
VOICE_SYNTHESIS_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_voice_synthesis_duration_seconds",
    documentation="Voice synthesis processing time in seconds by provider",
    labelnames=["provider", "tenant"],
    buckets=DEFAULT_BUCKETS
)

# Voice processing audio duration
VOICE_AUDIO_DURATION_SECONDS = Histogram(
    name=f"{NAMESPACE}_voice_audio_duration_seconds",
    documentation="Duration of processed audio in seconds",
    labelnames=["operation", "tenant"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

# ---------------------- Resource Usage Metrics ----------------------

# Memory usage by service
SERVICE_MEMORY_USAGE_BYTES = Gauge(
    name=f"{NAMESPACE}_service_memory_usage_bytes",
    documentation="Current memory usage of the service in bytes",
    labelnames=["service"]
)

# CPU usage by service
SERVICE_CPU_USAGE_PERCENT = Gauge(
    name=f"{NAMESPACE}_service_cpu_usage_percent",
    documentation="Current CPU usage of the service as a percentage",
    labelnames=["service"]
)

# Database connection pool size
DB_CONNECTION_POOL_SIZE = Gauge(
    name=f"{NAMESPACE}_db_connection_pool_size",
    documentation="Current size of database connection pool",
    labelnames=["database", "status"]  # status: in-use, idle, etc.
)

# Redis operations
REDIS_OPERATIONS_TOTAL = Counter(
    name=f"{NAMESPACE}_redis_operations_total",
    documentation="Total number of Redis operations by operation type",
    labelnames=["operation", "status"]
)

# ---------------------- API Quota Metrics ----------------------

# Tenant API usage
TENANT_API_REQUESTS_TOTAL = Counter(
    name=f"{NAMESPACE}_tenant_api_requests_total",
    documentation="Total number of API requests by tenant",
    labelnames=["tenant", "endpoint"]
)

# Tenant API quota remaining
TENANT_API_QUOTA_REMAINING = Gauge(
    name=f"{NAMESPACE}_tenant_api_quota_remaining",
    documentation="Remaining API quota for tenant",
    labelnames=["tenant", "plan"]
)

# Rate limited requests
RATE_LIMITED_REQUESTS_TOTAL = Counter(
    name=f"{NAMESPACE}_rate_limited_requests_total",
    documentation="Total number of rate-limited requests by tenant and endpoint",
    labelnames=["tenant", "endpoint"]
)

# ---------------------- Utility Functions ----------------------

def setup_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server on the specified port.
    
    Args:
        port: Port to expose metrics on (default: 8000)
    """
    try:
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")


class HttpMetricsMiddleware:
    """
    FastAPI middleware to automatically record HTTP request metrics.
    
    Usage:
        app = FastAPI()
        app.add_middleware(HttpMetricsMiddleware)
    """
    
    async def __call__(self, request, call_next):
        # Extract path and method
        path = request.url.path
        method = request.method
        tenant_id = getattr(request.state, "tenant_id", "unknown")
        
        # Skip metrics endpoint itself to avoid infinite recursion
        if path == "/metrics":
            return await call_next(request)
        
        # Track in-flight requests
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path, tenant=tenant_id).inc()
        
        # Time the request
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Record metrics
            status = str(response.status_code)
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status, tenant=tenant_id).inc()
            
            # Record latency
            duration = time.time() - start_time
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path, tenant=tenant_id).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            HTTP_REQUEST_ERRORS_TOTAL.labels(
                method=method, 
                path=path, 
                error=type(e).__name__,
                tenant=tenant_id
            ).inc()
            raise
            
        finally:
            # Decrement in-flight counter
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path, tenant=tenant_id).dec()


# Context manager for timing blocks of code and recording as metrics
class TimedOperation:
    """
    Context manager for timing operations and recording metrics.
    
    Usage:
        with TimedOperation(WORKFLOW_EXECUTION_DURATION_SECONDS, {"workflow_id": "123"}):
            # code to time
    """
    
    def __init__(self, metric: Histogram, labels: Dict[str, str]):
        self.metric = metric
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metric.labels(**self.labels).observe(duration)


# Function to time a specific function and record metrics
def timed_function(metric: Histogram, labels: Dict[str, str]):
    """
    Decorator for timing functions and recording metrics.
    
    Usage:
        @timed_function(WORKFLOW_EXECUTION_DURATION_SECONDS, {"workflow_id": "123"})
        def my_function():
            # code to time
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimedOperation(metric, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator

