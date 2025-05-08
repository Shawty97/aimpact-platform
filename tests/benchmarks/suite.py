"""
Comprehensive Performance Benchmark Suite for AImPact Platform.

This module contains benchmarks for:
- Memory Service
- Optimizer Service
- Recommendation Engine

Usage:
    python -m tests.benchmarks.suite --component memory
    python -m tests.benchmarks.suite --component optimizer
    python -m tests.benchmarks.suite --component recommendation
    python -m tests.benchmarks.suite --component all
"""

import asyncio
import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import random
import statistics
import sys
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import psutil
import pytest
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_results.log")],
)

logger = logging.getLogger("benchmark")

# Constants
BENCHMARK_ITERATIONS = int(os.environ.get("BENCHMARK_ITERATIONS", 100))
CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]
DATA_SIZES = [1, 10, 100, 1000, 10000]  # KB
VECTOR_DIMENSIONS = [128, 384, 768, 1536]
RESULT_DIR = "benchmark-results"

# Ensure results directory exists
os.makedirs(RESULT_DIR, exist_ok=True)


class BenchmarkTimer:
    """Context manager for timing code execution."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.cpu_percent = None
        self.memory_usage = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.cpu_percent = psutil.cpu_percent()
        self.memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        logger.info(
            f"{self.name} completed in {self.duration:.4f}s (CPU: {self.cpu_percent}%, Memory: {self.memory_usage:.2f}MB)"
        )


def benchmark(iterations: int = None, warmup: int = 3):
    """Decorator to benchmark a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal iterations
            if iterations is None:
                iterations = BENCHMARK_ITERATIONS
            
            # Warmup runs
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Actual benchmark
            durations = []
            cpu_usage = []
            memory_usage = []
            
            for i in tqdm(range(iterations), desc=f"Benchmarking {func.__name__}"):
                process = psutil.Process(os.getpid())
                start_mem = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.time()
                start_cpu = psutil.cpu_percent()
                
                result = func(*args, **kwargs)
                
                end_cpu = psutil.cpu_percent()
                end_time = time.time()
                end_mem = process.memory_info().rss / 1024 / 1024  # MB
                
                durations.append(end_time - start_time)
                cpu_usage.append((start_cpu + end_cpu) / 2)
                memory_usage.append(end_mem - start_mem)
            
            # Calculate statistics
            avg_duration = statistics.mean(durations)
            p95_duration = np.percentile(durations, 95)
            p99_duration = np.percentile(durations, 99)
            avg_cpu = statistics.mean(cpu_usage)
            avg_memory = statistics.mean(memory_usage)
            
            benchmark_result = {
                "function": func.__name__,
                "iterations": iterations,
                "average_duration_ms": avg_duration * 1000,
                "p95_duration_ms": p95_duration * 1000,
                "p99_duration_ms": p99_duration * 1000,
                "min_duration_ms": min(durations) * 1000,
                "max_duration_ms": max(durations) * 1000,
                "std_dev_ms": statistics.stdev(durations) * 1000,
                "average_cpu_percent": avg_cpu,
                "average_memory_mb": avg_memory,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Save results
            result_file = os.path.join(RESULT_DIR, f"{func.__name__}_results.json")
            with open(result_file, "w") as f:
                json.dump(benchmark_result, f, indent=2)
            
            # Plot performance
            plt.figure(figsize=(10, 6))
            plt.plot(durations)
            plt.title(f"Performance of {func.__name__}")
            plt.xlabel("Iteration")
            plt.ylabel("Duration (seconds)")
            plt.savefig(os.path.join(RESULT_DIR, f"{func.__name__}_performance.png"))
            plt.close()
            
            logger.info(f"Benchmark results for {func.__name__}:")
            logger.info(f"  Average duration: {avg_duration * 1000:.2f}ms")
            logger.info(f"  P95 duration: {p95_duration * 1000:.2f}ms")
            logger.info(f"  P99 duration: {p99_duration * 1000:.2f}ms")
            logger.info(f"  Average CPU usage: {avg_cpu:.2f}%")
            logger.info(f"  Average memory usage: {avg_memory:.2f}MB")
            
            return result
        
        return wrapper
    
    return decorator


class MemoryServiceBenchmarks:
    """Benchmarks for the Memory Service."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.headers = {"Content-Type": "application/json"}
        self.tenant_id = "benchmark-tenant"
        self.session_id = f"benchmark-session-{int(time.time())}"
    
    def _generate_random_data(self, size_kb: int) -> Dict[str, Any]:
        """Generate random data of specified size."""
        # Create a dictionary with random string data
        data = {
            "text": "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=size_kb * 1024)),
            "metadata": {
                "source": "benchmark",
                "timestamp": datetime.now().isoformat(),
                "tags": [f"tag-{i}" for i in range(10)],
            },
        }
        return data
    
    def _generate_random_vector(self, dimensions: int) -> List[float]:
        """Generate a random vector of specified dimensions."""
        return [random.random() for _ in range(dimensions)].tolist()
    
    @benchmark()
    def benchmark_context_write(self, data_size_kb: int = 10) -> Dict[str, Any]:
        """Benchmark writing context to memory service."""
        data = self._generate_random_data(data_size_kb)
        url = f"{self.api_base_url}/api/memory/context"
        
        payload = {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "context_type": "conversation",
            "data": data,
            "ttl": 3600,
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            logger.error(f"Failed to write context: {response.text}")
        
        return response.json() if response.status_code == 200 else None
    
    @benchmark()
    def benchmark_context_read(self, context_id: str) -> Dict[str, Any]:
        """Benchmark reading context from memory service."""
        url = f"{self.api_base_url}/api/memory/context/{context_id}"
        
        params = {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to read context: {response.text}")
        
        return response.json() if response.status_code == 200 else None
    
    @benchmark()
    def benchmark_vector_search(self, dimensions: int = 768, k: int = 5) -> Dict[str, Any]:
        """Benchmark vector search in memory service."""
        query_vector = self._generate_random_vector(dimensions)
        url = f"{self.api_base_url}/api/memory/search"
        
        payload = {
            "tenant_id": self.tenant_id,
            "query_vector": query_vector,
            "filters": {
                "session_id": self.session_id,
                "context_type": "conversation",
            },
            "k": k,
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            logger.error(f"Failed to perform vector search: {response.text}")
        
        return response.json() if response.status_code == 200 else None
    
    async def _concurrent_write(self, session, data_size_kb: int, i: int) -> Dict[str, Any]:
        """Helper for concurrent write test."""
        data = self._generate_random_data(data_size_kb)
        url = f"{self.api_base_url}/api/memory/context"
        
        payload = {
            "tenant_id": self.tenant_id,
            "session_id": f"{self.session_id}-{i}",
            "context_type": "conversation",
            "data": data,
            "ttl": 3600,
        }
        
        async with session.post(url, headers=self.headers, json=payload) as response:
            return await response.json() if response.status == 200 else None
    
    @benchmark(iterations=5)  # Reduced iterations due to concurrency
    def benchmark_concurrent_access(self, concurrency: int = 25, data_size_kb: int = 10) -> List[Dict[str, Any]]:
        """Benchmark concurrent access to memory service."""
        async def run_concurrent():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._concurrent_write(session, data_size_kb, i)
                    for i in range(concurrency)
                ]
                return await asyncio.gather(*tasks)
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(run_concurrent())
        
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Concurrent access: {success_count}/{concurrency} operations succeeded")
        
        return results
    
    @benchmark(iterations=5)  # Reduced iterations due to scale
    def benchmark_memory_scaling(self, data_size_kb: int = 10, num_contexts: int = 100) -> Dict[str, Any]:
        """Benchmark memory scaling with large number of contexts."""
        contexts = []
        
        with tqdm(total=num_contexts, desc="Writing contexts") as pbar:
            for i in range(num_contexts):
                data = self._generate_random_data(data_size_kb)
                url = f"{self.api_base_url}/api/memory/context"
                
                payload = {
                    "tenant_id": self.tenant_id,
                    "session_id": f"{self.session_id}-scaling-{i}",
                    "context_type": "conversation",
                    "data": data,
                    "ttl": 3600,
                }
                
                response = requests.post(url, headers=self.headers, json=payload)
                if response.status_code == 200:
                    contexts.append(response.json())
                pbar.update(1)
        
        # Now measure query performance with many contexts
        url = f"{self.api_base_url}/api/memory/search"
        query_vector = self._generate_random_vector(768)
        
        payload = {
            "tenant_id": self.tenant_id,
            "query_vector": query_vector,
            "filters": {
                "session_id": {"$regex": f"{self.session_id}-scaling-.*"},
                "context_type": "conversation",
            },
            "k": 10,
        }
        
        start_time = time.time()
        response = requests.post(url, headers=self.headers, json=payload)
        query_time = time.time() - start_time
        
        result = {
            "num_contexts": num_contexts,
            "query_time_ms": query_time * 1000,
            "results_count": len(response.json().get("results", [])) if response.status_code == 200 else 0,
        }
        
        logger.info(f"Memory scaling: Query over {num_contexts} contexts took {query_time * 1000:.2f}ms")
        
        return result
    
    def run_all(self):
        """Run all memory service benchmarks."""
        logger.info("Running Memory Service benchmarks...")
        
        # Basic read/write latency
        for size in [1, 10, 100]:
            logger.info(f"Testing with data size: {size}KB")
            context = self.benchmark_context_write(data_size_kb=size)
            if context and "context_id" in context:
                self.benchmark_context_read(context["context_id"])
        
        # Vector search performance
        for dim in [384, 768]:
            logger.info(f"Testing with vector dimensions: {dim}")
            self.benchmark_vector_search(dimensions=dim)
        
        # Concurrent access
        for concurrency in [5, 25, 50]:
            logger.info(f"Testing with concurrency: {concurrency}")
            self.benchmark_concurrent_access(concurrency=concurrency)
        
        # Memory scaling tests
        for num_contexts in [10, 50, 100]:
            logger.info(f"Testing memory scaling with {num_contexts} contexts")
            self.benchmark_memory_scaling(num_contexts=num_contexts)
        
        logger.info("Memory Service benchmarks completed")
        return True


class OptimizerServiceBenchmarks:
    """Benchmarks for the Optimizer Service."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.headers = {"Content-Type": "application/json"}
        self.tenant_id = "benchmark-tenant"
        self.agent_id = f"benchmark-agent-{int(time.time())}"
    
    def _generate_training_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic training data for optimization."""
        prompts = [
            "How can I improve my workflow?",
            "What's the best way to analyze this data?",
            "Can you help me with this financial model?",
            "How do I optimize this process?",
            "What's the most efficient approach here?",
        ]
        
        feedback_types = ["thumbs_up", "thumbs_down", "detailed"]
        scores = [1, 2, 3, 4, 5]
        
        data = []
        for i in range(num_samples):
            sample = {
                "interaction_id": f"interaction-{i}",
                "prompt": random.choice(prompts),
                "response": f"This is a synthetic response for benchmark testing, sample {i}",
                "feedback_type": random.choice(feedback_types),
                "score": random.choice(scores),
                "comments": f"Synthetic feedback comment {i}" if random.random() > 0.7 else None,
                "timestamp": (datetime.now() - pd.Timedelta(days=random.randint(0, 30))).isoformat(),
            }
            data.append(sample)
        
        return data
    
    @benchmark(iterations=3)  # Reduced due to longer execution time
    def benchmark_training_performance(self, num_samples: int = 100, epochs: int = 5) -> Dict[str, Any]:
        """Benchmark training performance of the optimizer."""
        training_data = self._generate_training_data(num_samples)
        url = f"{self.api_base_url}/api/optimizer/jobs"
        
        # First, upload training data
        upload_url = f"{self.api_base_url}/api/optimizer/feedback/batch"
        upload_payload = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "feedback_data": training_data,
        }
        
        upload_response = requests.post(upload_url, headers=self.headers, json=upload_payload)
        if upload_response.status_code != 200:
            logger.error(f"Failed to upload training data: {upload_response.text}")
            return None
        
        # Then, schedule an optimization job
        job_payload = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "config": {
                "epochs": epochs,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "max_seq_length": 512,
                "warmup_steps": 100,
                "weight_decay": 0.01,
            },
        }
        
        job_response = requests.post(url, headers=self.headers, json=job_payload)
        if job_response.status_code != 200:
            logger.error(f"Failed to schedule optimization job: {job_response.text}")
            return None
        
        job_data = job_response.json()
        job_id = job_data.get("job_id")
        
        if not job_id:
            logger.error("No job ID returned")
            return None
        
        # Poll for job completion
        status_url = f"{self.api_base_url}/api/optimizer/jobs/{job_id}"
        start_time = time.time()
        completed = False
        poll_count = 0
        
        while not completed and time.time() - start_time < 600:  # 10-minute timeout
            poll_count += 1
            time.sleep(5)  # Poll every 5 seconds
            
            status_response = requests.get(status_url, headers=self.headers)
            if status_response.status_code != 200:
                logger.error(f"Failed to get job status: {status_response.text}")
                continue
            
            status_data = status_response.json()
            status = status_data.get("status")
            
            if status == "completed":
                completed = True
                break
            elif status == "failed":
                logger.error(f"Job failed: {status_data.get('error')}")
                return None
            
            logger.info(f"Job status: {status}, progress: {status_data.get('progress', 0):.2f}%")
        
        total_time = time.time() - start_time
        
        if not completed:
            logger.error("Job timed out")
            # Cancel the job
            requests.delete(status_url, headers=self.headers)
            return None
        
        result = {
            "job_id": job_id,
            "num_samples": num_samples,
            "epochs": epochs,
            "total_time_seconds": total_time,
            "time_per_sample_ms": (total_time * 1000) / (num_samples * epochs),
            "poll_count": poll_count,
            "metrics": status_data.get("metrics", {}),
        }
        
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Time per sample: {result['time_per_sample_ms']:.2f}ms")
        
        return result
    
    @benchmark()
    def benchmark_feedback_processing(self, num_feedbacks: int = 100) -> Dict[str, Any]:
        """Benchmark feedback processing performance."""
        feedbacks = []
        url = f"{self.api_base_url}/api/optimizer/feedback"
        
        # Generate and send feedback in batches
        batch_size = 10
        num_batches = num_feedbacks // batch_size + (1 if num_feedbacks % batch_size > 0 else 0)
        
        for batch in range(num_batches):
            batch_feedbacks = []
            for i in range(batch_size):
                idx = batch * batch_size + i
                if idx >= num_feedbacks:
                    break
                    
                feedback = {
                    "tenant_id": self.tenant_id,
                    "agent_id": self.agent_id,
                    "session_id": f"benchmark-session-{batch}-{i}",
                    "interaction_id": f"interaction-{idx}",
                    "feedback_type": random.choice(["thumbs_up", "thumbs_down", "detailed"]),
                    "score": random.randint(1, 5),
                    "comments": f"Benchmark feedback {idx}" if random.random() > 0.7 else None,
                }
                batch_feedbacks.append(feedback)
            
            # Send batch
            batch_url = f"{self.api_base_url}/api/optimizer/feedback/batch"
            payload = {
                "tenant_id": self.tenant_id,
                "agent_id": self.agent_id,
                "feedback_data": batch_feedbacks,
            }
            
            response = requests.post(batch_url, headers=self.headers, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to process feedback batch: {response.text}")
            else:
                feedbacks.extend(batch_feedbacks)
        
        # Now retrieve stats to measure processing performance
        stats_url = f"{self.api_base_url}/api/optimizer/feedback/{self.agent_id}"
        params = {"tenant_id": self.tenant_id}
        
        stats_response = requests.get(stats_url, headers=self.headers, params=params)
        if stats_response.status_code != 200:
            logger.error(f"Failed to get feedback stats: {stats_response.text}")
            return None
        
        stats = stats_response.json()
        
        result = {
            "num_feedbacks": num_feedbacks,
            "processed_count": stats.get("total_feedback", 0),
            "processing_success_rate": stats.get("total_feedback", 0) / num_feedbacks if num_feedbacks > 0 else 0,
            "feedback_stats": stats,
        }
        
        logger.info(f"Processed {result['processed_count']}/{num_feedbacks} feedback items")
        logger.info(f"Success rate: {result['processing_success_rate'] * 100:.2f}%")
        
        return result
    
    @benchmark()
    def benchmark_model_versioning(self, num_versions: int = 5) -> Dict[str, Any]:
        """Benchmark model versioning operations."""
        versions = []
        
        # First, create multiple versions
        for i in range(num_versions):
            version_url = f"{self.api_base_url}/api/optimizer/versions/{self.agent_id}"
            
            # Generate a minimal model representation (would be larger in practice)
            model_data = {
                "version": f"v{i+1}",
                "parameters": {
                    "param1": random.random(),
                    "param2": random.random(),
                    "param3": random.random(),
                },
                "config": {
                    "batch_size": 16,
                    "learning_rate": 2e-5,
                    "epochs": 3,
                },
            }
            
            metadata = {
                "created_at": datetime.now().isoformat(),
                "feedback_count": (i + 1) * 100,
                "performance_metrics": {
                    "accuracy": 0.8 + random.random() * 0.1,
                    "loss": 0.2 - random.random() * 0.1,
                },
            }
            
            payload = {
                "tenant_id": self.tenant_id,
                "agent_id": self.agent_id,
                "model_data": model_data,
                "metadata": metadata,
            }
            
            response = requests.post(version_url, headers=self.headers, json=payload)
            if response.status_code != 200:
                logger.error(f"Failed to create version: {response.text}")
                continue
                
            version_data = response.json()
            versions.append(version_data)
            time.sleep(1)  # Small delay between version creations
        
        # Get the list of versions
        list_url = f"{self.api_base_url}/api/optimizer/versions/{self.agent_id}"
        params = {"tenant_id": self.tenant_id}
        
        list_response = requests.get(list_url, headers=self.headers, params=params)
        if list_response.status_code != 200:
            logger.error(f"Failed to list versions: {list_response.text}")
            return None
            
        version_list = list_response.json().get("versions", [])
        
        # Perform version activation (if we have versions)
        activation_timings = []
        if versions:
            for version in versions[:min(3, len(versions))]:  # Try activating up to 3 versions
                version_id = version.get("version_id")
                if not version_id:
                    continue
                    
                activate_url = f"{self.api_base_url}/api/optimizer/versions/{self.agent_id}/{version_id}/activate"
                
                start_time = time.time()
                activate_response = requests.post(activate_url, headers=self.headers, json={"tenant_id": self.tenant_id})
                activation_time = time.time() - start_time
                
                activation_timings.append(activation_time)
                if activate_response.status_code != 200:
                    logger.error(f"Failed to activate version: {activate_response.text}")
                
                time.sleep(1)  # Small delay between activations
        
        result = {
            "num_versions_created": len(versions),
            "num_versions_listed": len(version_list),
            "creation_success_rate": len(versions) / num_versions if num_versions > 0 else 0,
            "average_activation_time_ms": statistics.mean(activation_timings) * 1000 if activation_timings else 0,
            "activation_success_rate": sum(1 for v in versions[:min(3, len(versions))] if v.get("activated", False)) / min(3, len(versions)) if versions else 0,
        }
        
        logger.info(f"Created {len(versions)}/{num_versions} model versions")
        logger.info(f"Average activation time: {result.get('average_activation_time_ms', 0):.2f}ms")
        
        return result
    
    @benchmark()
    def benchmark_resource_utilization(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark resource utilization during optimization."""
        training_data = self._generate_training_data(num_samples=200)  # Larger dataset for stress testing
        
        # Upload training data
        upload_url = f"{self.api_base_url}/api/optimizer/feedback/batch"
        upload_payload = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "feedback_data": training_data,
        }
        
        requests.post(upload_url, headers=self.headers, json=upload_payload)
        
        # Start a resource-intensive optimization job
        job_url = f"{self.api_base_url}/api/optimizer/jobs"
        job_payload = {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "config": {
                "epochs": 10,
                "learning_rate": 2e-5,
                "batch_size": 32,
                "max_seq_length": 1024,  # Larger sequence length for resource stress
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "priority": "high",
            },
        }
        
        job_response = requests.post(job_url, headers=self.headers, json=job_payload)
        if job_response.status_code != 200:
            logger.error(f"Failed to start optimization job: {job_response.text}")
            return None
            
        job_data = job_response.json()
        job_id = job_data.get("job_id")
        
        if not job_id:
            logger.error("No job ID returned")
            return None
        
        # Monitor resource utilization
        resource_metrics = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        metrics_url = f"{self.api_base_url}/api/optimizer/metrics"
        params = {
            "tenant_id": self.tenant_id,
            "job_id": job_id,
        }
        
        with tqdm(total=duration_seconds, desc="Monitoring resources") as pbar:
            while time.time() < end_time:
                metrics_response = requests.get(metrics_url, headers=self.headers, params=params)
                
                if metrics_response.status_code == 200:
                    metrics = metrics_response.json()
                    resource_metrics.append({
                        "timestamp": time.time(),
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "memory_mb": metrics.get("memory_mb", 0),
                        "gpu_utilization": metrics.get("gpu_utilization", 0),
                        "gpu_memory_mb": metrics.get("gpu_memory_mb", 0),
                        "disk_io_mb_s": metrics.get("disk_io_mb_s", 0),
                        "network_io_mb_s": metrics.get("network_io_mb_s", 0),
                    })
                
                elapsed = time.time() - start_time
                pbar.update(min(int(elapsed), duration_seconds) - pbar.n)
                
                time.sleep(1)  # Sample every second
        
        # Cancel the job to avoid long-running processes
        cancel_url = f"{self.api_base_url}/api/optimizer/jobs/{job_id}"
        requests.delete(cancel_url, headers=self.headers)
        
        # Calculate resource utilization statistics
        if not resource_metrics:
            logger.error("No resource metrics collected")
            return None
            
        cpu_percentages = [m["cpu_percent"] for m in resource_metrics]
        memory_mbs = [m["memory_mb"] for m in resource_metrics]
        gpu_utilizations = [m["gpu_utilization"] for m in resource_metrics]
        gpu_memory_mbs = [m["gpu_memory_mb"] for m in resource_metrics]
        
        result = {
            "job_id": job_id,
            "duration_seconds": duration_seconds,
            "sample_count": len(resource_metrics),
            "average_cpu_percent": statistics.mean(cpu_percentages),
            "peak_cpu_percent": max(cpu_percentages),
            "average_memory_mb": statistics.mean(memory_mbs),
            "peak_memory_mb": max(memory_mbs),
            "average_gpu_utilization": statistics.mean(gpu_utilizations) if any(gpu_utilizations) else 0,
            "peak_gpu_utilization": max(gpu_utilizations) if any(gpu_utilizations) else 0,
            "average_gpu_memory_mb": statistics.mean(gpu_memory_mbs) if any(gpu_memory_mbs) else 0,
            "peak_gpu_memory_mb": max(gpu_memory_mbs) if any(gpu_memory_mbs) else 0,
        }
        
        logger.info(f"Resource utilization: Avg CPU {result['average_cpu_percent']:.2f}%, Peak CPU {result['peak_cpu_percent']:.2f}%")
        logger.info(f"Resource utilization: Avg Memory {result['average_memory_mb']:.2f}MB, Peak Memory {result['peak_memory_mb']:.2f}MB")
        if any(gpu_utilizations):
            logger.info(f"Resource utilization: Avg GPU {result['average_gpu_utilization']:.2f}%, Peak GPU {result['peak_gpu_utilization']:.2f}%")
            logger.info(f"Resource utilization: Avg GPU Memory {result['average_gpu_memory_mb']:.2f}MB, Peak GPU Memory {result['peak_gpu_memory_mb']:.2f}MB")
        
        return result
    
    @benchmark(iterations=2)  # Limited due to complexity
    def benchmark_multi_tenant(self, num_tenants: int = 5, num_samples_per_tenant: int = 50) -> Dict[str, Any]:
        """Benchmark multi-tenant optimization performance."""
        tenant_results = {}
        tenant_ids = [f"benchmark-tenant-{i}" for i in range(num_tenants)]
        agent_ids = [f"benchmark-agent-{i}-{int(time.time())}" for i in range(num_tenants)]
        
        # Create training data for each tenant
        for i, (tenant_id, agent_id) in enumerate(zip(tenant_ids, agent_ids)):
            training_data = self._generate_training_data(num_samples=num_samples_per_tenant)
            
            # Upload training data
            upload_url = f"{self.api_base_url}/api/optimizer/feedback/batch"
            upload_payload = {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "feedback_data": training_data,
            }
            
            upload_response = requests.post(upload_url, headers=self.headers, json=upload_payload)
            if upload_response.status_code != 200:
                logger.error(f"Failed to upload training data for tenant {tenant_id}: {upload_response.text}")
                continue
        
        # Start optimization jobs for all tenants concurrently
        job_ids = []
        start_time = time.time()
        
        for i, (tenant_id, agent_id) in enumerate(zip(tenant_ids, agent_ids)):
            job_url = f"{self.api_base_url}/api/optimizer/jobs"
            job_payload = {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "config": {
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "max_seq_length": 512,
                    "warmup_steps": 50,
                    "weight_decay": 0.01,
                },
            }
            
            job_response = requests.post(job_url, headers=self.headers, json=job_payload)
            if job_response.status_code != 200:
                logger.error(f"Failed to start optimization job for tenant {tenant_id}: {job_response.text}")
                continue
                
            job_data = job_response.json()
            job_id = job_data.get("job_id")
            
            if job_id:
                job_ids.append((tenant_id, agent_id, job_id))
                logger.info(f"Started job {job_id} for tenant {tenant_id}")
        
        # Monitor job progress
        completed_jobs = []
        max_wait_time = 600  # 10-minute timeout
        
        with tqdm(total=len(job_ids), desc="Monitoring tenant jobs") as pbar:
            while len(completed_jobs) < len(job_ids) and time.time() - start_time < max_wait_time:
                for tenant_id, agent_id, job_id in job_ids:
                    if (tenant_id, agent_id, job_id) in completed_jobs:
                        continue
                        
                    status_url = f"{self.api_base_url}/api/optimizer/jobs/{job_id}"
                    params = {"tenant_id": tenant_id}
                    
                    status_response = requests.get(status_url, headers=self.headers, params=params)
                    if status_response.status_code != 200:
                        logger.error(f"Failed to get job status for tenant {tenant_id}: {status_response.text}")
                        continue
                        
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status in ["completed", "failed"]:
                        completed_jobs.append((tenant_id, agent_id, job_id))
                        tenant_results[tenant_id] = {
                            "job_id": job_id,
                            "status": status,
                            "completion_time": time.time() - start_time,
                            "metrics": status_data.get("metrics", {}),
                        }
                        pbar.update(1)
                
                if len(completed_jobs) < len(job_ids):
                    time.sleep(5)  # Wait before checking again
        
        # Cancel any remaining jobs
        for tenant_id, agent_id, job_id in job_ids:
            if (tenant_id, agent_id, job_id) not in completed_jobs:
                cancel_url = f"{self.api_base_url}/api/optimizer/jobs/{job_id}"
                params = {"tenant_id": tenant_id}
                requests.delete(cancel_url, headers=self.headers, params=params)
                logger.info(f"Cancelled job {job_id} for tenant {tenant_id}")
        
        # Calculate multi-tenant performance metrics
        completion_times = [result["completion_time"] for result in tenant_results.values() if result["status"] == "completed"]
        
        result = {
            "num_tenants": num_tenants,
            "num_completed_jobs": len([r for r in tenant_results.values() if r["status"] == "completed"]),
            "num_failed_jobs": len([r for r in tenant_results.values() if r["status"] == "failed"]),
            "total_time_seconds": time.time() - start_time,
            "average_completion_time": statistics.mean(completion_times) if completion_times else 0,
            "min_completion_time": min(completion_times) if completion_times else 0,
            "max_completion_time": max(completion_times) if completion_times else 0,
            "tenant_results": tenant_results,
        }
        
        logger.info(f"Multi-tenant benchmark: {result['num_completed_jobs']}/{num_tenants} jobs completed")
        if completion_times:
            logger.info(f"Average completion time: {result['average_completion_time']:.2f}s")
            logger.info(f"Completion time range: {result['min_completion_time']:.2f}s - {result['max_completion_time']:.2f}s")
        
        return result
    
    def run_all(self):
        """Run all optimizer service benchmarks."""
        logger.info("Running Optimizer Service benchmarks...")
        
        # Training performance
        for num_samples in [50, 100]:
            logger.info(f"Testing training performance with {num_samples} samples")
            self.benchmark_training_performance(num_samples=num_samples, epochs=3)
        
        # Feedback processing
        for num_feedbacks in [50, 100, 200]:
            logger.info(f"Testing feedback processing with {num_feedbacks} feedback items")
            self.benchmark_feedback_processing(num_feedbacks=num_feedbacks)
        
        # Model versioning
        logger.info("Testing model versioning")
        self.benchmark_model_versioning()
        
        # Resource utilization
        logger.info("Testing resource utilization")
        self.benchmark_resource_utilization(duration_seconds=30)  # Shorter duration for benchmarks
        
        # Multi-tenant testing
        logger.info("Testing multi-tenant optimization")
        self.benchmark_multi_tenant(num_tenants=3, num_samples_per_tenant=30)  # Reduced for benchmarks
        
        logger.info("Optimizer Service benchmarks completed")
        return True


class RecommendationServiceBenchmarks:
    """Benchmarks for the Recommendation Service."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.headers = {"Content-Type": "application/json"}
        self.tenant_id = "benchmark-tenant"
        self.user_id = f"benchmark-user-{int(time.time())}"
    
    def _generate_sample_workflow(self, complexity: str = "medium") -> Dict[str, Any]:
        """Generate a synthetic workflow for benchmarking."""
        # Base workflow structure
        workflow = {
            "id": f"workflow-{int(time.time())}-{random.randint(1000, 9999)}",
            "name": f"Benchmark Workflow ({complexity})",
            "description": f"A {complexity} complexity workflow for benchmarking",
            "version": "1.0",
            "nodes": [],
            "edges": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "created_by": self.user_id,
                "tags": ["benchmark", complexity],
            }
        }
        
        # Node types
        node_types = ["input", "process", "decision", "output", "api_call", "transform", "llm"]
        
        # Complexity determines workflow size
        num_nodes = {
            "simple": random.randint(3, 5),
            "medium": random.randint(8, 15),
            "complex": random.randint(20, 30),
            "very_complex": random.randint(40, 60)
        }.get(complexity, 10)
        
        # Generate nodes
        for i in range(num_nodes):
            node_type = random.choice(node_types)
            node = {
                "id": f"node-{i}",
                "type": node_type,
                "name": f"{node_type.title()} Node {i}",
                "position": {"x": random.randint(0, 1000), "y": random.randint(0, 800)},
                "data": {
                    "prompt": f"Sample prompt for node {i}" if node_type == "llm" else None,
                    "api_endpoint": f"https://api.example.com/endpoint{i}" if node_type == "api_call" else None,
                    "transform_type": random.choice(["map", "filter", "reduce"]) if node_type == "transform" else None,
                    "conditions": [{"if": "x > 0", "then": "node-next"}] if node_type == "decision" else None,
                }
            }
            workflow["nodes"].append(node)
        
        # Generate edges (connections between nodes)
        # Ensure each node (except last) has at least one outgoing connection
        for i in range(num_nodes - 1):
            # Connect to a random later node
            target = random.randint(i + 1, num_nodes - 1)
            edge = {
                "id": f"edge-{i}-{target}",
                "source": f"node-{i}",
                "target": f"node-{target}",
                "type": "default"
            }
            workflow["edges"].append(edge)
            
            # Add some additional connections for more complex workflows
            if complexity in ["complex", "very_complex"] and random.random() > 0.7:
                target2 = random.randint(i + 1, num_nodes - 1)
                if target2 != target:
                    edge2 = {
                        "id": f"edge-{i}-{target2}",
                        "source": f"node-{i}",
                        "target": f"node-{target2}",
                        "type": "conditional"
                    }
                    workflow["edges"].append(edge2)
        
        return workflow
    
    @benchmark()
    def llm_inference_benchmark(self, prompt_complexity: str = "medium") -> Dict[str, Any]:
        """Benchmark LLM inference speed for recommendation generation."""
        # Generate prompts of different complexity
        prompt_lengths = {
            "simple": random.randint(50, 100),
            "medium": random.randint(200, 400),
            "complex": random.randint(600, 1000),
        }
        prompt_length = prompt_lengths.get(prompt_complexity, 300)
        
        # Create a prompt for suggestion generation
        prompt = f"Generate improvement suggestions for this workflow: {' '.join(['benchmark'] * prompt_length)}"
        
        # Generate sample workflow
        workflow = self._generate_sample_workflow(complexity=prompt_complexity)
        
        # Call the recommendations API for inference
        url = f"{self.api_base_url}/api/recommendations/analyze"
        payload = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "content_type": "workflow",
            "content_id": workflow["id"],
            "content": workflow,
            "prompt": prompt,
            "options": {
                "max_suggestions": 5,
                "suggestion_types": ["prompt", "structure", "module"],
                "detail_level": "detailed",
                "include_explanations": True,
            }
        }
        
        start_time = time.time()
        response = requests.post(url, headers=self.headers, json=payload)
        inference_time = time.time() - start_time
        
        if response.status_code != 200:
            logger.error(f"Failed to get recommendations: {response.text}")
            return None
        
        suggestions = response.json().get("suggestions", [])
        
        result = {
            "prompt_complexity": prompt_complexity,
            "prompt_length": len(prompt),
            "workflow_nodes": len(workflow["nodes"]),
            "inference_time_ms": inference_time * 1000,
            "num_suggestions": len(suggestions),
            "tokens_per_second": len(prompt) / inference_time if inference_time > 0 else 0,
        }
        
        logger.info(f"LLM inference: {result['inference_time_ms']:.2f}ms for {prompt_complexity} complexity")
        logger.info(f"Generated {result['num_suggestions']} suggestions")
        logger.info(f"Processing speed: {result['tokens_per_second']:.2f} tokens/second")
        
        return result
    
    @benchmark()
    def workflow_analysis_benchmark(self, workflow_complexity: str = "medium") -> Dict[str, Any]:
        """Benchmark workflow analysis performance."""
        workflow = self._generate_sample_workflow(complexity=workflow_complexity)
        
        # Call the workflow analysis API
        url = f"{self.api_base_url}/api/recommendations/workflows/analyze"
        payload = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "workflow": workflow,
            "analysis_type": "comprehensive",
            "options": {
                "check_logical_flow": True,
                "check_error_handling": True,
                "check_performance_issues": True,
                "check_best_practices": True,
            }
        }
        
        start_time = time.time()
        response = requests.post(url, headers=self.headers, json=payload)
        analysis_time = time.time() - start_time
        
        if response.status_code != 200:
            logger.error(f"Failed to analyze workflow: {response.text}")
            return None
        
        analysis_results = response.json()
        issues_found = analysis_results.get("issues_found", [])
        suggestions = analysis_results.get("improvement_suggestions", [])
        
        result = {
            "workflow_complexity": workflow_complexity,
            "node_count": len(workflow["nodes"]),
            "edge_count": len(workflow["edges"]),
            "analysis_time_ms": analysis_time * 1000,
            "issues_found": len(issues_found),
            "suggestions_generated": len(suggestions),
            "time_per_node_ms": (analysis_time * 1000) / len(workflow["nodes"]) if workflow["nodes"] else 0,
        }
        
        logger.info(f"Workflow analysis: {result['analysis_time_ms']:.2f}ms for {workflow_complexity} workflow")
        logger.info(f"Found {result['issues_found']} issues and generated {result['suggestions_generated']} suggestions")
        logger.info(f"Average time per node: {result['time_per_node_ms']:.2f}ms")
        
        return result
    
    @benchmark()
    def ui_responsiveness_benchmark(self, component_type: str = "recommendation_panel") -> Dict[str, Any]:
        """Benchmark UI component responsiveness."""
        # Different UI components to test
        components = {
            "recommendation_panel": "/api/ui/components/recommendation-panel",
            "suggestion_preview": "/api/ui/components/suggestion-preview",
            "workflow_comparison": "/api/ui/components/workflow-comparison",
            "make_better_button": "/api/ui/components/make-better-button",
        }
        
        component_url = components.get(component_type)
        if not component_url:
            logger.error(f"Unknown component type: {component_type}")
            return None
            
        url = f"{self.api_base_url}{component_url}"
        
        # Generate test data based on component type
        if component_type == "recommendation_panel":
            # Test loading recommendations panel with varying numbers of items
            num_recommendations = random.choice([5, 10, 20, 50])
            
            # Generate mock recommendations
            recommendations = []
            for i in range(num_recommendations):
                recommendations.append({
                    "id": f"rec-{i}",
                    "title": f"Recommendation {i}",
                    "description": f"This is a sample recommendation {i} for benchmarking",
                    "type": random.choice(["prompt", "structure", "module"]),
                    "confidence": random.random(),
                    "tags": [f"tag-{j}" for j in range(random.randint(1, 3))],
                })
                
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "recommendations": recommendations,
                "view_mode": random.choice(["list", "grid", "detailed"]),
            }
            
        elif component_type == "suggestion_preview":
            # Test suggestion preview rendering
            workflow = self._generate_sample_workflow(complexity="medium")
            suggestion = {
                "id": f"sug-{int(time.time())}",
                "title": "Add error handling",
                "description": "Add error handling to improve workflow reliability",
                "type": "structure",
                "changes": [
                    {"type": "add_node", "node_id": "error-handler", "node_type": "error_handler"},
                    {"type": "add_edge", "source": "node-1", "target": "error-handler", "condition": "on_error"},
                ]
            }
            
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "original_workflow": workflow,
                "suggestion": suggestion,
                "preview_mode": "split",
            }
            
        elif component_type == "workflow_comparison":
            # Test workflow comparison view
            original_workflow = self._generate_sample_workflow(complexity="medium")
            improved_workflow = original_workflow.copy()
            
            # Make some modifications to the improved workflow
            # Add a new node
            new_node = {
                "id": "error-handler",
                "type": "error_handler",
                "name": "Error Handler",
                "position": {"x": 500, "y": 500},
                "data": {
                    "error_types": ["api_error", "timeout", "validation_error"],
                    "retry_count": 3,
                }
            }
            improved_workflow["nodes"].append(new_node)
            
            # Add connections to the new node
            for i in range(min(3, len(improved_workflow["nodes"]) - 1)):
                edge = {
                    "id": f"err-edge-{i}",
                    "source": improved_workflow["nodes"][i]["id"],
                    "target": "error-handler",
                    "type": "error"
                }
                improved_workflow["edges"].append(edge)
            
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "original_workflow": original_workflow,
                "improved_workflow": improved_workflow,
                "comparison_mode": "side-by-side",
                "highlight_changes": True,
            }
            
        elif component_type == "make_better_button":
            # Test make better button interaction
            workflow = self._generate_sample_workflow(complexity="simple")
            
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "workflow_id": workflow["id"],
                "context": {
                    "current_view": "workflow_editor",
                    "selected_node_id": workflow["nodes"][0]["id"] if workflow["nodes"] else None,
                    "recent_actions": ["edit_node", "add_edge", "delete_node"],
                },
                "button_type": "floating",
            }
        
        # Call the UI component rendering API
        start_time = time.time()
        response = requests.post(url, headers=self.headers, json=payload)
        render_time = time.time() - start_time
        
        if response.status_code != 200:
            logger.error(f"Failed to render UI component: {response.text}")
            return None
        
        component_data = response.json()
        dom_elements = component_data.get("dom_element_count", 0)
        html_size_bytes = len(component_data.get("html", ""))
        
        # Calculate rendering metrics
        result = {
            "component_type": component_type,
            "render_time_ms": render_time * 1000,
            "dom_element_count": dom_elements,
            "html_size_bytes": html_size_bytes,
            "time_per_element_ms": (render_time * 1000) / dom_elements if dom_elements > 0 else 0,
            "render_complexity": component_data.get("render_complexity", "medium"),
        }
        
        logger.info(f"UI Component {component_type}: Rendered in {result['render_time_ms']:.2f}ms")
        logger.info(f"DOM Elements: {dom_elements}, HTML Size: {html_size_bytes/1024:.2f}KB")
        logger.info(f"Time per element: {result['time_per_element_ms']:.2f}ms")
        
        return result
    
    @benchmark()
    def suggestion_quality_benchmark(self, suggestion_type: str = "prompt") -> Dict[str, Any]:
        """Benchmark suggestion quality and relevance metrics."""
        # Create test artifacts based on suggestion type
        if suggestion_type == "prompt":
            # Test prompt improvement suggestions
            original_prompt = "Tell me about the weather."
            context = {
                "recent_conversations": [
                    {"role": "user", "content": "What's the weather like today?"},
                    {"role": "assistant", "content": "I don't have real-time weather data."},
                    {"role": "user", "content": "How can I check the weather?"},
                ],
                "user_preferences": {
                    "detail_level": "high",
                    "tone": "professional",
                    "format": "structured",
                }
            }
            
            url = f"{self.api_base_url}/api/recommendations/prompts/improve"
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "original_prompt": original_prompt,
                "context": context,
                "options": {
                    "max_suggestions": 3,
                    "improvement_aspects": ["clarity", "specificity", "context_utilization"],
                    "include_explanations": True,
                },
            }
            
        elif suggestion_type == "workflow":
            # Test workflow improvement suggestions
            workflow = self._generate_sample_workflow(complexity="medium")
            
            url = f"{self.api_base_url}/api/recommendations/workflows/improve"
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "workflow": workflow,
                "options": {
                    "max_suggestions": 3,
                    "improvement_aspects": ["error_handling", "performance", "structure"],
                    "include_explanations": True,
                },
            }
            
        elif suggestion_type == "module":
            # Test module suggestions
            current_modules = ["basic_llm", "text_analyzer"]
            use_case = "customer_support_automation"
            
            url = f"{self.api_base_url}/api/recommendations/modules/suggest"
            payload = {
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "current_modules": current_modules,
                "use_case": use_case,
                "options": {
                    "max_suggestions": 5,
                    "include_explanations": True,
                },
            }
        else:
            logger.error(f"Unknown suggestion type: {suggestion_type}")
            return None
        
        # Get suggestion quality metrics
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            logger.error(f"Failed to get suggestions: {response.text}")
            return None
            
        suggestions = response.json().get("suggestions", [])
        if not suggestions:
            logger.error("No suggestions returned")
            return None
        
        # Simulate user interaction with suggestions
        # For benchmarking, we'll use a pre-defined acceptance rate
        acceptance_rates = {
            "high_confidence": 0.85,
            "medium_confidence": 0.60,
            "low_confidence": 0.30,
        }
        
        accepted_suggestions = []
        implemented_suggestions = []
        
        for suggestion in suggestions:
            confidence = suggestion.get("confidence", 0.5)
            confidence_level = "high_confidence" if confidence >= 0.8 else ("medium_confidence" if confidence >= 0.5 else "low_confidence")
            
            # Simulate user accepting the suggestion
            if random.random() < acceptance_rates[confidence_level]:
                accepted_suggestions.append(suggestion)
                
                # Simulate implementing the suggestion
                if random.random() < 0.7:  # 70% implementation rate for accepted suggestions
                    implemented_suggestions.append(suggestion)
                    
                    # Log the implementation
                    feedback_url = f"{self.api_base_url}/api/recommendations/{suggestion['id']}/feedback"
                    feedback_payload = {
                        "tenant_id": self.tenant_id,
                        "user_id": self.user_id,
                        "action": "implemented",
                        "rating": random.randint(3, 5),  # 3-5 star rating
                        "comments": "Benchmark implementation feedback",
                    }
                    requests.post(feedback_url, headers=self.headers, json=feedback_payload)
                else:
                    # Log rejection after acceptance
                    feedback_url = f"{self.api_base_url}/api/recommendations/{suggestion['id']}/feedback"
                    feedback_payload = {
                        "tenant_id": self.tenant_id,
                        "user_id": self.user_id,
                        "action": "rejected_after_review",
                        "rating": random.randint(1, 3),  # 1-3 star rating
                        "comments": "Benchmark rejection feedback",
                    }
                    requests.post(feedback_url, headers=self.headers, json=feedback_payload)
            else:
                # Log immediate rejection
                feedback_url = f"{self.api_base_url}/api/recommendations/{suggestion['id']}/feedback"
                feedback_payload = {
                    "tenant_id": self.tenant_id,
                    "user_id": self.user_id,
                    "action": "rejected_immediately",
                    "rating": random.randint(1, 2),  # 1-2 star rating
                    "comments": "Benchmark immediate rejection",
                }
                requests.post(feedback_url, headers=self.headers, json=feedback_payload)
        
        # Calculate quality metrics
        result = {
            "suggestion_type": suggestion_type,
            "total_suggestions": len(suggestions),
            "accepted_count": len(accepted_suggestions),
            "implemented_count": len(implemented_suggestions),
            "acceptance_rate": len(accepted_suggestions) / len(suggestions) if suggestions else 0,
            "implementation_rate": len(implemented_suggestions) / len(accepted_suggestions) if accepted_suggestions else 0,
            "overall_success_rate": len(implemented_suggestions) / len(suggestions) if suggestions else 0,
            "average_confidence": statistics.mean([s.get("confidence", 0) for s in suggestions]) if suggestions else 0,
            "average_relevance": statistics.mean([s.get("relevance_score", 0) for s in suggestions]) if suggestions else 0,
        }
        
        logger.info(f"Suggestion quality ({suggestion_type}): {result['acceptance_rate']*100:.2f}% acceptance rate")
        logger.info(f"Implementation rate: {result['implementation_rate']*100:.2f}% of accepted suggestions")
        logger.info(f"Overall success rate: {result['overall_success_rate']*100:.2f}%")
        
        return result
    
    @benchmark(iterations=3)  # Reduced due to complexity
    def multi_tenant_benchmark(self, num_tenants: int = 5, requests_per_tenant: int = 10) -> Dict[str, Any]:
        """Benchmark multi-tenant recommendation performance."""
        tenant_ids = [f"benchmark-tenant-{i}" for i in range(num_tenants)]
        user_ids = [f"benchmark-user-{i}-{int(time.time())}" for i in range(num_tenants)]
        
        # Total requests counter
        total_requests = 0
        successful_requests = 0
        latencies = []
        
        # API endpoints to test
        endpoints = [
            "/api/recommendations/workflows/analyze",
            "/api/recommendations/prompts/improve",
            "/api/recommendations/modules/suggest",
        ]
        
        # Start time for overall benchmark
        start_time = time.time()
        
        # Create async tasks for parallel execution
        async def run_tenant_requests(session, tenant_id, user_id, num_requests):
            nonlocal total_requests, successful_requests, latencies
            
            for i in range(num_requests):
                # Select random endpoint
                endpoint = random.choice(endpoints)
                url = f"{self.api_base_url}{endpoint}"
                
                # Generate appropriate payload
                if "workflows/analyze" in endpoint:
                    workflow = self._generate_sample_workflow(
                        complexity=random.choice(["simple", "medium"])
                    )
                    payload = {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "workflow": workflow,
                        "analysis_type": "comprehensive",
                        "options": {
                            "check_logical_flow": True,
                            "check_error_handling": True,
                            "check_performance_issues": True,
                            "check_best_practices": True,
                        }
                    }
                elif "prompts/improve" in endpoint:
                    # For prompt improvement endpoint
                    original_prompt = f"Sample prompt for benchmarking tenant {tenant_id}, request {i}"
                    payload = {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "original_prompt": original_prompt,
                        "context": {
                            "recent_conversations": [
                                {"role": "user", "content": f"Question {j} from tenant {tenant_id}"} 
                                for j in range(3)
                            ],
                            "user_preferences": {
                                "detail_level": random.choice(["low", "medium", "high"]),
                                "tone": random.choice(["casual", "professional", "technical"]),
                            }
                        },
                        "options": {
                            "max_suggestions": 3,
                            "improvement_aspects": ["clarity", "specificity", "context_utilization"],
                            "include_explanations": True,
                        }
                    }
                else:  # modules/suggest endpoint
                    # For module suggestions endpoint
                    modules = ["basic_llm", "text_analyzer", "data_processor", "api_connector", "workflow_engine"]
                    current_modules = random.sample(modules, k=min(2, len(modules)))
                    use_cases = ["customer_support", "content_creation", "data_analysis", "document_processing"]
                    
                    payload = {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "current_modules": current_modules,
                        "use_case": random.choice(use_cases),
                        "options": {
                            "max_suggestions": 3,
                            "include_explanations": True,
                        }
                    }
                
                total_requests += 1
                
                # Make the request and measure latency
                request_start = time.time()
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        successful_requests += 1
                        response_data = await response.json()
                        request_latency = time.time() - request_start
                        latencies.append(request_latency)
                        
                        # Verify tenant isolation (responses should only contain the requesting tenant's data)
                        response_tenant = response_data.get("tenant_id")
                        if response_tenant and response_tenant != tenant_id:
                            logger.error(f"Tenant isolation breach: {tenant_id} received data for {response_tenant}")
                    else:
                        logger.error(f"Request failed: {response.status}, {await response.text()}")
        
        # Run all tenant requests concurrently
        async def run_all_tenants():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for tenant_id, user_id in zip(tenant_ids, user_ids):
                    tasks.append(run_tenant_requests(session, tenant_id, user_id, requests_per_tenant))
                await asyncio.gather(*tasks)
        
        # Execute the concurrent requests
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_all_tenants())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        result = {
            "num_tenants": num_tenants,
            "requests_per_tenant": requests_per_tenant,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "average_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000 if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000 if latencies else 0,
            "min_latency_ms": min(latencies) * 1000 if latencies else 0,
            "max_latency_ms": max(latencies) * 1000 if latencies else 0,
        }
        
        logger.info(f"Multi-tenant benchmark: {result['success_rate']*100:.2f}% success rate")
        logger.info(f"Throughput: {result['requests_per_second']:.2f} requests/second")
        logger.info(f"Average latency: {result['average_latency_ms']:.2f}ms")
        logger.info(f"P95 latency: {result['p95_latency_ms']:.2f}ms")
        
        return result
    
    def run_all(self):
        """Run all recommendation service benchmarks."""
        logger.info("Running Recommendation Service benchmarks...")
        
        # LLM inference benchmark
        for complexity in ["simple", "medium", "complex"]:
            logger.info(f"Testing LLM inference with {complexity} complexity")
            self.llm_inference_benchmark(prompt_complexity=complexity)
        
        # Workflow analysis benchmark
        for complexity in ["simple", "medium", "complex"]:
            logger.info(f"Testing workflow analysis with {complexity} workflow")
            self.workflow_analysis_benchmark(workflow_complexity=complexity)
        
        # UI responsiveness benchmark
        for component in ["recommendation_panel", "suggestion_preview", "workflow_comparison", "make_better_button"]:
            logger.info(f"Testing UI responsiveness for {component}")
            self.ui_responsiveness_benchmark(component_type=component)
        
        # Suggestion quality benchmark
        for suggestion_type in ["prompt", "workflow", "module"]:
            logger.info(f"Testing suggestion quality for {suggestion_type} suggestions")
            self.suggestion_quality_benchmark(suggestion_type=suggestion_type)
        
        # Multi-tenant benchmark
        logger.info("Testing multi-tenant performance")
        self.multi_tenant_benchmark(num_tenants=3, requests_per_tenant=5)  # Reduced for benchmarks
        
        logger.info("Recommendation Service benchmarks completed")
        return True


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks for AImPact Platform")
    parser.add_argument(
        "--component",
        choices=["memory", "optimizer", "recommendation", "all"],
        default="all",
        help="Component to benchmark",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Base URL for the API",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations for each benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-results",
        help="Directory to store benchmark results",
    )
    
    args = parser.parse_args()
    
    # Update global constants if specified
    global BENCHMARK_ITERATIONS
    global RESULT_DIR
    
    if args.iterations:
        BENCHMARK_ITERATIONS = args.iterations
    
    if args.output_dir:
        RESULT_DIR = args.output_dir
        os.makedirs(RESULT_DIR, exist_ok=True)
    
    logger.info(f"Starting benchmarks with {BENCHMARK_ITERATIONS} iterations")
    logger.info(f"Results will be stored in {RESULT_DIR}")
    
    # Run the selected benchmarks
    if args.component in ["memory", "all"]:
        memory_benchmarks = MemoryServiceBenchmarks(api_base_url=args.api_base_url)
        memory_benchmarks.run_all()
    
    if args.component in ["optimizer", "all"]:
        optimizer_benchmarks = OptimizerServiceBenchmarks(api_base_url=args.api_base_url)
        optimizer_benchmarks.run_all()
    
    if args.component in ["recommendation", "all"]:
        recommendation_benchmarks = RecommendationServiceBenchmarks(api_base_url=args.api_base_url)
        recommendation_benchmarks.run_all()
    
    logger.info("All benchmarks completed")
    
    # Generate summary report
    generate_summary_report()


def generate_summary_report():
    """Generate a summary report of all benchmark results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "memory_service": {},
        "optimizer_service": {},
        "recommendation_service": {},
    }
    
    # Collect all result files
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith("_results.json")]
    
    for file in result_files:
        with open(os.path.join(RESULT_DIR, file), "r") as f:
            result = json.load(f)
            
            # Categorize by service
            if "context" in file or "memory" in file:
                service = "memory_service"
            elif "optimizer" in file or "training" in file or "model" in file:
                service = "optimizer_service"
            else:
                service = "recommendation_service"
            
            # Add to summary
            summary[service][result["function"]] = {
                "average_duration_ms": result["average_duration_ms"],
                "p95_duration_ms": result["p95_duration_ms"],
                "average_cpu_percent": result["average_cpu_percent"],
                "average_memory_mb": result["average_memory_mb"],
            }
    
    # Save the summary report
    with open(os.path.join(RESULT_DIR, "summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report generated: {os.path.join(RESULT_DIR, 'summary_report.json')}")
    
    # Generate HTML report
    generate_html_report(summary)


def generate_html_report(summary):
    """Generate an HTML report from the summary data."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>AImPact Platform Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .service-section { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>AImPact Platform Benchmark Results</h1>
    <p>Generated: """ + summary["timestamp"] + """</p>
    """
    
    # Add each service section
    for service_name, benchmarks in summary.items():
        if service_name == "timestamp":
            continue
            
        html += f"""
    <div class="service-section">
