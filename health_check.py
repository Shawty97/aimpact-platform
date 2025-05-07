#!/usr/bin/env python3
"""
Health Check Script for AIMpact Platform

This script performs various health checks on the AIMpact system components:
- Database connection
- Redis connection
- API endpoints
- System resources
- Network connectivity

Usage:
  python health_check.py [options]

Options:
  --json              Output results in JSON format
  --verbose, -v       Show detailed output
  --timeout SECONDS   Set timeout for checks (default: 5)
  --endpoint URL      API endpoint to check (default: http://localhost:8000/health)
  --db-url URL        Database connection URL
  --redis-host HOST   Redis host (default: localhost)
  --redis-port PORT   Redis port (default: 6379)
  --check TYPE        Run only specific check (db, redis, api, system, network)
  --all               Run all checks (default)
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class HealthCheckTimeout(Exception):
    """Exception raised when a health check times out."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager to timeout a block of code."""
    def timeout_handler(signum, frame):
        raise HealthCheckTimeout(f"Operation timed out after {seconds} seconds")

    import signal
    # Set the timeout handler
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class HealthCheck:
    """Class to perform health checks on system components."""
    
    def __init__(self, options: Dict[str, Any]):
        """Initialize with options."""
        self.options = options
        self.results = {}
        self.timeout_seconds = options.get('timeout', 5)
        self.verbose = options.get('verbose', False)
        
    def _log(self, message: str):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            
    def check_database_connection(self) -> Dict[str, Any]:
        """Check database connection."""
        result = {
            "name": "database_connection",
            "status": "skipped",
            "message": "SQLAlchemy not installed",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }
        
        if not SQLALCHEMY_AVAILABLE:
            return result
        
        db_url = self.options.get('db_url')
        if not db_url:
            result["message"] = "No database URL provided"
            return result
        
        self._log(f"Checking database connection to {db_url.split('@')[-1]}")
        
        try:
            with timeout(self.timeout_seconds):
                start_time = time.time()
                engine = create_engine(db_url)
                connection = engine.connect()
                connection.execute(sqlalchemy.text("SELECT 1"))
                connection.close()
                end_time = time.time()
                
                result.update({
                    "status": "success",
                    "message": "Database connection successful",
                    "details": {
                        "response_time_ms": round((end_time - start_time) * 1000, 2)
                    }
                })
        except HealthCheckTimeout:
            result.update({
                "status": "error",
                "message": f"Database connection timed out after {self.timeout_seconds} seconds"
            })
        except SQLAlchemyError as e:
            result.update({
                "status": "error",
                "message": f"Database connection failed: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
            
        return result
    
    def check_redis_connection(self) -> Dict[str, Any]:
        """Check Redis connection."""
        result = {
            "name": "redis_connection",
            "status": "skipped",
            "message": "Redis package not installed",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }
        
        if not REDIS_AVAILABLE:
            return result
        
        redis_host = self.options.get('redis_host', 'localhost')
        redis_port = self.options.get('redis_port', 6379)
        
        self._log(f"Checking Redis connection to {redis_host}:{redis_port}")
        
        try:
            with timeout(self.timeout_seconds):
                start_time = time.time()
                r = redis.Redis(host=redis_host, port=redis_port)
                r.ping()
                end_time = time.time()
                
                # Get some info about Redis
                info = r.info(section='memory')
                
                result.update({
                    "status": "success",
                    "message": "Redis connection successful",
                    "details": {
                        "response_time_ms": round((end_time - start_time) * 1000, 2),
                        "redis_version": r.info().get('redis_version', 'unknown'),
                        "used_memory_human": info.get('used_memory_human', 'unknown')
                    }
                })
        except HealthCheckTimeout:
            result.update({
                "status": "error",
                "message": f"Redis connection timed out after {self.timeout_seconds} seconds"
            })
        except redis.RedisError as e:
            result.update({
                "status": "error",
                "message": f"Redis connection failed: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
            
        return result
    
    def check_api_endpoint(self) -> Dict[str, Any]:
        """Check API endpoint."""
        result = {
            "name": "api_endpoint",
            "status": "skipped",
            "message": "Requests package not installed",
            "timestamp": datetime.now().isoformat(),
            "details": {}
        }
        
        if not REQUESTS_AVAILABLE:
            return result
        
        endpoint = self.options.get('endpoint', 'http://localhost:8000/health')
        
        self._log(f"Checking API endpoint: {endpoint}")
        
        try:
            with timeout(self.timeout_seconds):
                start_time = time.time()
                response = requests.get(endpoint, timeout=self.timeout_seconds)
                end_time = time.time()
                
                result.update({
                    "status": "success" if response.status_code == 200 else "warning",
                    "message": f"API responded with status code {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response_time_ms": round((end_time - start_time) * 1000, 2),
                        "content_type": response.headers.get('Content-Type', 'unknown')
                    }
                })
                
                # Try to parse JSON response
                try:
                    json_response = response.json()
                    result["details"]["response"] = json_response
                except ValueError:
                    result["details"]["response_text"] = (
                        response.text[:100] + "..." if len(response.text) > 100 else response.text
                    )
                    
        except HealthCheckTimeout:
            result.update({
                "status": "error",
                "message": f"API request timed out after {self.timeout_seconds} seconds"
            })
        except requests.RequestException as e:
            result.update({
                "status": "error",
                "message": f"API request failed: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "details": {"error_type": e.__class__.__name__}
            })
            
        return result
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources."""
        result = {
            "name": "system_resources",
            "status": "skipped",
            "message": "psutil package not installed",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "system": platform.system(),
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }
        }
        
        if not PSUTIL_AVAILABLE:
            return result
        
        self._log("Checking system resources")
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            
            # Memory
            memory = psutil.virtual_memory()
            memory_total_gb = round(memory.total / (1024**3), 2)
            memory_used_gb = round(memory.used / (1024**3), 2)
            memory_percent = memory.percent
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_total_gb = round(disk.total / (1024**3), 2)
            disk_used_gb = round(disk.used / (1024**3), 2)
            disk_percent = disk.percent
            
            # Process information for current Python process
            process = psutil.Process()
            process_cpu_percent = process.cpu_percent(interval=0.5)
            process_memory_mb = round(process.memory_info().rss / (1024**2), 2)
            
            # Determine status based on thresholds
            status = "success"
            message = "System resources are healthy"
            
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = "error"
                message = "System resources critically low"
            elif cpu_percent > 75 or memory_percent > 75 or disk_percent > 75:
                status = "warning"
                message = "System resources running low"
                
            result.update({
                "status": status,
                "message": message,
                "details": {
                    **result["details"],
                    "cpu": {
                        "percent": cpu_percent,
                        "count": cpu_count
                    },
                    "memory": {
                        "total_gb": memory_total_gb,
                        "used_gb": memory_used_gb,
                        "percent": memory_percent
                    },
                    "disk": {
                        "total_gb": disk_total_gb,
                        "used_gb": disk_used_gb,
                        "percent": disk_percent
                    },
                    "process": {
                        "pid": process.pid,
                        "cpu_percent": process_cpu_percent,
                        "memory_mb": process_memory_mb
                    }
                }
            })
            
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"Error checking system resources: {str(e)}",
                "details": {**result["details"], "error_type": e.__class__.__name__}
            })
            
        return result
    
    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to important services."""
        result = {
            "name": "network_connectivity",
            "status": "running",
            "message": "Checking network connectivity",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "checks": []
            }
        }
        
        # Define important endpoints to check
        endpoints = [
            ("google.com", 443, "Internet connectivity"),
            ("api.openai.com", 443, "OpenAI API"),
            # Add other important services here
        ]
        
        if self.options.get('redis_host'):
            redis_host = self.options.get('redis_host')
            redis_port = self.options.get('redis_port', 6379)
            endpoints.append((redis_host, redis_port, "Redis server"))
        
        all_successful = True
        
        for host, port, description in endpoints:
            check_result = {
                "host": host,
                "port": port,
                "description": description
            }
            
            try:
                self._log(f"Checking connectivity to {host}:{port} ({description})")
                
                with timeout(self.timeout_seconds):
                    start_time = time.time()
                    # Create socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.timeout_seconds)
                    
                    # Connect to the host and port
                    sock.connect((host, port))
                    sock.close()
                    
                    end_time = time.time()
                    
                    check_result.update({
                        "status": "success",
                        "message": f"Connected successfully",
                        "response_time_ms": round((end_time - start_time) * 1000, 2)
                    })
            except HealthCheckTimeout:
                all_successful = False
                check_result.update({
                    "status": "error",
                    "message": f"Connection timed out after {self.timeout_seconds} seconds"
                })
            except socket.error as e:
                all_successful = False
                check_result.update({
                    "status": "error",
                    "message": f"Connection failed: {str(e)}",
                    "error_type": e.__class__.__name__
                })
            except Exception as e:
                all_successful = False
                check_result.update({
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}",
                    "error_type": e.__class__.__name__
                })
                
            result["details"]["checks"].append(check_result)
                
        # Update overall status based on individual checks
        if all_successful:
            result.update({
                "status": "success",
                "message": "All network connectivity checks passed"
            })
        else:
            result.update({
                "status": "warning",
                "message": "Some network connectivity checks failed"
            })
            
        return result
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all requested health checks and return results."""
        self._log("Starting health checks")
        start_time = time.time()
        
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "summary": {
                "total": 0,
                "success": 0,
                "warning": 0,
                "error": 0,
                "skipped": 0
            }
        }
        
        # Determine which checks to run
        checks_to_run = []
        specific_check = self.options.get('check')
        
        if specific_check:
            if specific_check == 'db':
                checks_to_run.append(('database', self.check_database_connection))
            elif specific_check == 'redis':
                checks_to_run.append(('redis', self.check_redis_connection))
            elif specific_check == 'api':
                checks_to_run.append(('api', self.check_api_endpoint))
            elif specific_check == 'system':
                checks_to_run.append(('system', self.check_system_resources))
            elif specific_check == 'network':
                checks_to_run.append(('network', self.check_network_connectivity))
        else:
            # Run all checks by default
            checks_to_run = [
                ('database', self.check_database_connection),
                ('redis', self.check_redis_connection),
                ('api', self.check_api_endpoint),
                ('system', self.check_system_resources),
                ('network', self.check_network_connectivity)
            ]
        
        # Run each check
        for check_name, check_func in checks_to_run:
            self._log(f"Running {check_name} check")
            try:
                check_result = check_func()
                results["checks"].append(check_result)
                
                # Update summary counters
                results["summary"]["total"] += 1
                status = check_result.get("status", "unknown")
                if status in ["success", "warning", "error", "skipped"]:
                    results["summary"][status] += 1
                    
            except Exception as e:
                error_result = {
                    "name": check_name,
                    "status": "error",
                    "message": f"Check failed with exception: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "details": {"error_type": e.__class__.__name__}
                }
                results["checks"].append(error_result)
                results["summary"]["total"] += 1
                results["summary"]["error"] += 1
                
        # Determine overall status
        if results["summary"]["error"] > 0:
            results["status"] = "error"
            results["message"] = "One or more health checks failed"
        elif results["summary"]["warning"] > 0:
            results["status"] = "warning"
            results["message"] = "One or more health checks reported warnings"
        else:
            results["status"] = "success"
            results["message"] = "All health checks passed successfully"
            
        end_time = time.time()
        results["duration_ms"] = round((end_time - start_time) * 1000, 2)
        
        self._log(f"Health checks completed in {results['duration_ms']}ms with status: {results['status']}")
        
        return results


def format_output(results: Dict[str, Any], json_output: bool = False) -> str:
    """Format the health check results for output."""
    if json_output:
        return json.dumps(results, indent=2)
    
    # Console output formatting
    output = []
    
    # Header
    output.append("=" * 80)
    output.append(f"AImpact Platform Health Check Results")
    output.append(f"Timestamp: {results['timestamp']}")
    output.append(f"Duration: {results.get('duration_ms', 0)}ms")
    output.append(f"Overall Status: {results['status'].upper()}")
    output.append("=" * 80)
    
    # Summary
    summary = results.get("summary", {})
    output.append("\nSummary:")
    output.append(f"  Total Checks: {summary.get('total', 0)}")
    output.append(f"  Success: {summary.get('success', 0)}")
    output.append(f"  Warning: {summary.get('warning', 0)}")
    output.append(f"  Error: {summary.get('error', 0)}")
    output.append(f"  Skipped: {summary.get('skipped', 0)}")
    
    # Individual check results
    output.append("\nCheck Results:")
    for check in results.get("checks", []):
        status = check.get("status", "unknown").upper()
        name = check.get("name", "unknown")
        message = check.get("message", "No message")
        
        status_color = ""
        reset_color = ""
        
        # Add colors if terminal supports it
        if sys.stdout.isatty():
            if status == "SUCCESS":
                status_color = "\033[92m"  # Green
            elif status == "WARNING":
                status_color = "\033[93m"  # Yellow
            elif status == "ERROR":
                status_color = "\033[91m"  # Red
            elif status == "SKIPPED":
                status_color = "\033[94m"  # Blue
            
            reset_color = "\033[0m"
        
        output.append(f"\n  {status_color}[{status}]{reset_color} {name}")
        output.append(f"    {message}")
        
        # Add details
        details = check.get("details", {})
        if "checks" in details and isinstance(details["checks"], list):
            # Handle network connectivity details
            for subcheck in details["checks"]:
                sub_status = subcheck.get("status", "unknown").upper()
                host = subcheck.get("host", "unknown")
                port = subcheck.get("port", "?")
                description = subcheck.get("description", "")
                sub_message = subcheck.get("message", "")
                
                sub_status_color = ""
                if sys.stdout.isatty():
                    if sub_status == "SUCCESS":
                        sub_status_color = "\033[92m"  # Green
                    elif sub_status == "WARNING":
                        sub_status_color = "\033[93m"  # Yellow
                    elif sub_status == "ERROR":
                        sub_status_color = "\033[91m"  # Red
                
                output.append(f"      {sub_status_color}[{sub_status}]{reset_color} {host}:{port} ({description})")
                output.append(f"        {sub_message}")
                if "response_time_ms" in subcheck:
                    output.append(f"        Response time: {subcheck['response_time_ms']}ms")
        else:
            # Handle other types of details
            for key, value in details.items():
                if isinstance(value, dict):
                    output.append(f"    {key}:")
                    for k, v in value.items():
                        output.append(f"      {k}: {v}")
                else:
                    output.append(f"    {key}: {value}")
    
    return "\n".join(output)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Health Check Script for AImpact Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--timeout", type=int, default=5, help="Set timeout for checks in seconds")
    parser.add_argument("--endpoint", default="http://localhost:8000/health", help="API endpoint to check")
    parser.add_argument("--db-url", help="Database connection URL")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--check", 
        choices=["db", "redis", "api", "system", "network"],
        help="Run only specific check"
    )
    parser.add_argument("--all", action="store_true", help="Run all checks (default)")
    
    return parser.parse_args()


def run():
    """Main entry point for the health check script."""
    args = parse_arguments()
    
    # Configure options
    options = {
        'verbose': args.verbose,
        'timeout': args.timeout,
        'endpoint': args.endpoint,
        'db_url': args.db_url,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'check': args.check
    }
    
    # Check for missing dependencies
    missing_deps = []
    if not REQUESTS_AVAILABLE:
        missing_deps.append("requests")
    if not SQLALCHEMY_AVAILABLE and args.db_url:
        missing_deps.append("sqlalchemy")
    if not REDIS_AVAILABLE and (args.redis_host != "localhost" or args.redis_port != 6379):
        missing_deps.append("redis")
    if not PSUTIL_AVAILABLE:
        missing_deps.append("psutil")
        
    if missing_deps and not args.json:
        print("Warning: The following dependencies are missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("Some health checks may be skipped.")
        print("Install dependencies with: pip install " + " ".join(missing_deps))
        print()
    
    # Run health checks
    health_checker = HealthCheck(options)
    results = health_checker.run_checks()
    
    # Format and output results
    output = format_output(results, args.json)
    print(output)
    
    # Return non-zero exit code if there are errors
    if results["status"] == "error":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
