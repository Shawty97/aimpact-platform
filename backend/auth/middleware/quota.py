"""
API Quota Middleware

This middleware tracks API usage and enforces quota limits based on
the tenant's subscription plan.
"""
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Tuple, List
import asyncio
import logging
from collections import defaultdict

from fastapi import Request, HTTPException, status, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from starlette.responses import JSONResponse

from backend.app.core.config import settings
from backend.auth.db.database import get_db
from backend.auth.db.models import ApiUsage, Tenant, Subscription, Plan


# Configure logging
logger = logging.getLogger("backend.auth.middleware.quota")

# Cache for rate limiting and quota checking to reduce database hits
# Structure: {tenant_id: {'count': int, 'last_reset': datetime, 'rate_limit': {'endpoint': {'count': int, 'last_check': timestamp}}}}
usage_cache = defaultdict(lambda: {'count': 0, 'last_reset': datetime.now(), 'rate_limit': defaultdict(dict)})

# Rate limit configurations (requests per minute by default)
RATE_LIMIT_DEFAULT = 60  # Default requests per minute
RATE_LIMIT_WINDOW = 60  # Window in seconds for rate limiting (1 minute)


class QuotaMiddleware:
    """
    Middleware to track API usage and enforce quota limits.
    """
    
    def __init__(self, exempt_paths: List[str] = None, db_session_factory=None):
        """
        Initialize quota middleware.
        
        Args:
            exempt_paths: List of path prefixes that are exempt from quota checks
            db_session_factory: Function to get database session
        """
        self.exempt_paths = exempt_paths or ['/docs', '/openapi.json', '/redoc', '/health']
        self.db_session_factory = db_session_factory or get_db
        
    async def check_quota(self, tenant_id: str, db: Session) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Check if the tenant has exceeded their quota.
        
        Returns:
            Tuple of (quota_exceeded, current_usage, max_quota)
        """
        if not tenant_id:
            # No tenant, so can't check quota
            return False, None, None
            
        # Check cache first to avoid database hits
        tenant_cache = usage_cache[tenant_id]
        cache_age = datetime.now() - tenant_cache['last_reset']
        
        # If we have recent cache data, use it
        if cache_age < timedelta(minutes=5) and tenant_cache['count'] > 0:
            # Check subscription from db
            subscription = db.query(Subscription).filter(
                Subscription.tenant_id == tenant_id,
                Subscription.status == 'active'
            ).order_by(desc(Subscription.current_period_end)).first()
            
            if subscription:
                plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
                if plan:
                    return tenant_cache['count'] >= plan.api_quota, tenant_cache['count'], plan.api_quota
        
        # Get the tenant's active subscription
        subscription = db.query(Subscription).filter(
            Subscription.tenant_id == tenant_id,
            Subscription.status == 'active'
        ).order_by(desc(Subscription.current_period_end)).first()
        
        if not subscription:
            # No active subscription, use default free quota
            quota_limit = settings.DEFAULT_API_QUOTA
        else:
            # Get the plan's quota
            plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
            quota_limit = plan.api_quota if plan else settings.DEFAULT_API_QUOTA
        
        # Calculate current month's usage
        first_day_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usage_count = db.query(func.count(ApiUsage.id)).filter(
            ApiUsage.tenant_id == tenant_id,
            ApiUsage.timestamp >= first_day_of_month
        ).scalar()
        
        # Update cache
        tenant_cache['count'] = usage_count
        tenant_cache['last_reset'] = datetime.now()
        
        return usage_count >= quota_limit, usage_count, quota_limit
    
    async def check_rate_limit(self, tenant_id: str, endpoint: str) -> bool:
        """
        Check if the request exceeds rate limits.
        
        Returns:
            True if rate limit exceeded, False otherwise
        """
        if not tenant_id:
            return False
            
        # Get tenant's rate limit cache
        tenant_cache = usage_cache[tenant_id]
        rate_cache = tenant_cache['rate_limit'][endpoint]
        
        # Get current timestamp
        now = time.time()
        
        # Initialize if this is the first request for this endpoint
        if 'count' not in rate_cache:
            rate_cache['count'] = 0
            rate_cache['last_check'] = now
            return False
            
        # Check if we need to reset the counter (new window)
        if now - rate_cache['last_check'] > RATE_LIMIT_WINDOW:
            rate_cache['count'] = 0
            rate_cache['last_check'] = now
            return False
            
        # Check if rate limit exceeded
        if rate_cache['count'] >= RATE_LIMIT_DEFAULT:
            return True
            
        # Increment counter
        rate_cache['count'] += 1
        return False
    
    async def record_usage(self, request: Request, tenant_id: str, user_id: Optional[str], db: Session):
        """
        Record API usage in the database.
        """
        if not tenant_id:
            return
            
        # Extract endpoint information
        endpoint = request.url.path
        method = request.method
        
        # Create usage record
        usage = ApiUsage(
            tenant_id=tenant_id,
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            timestamp=datetime.now()
        )
        
        try:
            # Add to database
            db.add(usage)
            db.commit()
            
            # Update cache
            tenant_cache = usage_cache[tenant_id]
            tenant_cache['count'] += 1
            
        except Exception as e:
            # Don't let usage recording failures affect the request
            logger.error(f"Failed to record API usage: {str(e)}")
            db.rollback()
    
    async def __call__(self, request: Request, call_next):
        """
        Check if the tenant has exceeded their API quota and track usage.
        """
        # Skip quota checks for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get tenant and user IDs from request state (set by TenantMiddleware)
        tenant_id = getattr(request.state, 'tenant_id', None)
        user_id = getattr(request.state, 'user_id', None)
        
        # Skip quota checks if no tenant ID
        if not tenant_id:
            return await call_next(request)
            
        # Get database session
        db = next(self.db_session_factory())
        
        try:
            # Check rate limit
            endpoint = request.url.path
            rate_limited = await self.check_rate_limit(tenant_id, endpoint)
            
            if rate_limited:
                # Return 429 Too Many Requests
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded. Please try again later."
                    },
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
                )
            
            # Check quota
            quota_exceeded, current_usage, max_quota = await self.check_quota(tenant_id, db)
            
            if quota_exceeded:
                # Return 402 Payment Required
                return JSONResponse(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    content={
                        "detail": "API quota exceeded. Please upgrade your subscription.",
                        "current_usage": current_usage,
                        "quota_limit": max_quota
                    }
                )
            
            # Process the request
            response = await call_next(request)
            
            # Only record successful requests
            if 200 <= response.status_code < 300:
                # Record usage in the background to not block the response
                asyncio.create_task(self.record_usage(request, tenant_id, user_id, db))
            
            # Add usage headers to response
            if current_usage is not None and max_quota is not None:
                response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_DEFAULT)
                response.headers["X-RateLimit-Remaining"] = str(max(0, RATE_LIMIT_DEFAULT - usage_cache[tenant_id]['rate_limit'].get(endpoint, {}).get('count', 0)))
                response.headers["X-RateLimit-Reset"] = str(int(usage_cache[tenant_id]['rate_limit'].get(endpoint, {}).get('last_check', time.time()) + RATE_LIMIT_WINDOW))
                response.headers["X-Quota-Limit"] = str(max_quota)
                response.headers["X-Quota-Remaining"] = str(max(0, max_quota - current_usage))
            
            return response
            
        except Exception as e:
            # Log error but don't block the request
            logger.error(f"Error in quota middleware: {str(e)}")
            return await call_next(request)
        finally:
            # Close DB session
            db.close()


# FastAPI dependency for checking quota in individual routes
async def check_quota_limit(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    FastAPI dependency for checking quota limits.
    Can be used in individual routes for more fine-grained control.
    """
    tenant_id = getattr(request.state, 'tenant_id', None)
    
    if not tenant_id:
        return
        
    # Create middleware instance and check quota
    middleware = QuotaMiddleware()
    quota_exceeded, current_usage, max_quota = await middleware.check_quota(tenant_id, db)
    
    if quota_exceeded:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "message": "API quota exceeded. Please upgrade your subscription.",
                "current_usage": current_usage,
                "quota_limit": max_quota
            }
        )

