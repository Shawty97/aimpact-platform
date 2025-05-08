"""
Tenant Context Middleware

This middleware extracts tenant information from JWT tokens and adds
tenant context to the request.
"""
from typing import Optional

from fastapi import Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt

from backend.app.core.config import settings
from backend.auth.db.database import get_db
from backend.auth.db.models import Tenant, User, TenantUser

# OAuth2 scheme for JWT
oauth2_scheme = HTTPBearer()


class TenantMiddleware:
    """
    Middleware to extract tenant context from JWT tokens and add it to the request.
    """
    
    async def __call__(self, request: Request, call_next):
        """
        Extract tenant information from authorization header and add to request state.
        """
        # Set default tenant to None
        request.state.tenant_id = None
        request.state.tenant = None
        request.state.user_id = None
        
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            
            try:
                # Decode JWT without verification (we'll verify in auth dependencies)
                # We just need the tenant information here
                payload = jwt.decode(
                    token, 
                    settings.JWT_SECRET_KEY, 
                    algorithms=[settings.JWT_ALGORITHM],
                    options={"verify_signature": True}
                )
                
                # Extract tenant_id from token if available
                tenant_id = payload.get("tenant_id")
                user_id = payload.get("sub")
                
                if tenant_id:
                    request.state.tenant_id = tenant_id
                
                if user_id:
                    request.state.user_id = user_id
                    
                # Database access is done in dependencies to avoid performance issues
                # in middleware
            
            except jwt.PyJWTError:
                # Invalid token, continue without tenant context
                pass
        
        # Continue processing the request
        response = await call_next(request)
        return response


async def get_current_tenant(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[Tenant]:
    """
    Get the current tenant from the request state.
    This is a FastAPI dependency that can be used in route handlers.
    """
    tenant_id = request.state.tenant_id
    
    if not tenant_id:
        return None
    
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    return tenant


async def get_tenant_for_user(
    user_id: str,
    tenant_id: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Optional[Tenant]:
    """
    Get a tenant for a specific user.
    If tenant_id is provided, verify the user has access to that tenant.
    Otherwise, return the user's default tenant.
    """
    if tenant_id:
        # Verify the user has access to the specified tenant
        tenant_user = db.query(TenantUser).filter(
            TenantUser.user_id == user_id,
            TenantUser.tenant_id == tenant_id
        ).first()
        
        if tenant_user:
            return db.query(Tenant).filter(Tenant.id == tenant_id).first()
        return None
    else:
        # Get the user's default tenant
        tenant_user = db.query(TenantUser).filter(
            TenantUser.user_id == user_id,
            TenantUser.is_default == True
        ).first()
        
        if tenant_user:
            return db.query(Tenant).filter(Tenant.id == tenant_user.tenant_id).first()
        
        # If no default tenant, try to get any tenant
        tenant_user = db.query(TenantUser).filter(
            TenantUser.user_id == user_id
        ).first()
        
        if tenant_user:
            return db.query(Tenant).filter(Tenant.id == tenant_user.tenant_id).first()
            
        return None

