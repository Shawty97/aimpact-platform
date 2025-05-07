"""Token models for JWT authentication."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from backend.auth.models.user import UserRole


class Token(BaseModel):
    """Token model returned to client."""
    
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class TokenData(BaseModel):
    """Token data model for JWT claims."""
    
    sub: str  # User ID
    email: str
    roles: List[UserRole]
    exp: datetime
    jti: str  # JWT ID for token revocation
    iat: datetime  # Issued at


class TokenBlacklist(BaseModel):
    """Token blacklist model for revoked tokens."""
    
    jti: str
    exp: datetime
    revoked_at: datetime
    revoked_by: Optional[str] = None

