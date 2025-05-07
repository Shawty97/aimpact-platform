"""Authentication service for handling user authentication."""
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from backend.auth.models.token import Token, TokenData
from backend.auth.models.user import User, UserInDB
from backend.auth.services.user_service import UserService


class AuthService:
    """Service for handling authentication operations."""
    
    # To be loaded from environment variables in production
    SECRET_KEY = "temporary_secret_key_replace_in_production"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")
    
    def __init__(self, user_service: UserService):
        """Initialize with user service."""
        self.user_service = user_service
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password."""
        user = await self.user_service.get_user_by_email(email)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return User(**user.dict())
    
    def create_access_token(self, user: User) -> Token:
        """Create an access token for the user."""
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        token_data = TokenData(
            sub=str(user.id),
            email=user.email,
            roles=user.roles,
            exp=expires_at,
            jti=f"{user.id}:{now.timestamp()}",
            iat=now
        )
        
        encoded_jwt = jwt.encode(
            token_data.dict(),
            self.SECRET_KEY,
            algorithm=self.ALGORITHM
        )
        
        return Token(
            access_token=encoded_jwt,
            expires_at=expires_at
        )
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """Get the current user from the JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(
                token,
                self.SECRET_KEY,
                algorithms=[self.ALGORITHM]
            )
            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception
        except jwt.PyJWTError:
            raise credentials_exception
        
        user = await self.user_service.get_user_by_id(user_id)
        if user is None:
            raise credentials_exception
        
        return User(**user.dict())

