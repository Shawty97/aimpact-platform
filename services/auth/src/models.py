"""
Authentication Service Data Models

This module defines the data models used by the authentication service,
including user models and token models.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# SQLAlchemy Models

class UserRole(str, Enum):
    """User role enum for role-based access control"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"

# Association table for user_roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String(36), ForeignKey('users.id')),
    Column('role', String(50))
)

class UserDB(Base):
    """SQLAlchemy model for user in database"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship for roles
    roles = relationship("Role", secondary=user_roles, backref="users")


# Pydantic Models

class UserBase(BaseModel):
    """Base model for user data"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    roles: List[UserRole] = [UserRole.USER]


class UserCreate(UserBase):
    """Model for user creation"""
    password: str

    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v


class UserUpdate(BaseModel):
    """Model for user updates"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    roles: Optional[List[UserRole]] = None
    password: Optional[str] = None

    @validator('password')
    def password_strength(cls, v):
        """Validate password strength if provided"""
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v


class User(UserBase):
    """Response model for user data"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    """Model for authentication tokens"""
    access_token: str
    token_type: str = "bearer"
    expires_at: int  # Unix timestamp


class TokenData(BaseModel):
    """Model for token payload data"""
    sub: str  # user id
    email: EmailStr
    roles: List[UserRole]
    exp: int  # expiration time
    iat: int  # issued at time
    jti: str  # JWT ID


class TokenBlacklist(Base):
    """SQLAlchemy model for blacklisted tokens"""
    __tablename__ = "token_blacklist"

    jti = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), index=True)
    exp = Column(Integer)  # expiration timestamp
    revoked_at = Column(DateTime, default=datetime.utcnow)

