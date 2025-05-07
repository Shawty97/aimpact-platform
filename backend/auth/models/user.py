"""User model for authentication system."""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User role enum for role-based access control."""
    
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"


class UserBase(BaseModel):
    """Base user model."""
    
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    roles: List[UserRole] = [UserRole.USER]


class UserCreate(UserBase):
    """User creation model."""
    
    password: str


class UserInDB(UserBase):
    """User model as stored in the database."""
    
    id: str = Field(..., alias="_id")
    hashed_password: str
    created_at: datetime
    updated_at: datetime


class User(UserBase):
    """User model returned to client."""
    
    id: str
    created_at: datetime

