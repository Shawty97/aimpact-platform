"""
Multi-Tenant Auth Database Models

This module defines the SQLAlchemy models for the multi-tenant authentication system.
"""
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, Table, JSON, UniqueConstraint
from sqlalchemy.orm import relationship

from backend.auth.db.database import Base

# Roles
class UserRole(str):
    """User role constants."""
    ADMIN = "admin"
    USER = "user"


# Association table for user_roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String(36), ForeignKey('users.id')),
    Column('role', String(50))
)


# Tenant model
class Tenant(Base):
    """SQLAlchemy model for tenant."""
    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True, index=True, nullable=True)
    settings = Column(JSON, nullable=True)  # JSON object for tenant-specific settings
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary="tenant_users", back_populates="tenants")
    
    def __repr__(self):
        return f"<Tenant {self.name}>"


# User model
class User(Base):
    """SQLAlchemy model for user."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True)
    username = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenants = relationship("Tenant", secondary="tenant_users", back_populates="users")
    roles = relationship("Role", secondary=user_roles, backref="users")
    
    def __repr__(self):
        return f"<User {self.username}>"


# Tenant-User association model with additional attributes
class TenantUser(Base):
    """SQLAlchemy model for tenant-user association with additional attributes."""
    __tablename__ = "tenant_users"

    tenant_id = Column(String(36), ForeignKey("tenants.id"), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    is_default = Column(Boolean, default=False)  # Whether this is the user's default tenant
    role = Column(String(50), default=UserRole.USER)  # Role within this tenant
    settings = Column(JSON, nullable=True)  # User-specific settings for this tenant
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        # A user can only have one default tenant
        UniqueConstraint('user_id', 'is_default', name='uix_user_default_tenant', 
                        sqlite_on_conflict='REPLACE'),
    )
    
    tenant = relationship("Tenant")
    user = relationship("User")
    
    def __repr__(self):
        return f"<TenantUser {self.user_id} in {self.tenant_id}>"


# Role model
class Role(Base):
    """SQLAlchemy model for role."""
    __tablename__ = "roles"

    name = Column(String(50), primary_key=True)
    description = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<Role {self.name}>"


# Token blacklist
class TokenBlacklist(Base):
    """SQLAlchemy model for blacklisted tokens."""
    __tablename__ = "token_blacklist"

    jti = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), index=True)
    exp = Column(Integer)  # expiration timestamp
    revoked_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TokenBlacklist {self.jti}>"


# Subscription Plan model
class Plan(Base):
    """SQLAlchemy model for subscription plans."""
    __tablename__ = "plans"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)
    stripe_price_id = Column(String(100), unique=True, nullable=True)
    features = Column(JSON, nullable=True)  # JSON containing plan features
    api_quota = Column(Integer, default=1000)  # Number of API calls allowed per month
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
    
    def __repr__(self):
        return f"<Plan {self.name}>"


# Subscription model
class Subscription(Base):
    """SQLAlchemy model for subscriptions."""
    __tablename__ = "subscriptions"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=False)
    stripe_subscription_id = Column(String(100), unique=True, nullable=True)
    status = Column(String(50), default="active")  # active, canceled, past_due, etc.
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant")
    plan = relationship("Plan", back_populates="subscriptions")
    
    def __repr__(self):
        return f"<Subscription {self.id} for Tenant {self.tenant_id}>"


# API Usage model for tracking quota consumption
class ApiUsage(Base):
    """SQLAlchemy model for tracking API usage."""
    __tablename__ = "api_usage"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)  # GET, POST, etc.
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant")
    user = relationship("User")
    
    def __repr__(self):
        return f"<ApiUsage {self.endpoint} by Tenant {self.tenant_id}>"

