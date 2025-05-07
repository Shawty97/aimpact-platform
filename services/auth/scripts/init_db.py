#!/usr/bin/env python3
"""
Database initialization script for the AImpact Authentication Service.

This script:
1. Creates database tables if they don't exist
2. Sets up default roles (admin, manager, user, guest)
3. Creates an initial admin user

Usage:
    python init_db.py [--force] [--admin-email EMAIL] [--admin-password PASSWORD]

Options:
    --force             Drop existing tables before creating new ones
    --admin-email       Email for the admin user (default: admin@aimpact.ai)
    --admin-password    Password for the admin user (generated if not provided)
"""

import argparse
import asyncio
import logging
import os
import secrets
import string
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sqlalchemy as sa
from passlib.context import CryptContext
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.config import settings
from db.base import Base
from models.role import Role
from models.token import TokenBlacklist
from models.user import User


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("init_db")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_password(length=16):
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_-+=<>?"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


async def create_tables(engine, drop_existing=False):
    """Create database tables."""
    if drop_existing:
        logger.info("Dropping existing tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    logger.info("Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_roles(session: AsyncSession):
    """Create default roles if they don't exist."""
    logger.info("Creating default roles...")
    
    # Default roles with descriptions and permissions
    default_roles = [
        {
            "name": "admin",
            "description": "Full administrative access to all system functions",
            "permissions": ["admin:read", "admin:write", "user:read", "user:write", "agent:read", "agent:write"]
        },
        {
            "name": "manager",
            "description": "Manage users and view system data",
            "permissions": ["user:read", "user:write", "agent:read", "agent:write"]
        },
        {
            "name": "user",
            "description": "Regular user with standard permissions",
            "permissions": ["user:read", "agent:read", "agent:write"]
        },
        {
            "name": "guest",
            "description": "Limited read-only access",
            "permissions": ["user:read", "agent:read"]
        }
    ]

    created_roles = []
    
    for role_data in default_roles:
        # Check if role already exists
        stmt = sa.select(Role).where(Role.name == role_data["name"])
        result = await session.execute(stmt)
        existing_role = result.scalar_one_or_none()
        
        if existing_role:
            logger.info(f"Role '{role_data['name']}' already exists")
            created_roles.append(existing_role)
        else:
            # Create new role
            new_role = Role(
                name=role_data["name"], 
                description=role_data["description"],
                permissions=role_data["permissions"]
            )
            session.add(new_role)
            logger.info(f"Created role: {role_data['name']}")
            created_roles.append(new_role)
    
    await session.commit()
    
    # Refresh roles to get their IDs
    for role in created_roles:
        await session.refresh(role)
    
    return created_roles


async def create_admin_user(session: AsyncSession, roles, email: str, password: str = None):
    """Create admin user if it doesn't exist."""
    # Find admin role
    admin_role = next((r for r in roles if r.name == "admin"), None)
    if not admin_role:
        logger.error("Admin role not found!")
        return None
    
    # Check if admin already exists
    stmt = sa.select(User).where(User.email == email)
    result = await session.execute(stmt)
    existing_admin = result.scalar_one_or_none()
    
    if existing_admin:
        logger.info(f"Admin user with email '{email}' already exists")
        return existing_admin
    
    # Generate password if not provided
    if not password:
        password = generate_password()
        logger.info(f"Generated admin password: {password}")
    
    # Hash the password
    hashed_password = pwd_context.hash(password)
    
    # Create admin user
    admin_user = User(
        email=email,
        hashed_password=hashed_password,
        first_name="Admin",
        last_name="User",
        is_active=True,
        role_id=admin_role.id
    )
    
    session.add(admin_user)
    await session.commit()
    await session.refresh(admin_user)
    
    logger.info(f"Created admin user: {email}")
    return admin_user


async def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description="Initialize the authentication database")
    parser.add_argument("--force", action="store_true", help="Force recreate all tables")
    parser.add_argument("--admin-email", default="admin@aimpact.ai", help="Admin email address")
    parser.add_argument("--admin-password", default=None, help="Admin password (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Check for valid email format
    admin_email = args.admin_email
    
    # Create engine and session
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        # Create tables
        await create_tables(engine, drop_existing=args.force)
        
        # Create session
        async with async_session() as session:
            # Create roles
            roles = await create_roles(session)
            
            # Create admin user
            admin = await create_admin_user(session, roles, admin_email, args.admin_password)
            
            if admin and not args.admin_password:
                logger.info("======== IMPORTANT ========")
                logger.info("Database initialized successfully!")
                logger.info(f"Admin user: {admin_email}")
                logger.info("Please save the generated admin password shown above.")
                logger.info("===========================")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

