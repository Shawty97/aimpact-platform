#!/usr/bin/env python3
"""
Management utility for the AImpact Authentication Service.

This script provides administrative commands for:
- User management (create, update, delete, list)
- Role management (create, update, delete, list)
- Token management (list, revoke)
- Database maintenance (purge expired tokens, optimize)

Usage:
    python manage.py [command] [options]

Commands:
    user                User management commands
    role                Role management commands
    token               Token management commands
    db                  Database maintenance commands
    serve               Run the authentication service
"""

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sqlalchemy as sa
import uvicorn
from passlib.context import CryptContext
from pydantic import EmailStr, ValidationError
from rich.console import Console
from rich.table import Table
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
logger = logging.getLogger("manage")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rich console for pretty output
console = Console()


async def setup_db_session():
    """Set up and return a database session."""
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, async_session()


async def close_db_session(engine, session):
    """Close the database session and engine."""
    await session.close()
    await engine.dispose()


# User Management Commands
async def create_user(args):
    """Create a new user."""
    engine, session = await setup_db_session()
    
    try:
        # Check if role exists
        stmt = sa.select(Role).where(Role.id == args.role_id)
        result = await session.execute(stmt)
        role = result.scalar_one_or_none()
        
        if not role:
            console.print(f"[bold red]Error:[/bold red] Role with ID {args.role_id} not found")
            return
        
        # Check if user with email already exists
        stmt = sa.select(User).where(User.email == args.email)
        result = await session.execute(stmt)
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            console.print(f"[bold red]Error:[/bold red] User with email {args.email} already exists")
            return
        
        # Hash the password
        hashed_password = pwd_context.hash(args.password)
        
        # Create the user
        new_user = User(
            email=args.email,
            hashed_password=hashed_password,
            first_name=args.first_name,
            last_name=args.last_name,
            is_active=not args.inactive,
            role_id=args.role_id
        )
        
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        
        console.print(f"[bold green]Success:[/bold green] Created user {new_user.email} with ID {new_user.id}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def update_user(args):
    """Update an existing user."""
    engine, session = await setup_db_session()
    
    try:
        # Get user by ID or email
        if args.id:
            stmt = sa.select(User).where(User.id == args.id)
        else:
            stmt = sa.select(User).where(User.email == args.email)
        
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            console.print(f"[bold red]Error:[/bold red] User not found")
            return
        
        # Update user fields
        if args.new_email:
            user.email = args.new_email
        
        if args.first_name:
            user.first_name = args.first_name
        
        if args.last_name:
            user.last_name = args.last_name
        
        if args.password:
            user.hashed_password = pwd_context.hash(args.password)
        
        if args.role_id:
            # Check if role exists
            stmt = sa.select(Role).where(Role.id == args.role_id)
            result = await session.execute(stmt)
            role = result.scalar_one_or_none()
            
            if not role:
                console.print(f"[bold red]Error:[/bold red] Role with ID {args.role_id} not found")
                return
            
            user.role_id = args.role_id
        
        if args.activate:
            user.is_active = True
        
        if args.deactivate:
            user.is_active = False
        
        await session.commit()
        console.print(f"[bold green]Success:[/bold green] Updated user {user.email}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def list_users(args):
    """List users with optional filtering."""
    engine, session = await setup_db_session()
    
    try:
        # Base query
        stmt = sa.select(User)
        
        # Apply filters
        if args.role_id:
            stmt = stmt.where(User.role_id == args.role_id)
        
        if args.active_only:
            stmt = stmt.where(User.is_active == True)
        
        if args.inactive_only:
            stmt = stmt.where(User.is_active == False)
        
        # Apply sorting
        if args.sort_by == "email":
            stmt = stmt.order_by(User.email)
        elif args.sort_by == "created":
            stmt = stmt.order_by(User.created_at)
        elif args.sort_by == "id":
            stmt = stmt.order_by(User.id)
        
        # Get users
        result = await session.execute(stmt)
        users = result.scalars().all()
        
        # Get roles for users
        role_ids = set(user.role_id for user in users)
        stmt = sa.select(Role).where(Role.id.in_(role_ids))
        result = await session.execute(stmt)
        roles = {role.id: role for role in result.scalars().all()}
        
        # Create table
        table = Table(title="Users")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Email", style="green")
        table.add_column("Name", style="green")
        table.add_column("Role", style="yellow")
        table.add_column("Active", justify="center")
        table.add_column("Created At", style="blue")
        
        # Add rows
        for user in users:
            role_name = roles.get(user.role_id, Role(name="Unknown")).name
            active_status = "[green]Yes[/green]" if user.is_active else "[red]No[/red]"
            created_at = user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else "N/A"
            
            table.add_row(
                str(user.id),
                user.email,
                f"{user.first_name} {user.last_name}",
                role_name,
                active_status,
                created_at
            )
        
        # Print table
        console.print(table)
        console.print(f"Total users: {len(users)}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def delete_user(args):
    """Delete a user."""
    engine, session = await setup_db_session()
    
    try:
        # Get user by ID or email
        if args.id:
            stmt = sa.select(User).where(User.id == args.id)
        else:
            stmt = sa.select(User).where(User.email == args.email)
        
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            console.print(f"[bold red]Error:[/bold red] User not found")
            return
        
        # Delete user
        await session.delete(user)
        await session.commit()
        
        console.print(f"[bold green]Success:[/bold green] Deleted user {user.email} (ID: {user.id})")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


# Role Management Commands
async def create_role(args):
    """Create a new role."""
    engine, session = await setup_db_session()
    
    try:
        # Check if role with name already exists
        stmt = sa.select(Role).where(Role.name == args.name)
        result = await session.execute(stmt)
        existing_role = result.scalar_one_or_none()
        
        if existing_role:
            console.print(f"[bold red]Error:[/bold red] Role with name '{args.name}' already exists")
            return
        
        # Parse permissions
        permissions = []
        if args.permissions:
            permissions = [p.strip() for p in args.permissions.split(',')]
        
        # Create the role
        new_role = Role(
            name=args.name,
            description=args.description,
            permissions=permissions
        )
        
        session.add(new_role)
        await session.commit()
        await session.refresh(new_role)
        
        console.print(f"[bold green]Success:[/bold green] Created role '{new_role.name}' with ID {new_role.id}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def update_role(args):
    """Update an existing role."""
    engine, session = await setup_db_session()
    
    try:
        # Get role by ID or name
        if args.id:
            stmt = sa.select(Role).where(Role.id == args.id)
        else:
            stmt = sa.select(Role).where(Role.name == args.name)
        
        result = await session.execute(stmt)
        role = result.scalar_one_or_none()
        
        if not role:
            console.print(f"[bold red]Error:[/bold red] Role not found")
            return
        
        # Update role fields
        if args.new_name:
            role.name = args.new_name
        
        if args.description:
            role.description = args.description
        
        if args.permissions:
            permissions = [p.strip() for p in args.permissions.split(',')]
            role.permissions = permissions
        
        await session.commit()
        console.print(f"[bold green]Success:[/bold green] Updated role '{role.name}'")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def list_roles(args):
    """List all roles."""
    engine, session = await setup_db_session()
    
    try:
        # Get all roles
        stmt = sa.select(Role)
        result = await session.execute(stmt)
        roles = result.scalars().all()
        
        # Count users per role
        role_user_counts = {}
        for role in roles:
            stmt = sa.select(sa.func.count()).where(User.role_id == role.id)
            result = await session.execute(stmt)
            role_user_counts[role.id] = result.scalar() or 0
        
        # Create table
        table = Table(title="Roles")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Permissions", style="blue")
        table.add_column("Users", justify="right", style="magenta")
        
        # Add rows
        for role in roles:
            permissions = ", ".join(role.permissions or [])
            table.add_row(
                str(role.id),
                role.name,
                role.description or "",
                permissions,
                str(role_user_counts.get(role.id, 0))
            )
        
        # Print table
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def delete_role(args):
    """Delete a role if it's not in use."""
    engine, session = await setup_db_session()
    
    try:
        # Get role by ID or name
        if args.id:
            stmt = sa.select(Role).where(Role.id == args.id)
        else:
            stmt = sa.select(Role).where(Role.name == args.name)
        
        result = await session.execute(stmt)
        role = result.scalar_one_or_none()
        
        if not role:
            console.print(f"[bold red]Error:[/bold red] Role not found")
            return
        
        # Check if role is in use
        stmt = sa.select(sa.func.count()).where(User.role_id == role.id)
        result = await session.execute(stmt)
        user_count = result.scalar() or 0
        
        if user_count > 0 and not args.force:
            console.print(f"[bold red]Error:[/bold red] Role is assigned to {user_count} users. Use --force to delete anyway.")
            return
        
        # Delete role
        await session.delete(role)
        await session.commit()
        
        console.print(f"[bold green]Success:[/bold green] Deleted role '{role.name}' (ID: {role.id})")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


# Token Management Commands
async def list_tokens(args):
    """List blacklisted tokens."""
    engine, session = await setup_db_session()
    
    try:
        # Get blacklisted tokens
        stmt = sa.select(TokenBlacklist)
        
        if args.active_only:
            now = datetime.utcnow()
            stmt = stmt.where(TokenBlacklist.expires_at > now)
        
        if args.expired_only:
            now = datetime.utcnow()
            stmt = stmt.where(TokenBlacklist.expires_at <= now)
        
        # Apply sorting
        if args.sort_by == "expiry":
            stmt = stmt.order_by(TokenBlacklist.expires_at)
        elif args.sort_by == "id":
            stmt = stmt.order_by(TokenBlacklist.id)
        
        # Get tokens
        result = await session.execute(stmt)
        tokens = result.scalars().all()
        
        # Create table
        table = Table(title="Blacklisted Tokens")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Token ID", style="green")
        table.add_column("Expires At", style="yellow")
        table.add_column("Status", style="magenta")
        
        # Add rows
        now = datetime.utcnow()
        for token in tokens:
            status = "[green]Active[/green]" if token.expires_at > now else "[red]Expired[/red]"
            expires_at = token.expires_at.strftime("%Y-%m-%d %H:%M:%S") if token.expires_at else "N/A"
            
            table.add_row(
                str(token.id),
                str(token.token_id),
                expires_at,
                status
            )
        
        # Print table
        console.print(table)
        console.print(f"Total tokens: {len(tokens)}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def blacklist_token(args):
    """Manually blacklist a token."""
    engine, session = await setup_db_session()
    
    try:
        # Parse or generate token expiry
        if args.expires:
            try:
                expires_at = datetime.fromisoformat(args.expires)
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Invalid expiry format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                return
        else:
            # Default to 1 hour from now
            expires_at = datetime.utcnow() + timedelta(hours=1)
        
        # Create token hash (for real JWTs, you'd extract the exp claim)
        token_id = hash(args.token)
        
        # Create token blacklist entry
        new_entry = TokenBlacklist(
            token_id=token_id,
            expires_at=expires_at
        )
        
        session.add(new_entry)
        await session.commit()
        await session.refresh(new_entry)
        
        console.print(f"[bold green]Success:[/bold green] Token blacklisted until {expires_at}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def remove_token(args):
    """Remove a token from the blacklist."""
    engine, session = await setup_db_session()
    
    try:
        # Get token by ID
        stmt = sa.select(TokenBlacklist).where(TokenBlacklist.id == args.id)
        result = await session.execute(stmt)
        token = result.scalar_one_or_none()
        
        if not token:
            console.print(f"[bold red]Error:[/bold red] Token not found")
            return
        
        # Delete token
        await session.delete(token)
        await session.commit()
        
        console.print(f"[bold green]Success:[/bold green] Token removed from blacklist")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def purge_expired_tokens(args):
    """Purge expired tokens from the blacklist."""
    engine, session = await setup_db_session()
    
    try:
        # Get current time
        now = datetime.utcnow()
        
        # Find expired tokens
        stmt = sa.select(TokenBlacklist).where(TokenBlacklist.expires_at <= now)
        result = await session.execute(stmt)
        expired_tokens = result.scalars().all()
        
        if not expired_tokens:
            console.print("[bold yellow]Info:[/bold yellow] No expired tokens found")
            return
        
        # Delete expired tokens
        for token in expired_tokens:
            await session.delete(token)
        
        await session.commit()
        
        console.print(f"[bold green]Success:[/bold green] Purged {len(expired_tokens)} expired tokens")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


# Database Maintenance Commands
async def stats(args):
    """Show database statistics."""
    engine, session = await setup_db_session()
    
    try:
        # Get counts
        user_stmt = sa.select(sa.func.count()).select_from(User)
        role_stmt = sa.select(sa.func.count()).select_from(Role)
        token_stmt = sa.select(sa.func.count()).select_from(TokenBlacklist)
        active_token_stmt = sa.select(sa.func.count()).where(
            TokenBlacklist.expires_at > datetime.utcnow()
        ).select_from(TokenBlacklist)
        
        user_count = await session.execute(user_stmt)
        role_count = await session.execute(role_stmt)
        token_count = await session.execute(token_stmt)
        active_token_count = await session.execute(active_token_stmt)
        
        user_count = user_count.scalar() or 0
        role_count = role_count.scalar() or 0
        token_count = token_count.scalar() or 0
        active_token_count = active_token_count.scalar() or 0
        
        # Create table
        table = Table(title="Database Statistics")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Users", str(user_count))
        table.add_row("Roles", str(role_count))
        table.add_row("Blacklisted Tokens", str(token_count))
        table.add_row("Active Blacklisted Tokens", str(active_token_count))
        
        # Print table
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


async def vacuum(args):
    """Perform database maintenance."""
    engine, session = await setup_db_session()
    
    try:
        console.print("[bold yellow]Info:[/bold yellow] Running database maintenance...")
        
        # Purge expired tokens
        now = datetime.utcnow()
        stmt = sa.select(TokenBlacklist).where(TokenBlacklist.expires_at <= now)
        result = await session.execute(stmt)
        expired_tokens = result.scalars().all()
        
        for token in expired_tokens:
            await session.delete(token)
        
        await session.commit()
        
        if expired_tokens:
            console.print(f"[bold green]Success:[/bold green] Purged {len(expired_tokens)} expired tokens")
        else:
            console.print("[bold yellow]Info:[/bold yellow] No expired tokens to purge")
        
        # For PostgreSQL, execute VACUUM ANALYZE
        if args.full and 'postgresql' in settings.SQLALCHEMY_DATABASE_URI:
            console.print("[bold yellow]Info:[/bold yellow] Running VACUUM ANALYZE...")
            await session.execute(sa.text("VACUUM ANALYZE"))
            console.print("[bold green]Success:[/bold green] VACUUM ANALYZE completed")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


# Service Commands
def serve(args):
    """Run the authentication service."""
    # Parse host and port
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    
    # Get reload flag
    reload = args.reload
    
    console.print(f"[bold green]Starting server on {host}:{port}[/bold green]")
    
    # Import and run the main application
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


# Main command parser
def get_parser():
    """Create the command line parser."""
    parser = argparse.ArgumentParser(description="AImpact Authentication Service Management Tool")
    subparsers = parser.add_subparsers(title="commands", dest="command")
    
    # User commands
    user_parser = subparsers.add_parser("user", help="User management commands")
    user_subparsers = user_parser.add_subparsers(title="user commands", dest="user_command")
    
    # Create user
    create_user_parser = user_subparsers.add_parser("create", help="Create a new user")
    create_user_parser.add_argument("--email", required=True, help="User email")
    create_user_parser.add_argument("--password", required=True, help="User password")
    create_user_parser.add_argument("--first-name", required=True, help="User first name")
    create_user_parser.add_argument("--last-name", required=True, help="User last name")
    create_user_parser.add_argument("--role-id", type=int, required=True, help="User role ID")
    create_user_parser.add_argument("--inactive", action="store_true", help="Create as inactive user")
    create_user_parser.set_defaults(func=create_user)
    
    # Update user
    update_user_parser = user_subparsers.add_parser("update", help="Update an existing user")
    update_user_parser.add_argument("--id", type=int, help="User ID")
    update_user_parser.add_argument("--email", help="User email (used for lookup if ID not provided)")
    update_user_parser.add_argument("--new-email", help="New email for the user")
    update_user_parser.add_argument("--first-name", help="New first name")
    update_user_parser.add_argument("--last-name", help="New last name")
    update_user_parser.add_argument("--password", help="New password")
    update_user_parser.add_argument("--role-id", type=int, help="New role ID")
    update_user_parser.add_argument("--activate", action="store_true", help="Activate the user")
    update_user_parser.add_argument("--deactivate", action="store_true", help="Deactivate the user")
    update_user_parser.set_defaults(func=update_user)
    
    # List users
    list_users_parser = user_subparsers.add_parser("list", help="List users")
    list_users_parser.add_argument("--role-id", type=int, help="Filter by role ID")
    list_users_parser.add_argument("--active-only", action="store_true", help="Show only active users")
    list_users_parser.add_argument("--inactive-only", action="store_true", help="Show only inactive users")
    list_users_parser.add_argument("--sort-by", choices=["id", "email", "created"], default="id", help="Sort results by")
    list_users_parser.set_defaults(func=list_users)
    
    # Delete user
    delete_user_parser = user_subparsers.add_parser("delete", help="Delete a user")
    delete_user_parser.add_argument("--id", type=int, help="User ID")
    delete_user_parser.add_argument("--email", help="User email (used for lookup if ID not provided)")
    delete_user_parser.set_defaults(func=delete_user)
    
    # Role commands
    role_parser = subparsers.add_parser("role", help="Role management commands")
    role_subparsers = role_parser.add_subparsers(title="role commands", dest="role_command")
    
    # Create role
    create_role_parser = role_subparsers.add_parser("create", help="Create a new role")
    create_role_parser.add_argument("--name", required=True, help="Role name")
    create_role_parser.add_argument("--description", help="Role description")
    create_role_parser.add_argument("--permissions", help="Comma-separated list of permissions")
    create_role_parser.set_defaults(func=create_role)
    
    # Update role
    update_role_parser = role_subparsers.add_parser("update", help="Update an existing role")
    update_role_parser.add_argument("--id", type=int, help="Role ID")
    update_role_parser.add_argument("--name", help="Role name (used for lookup if ID not provided)")
    update_role_parser.add_argument("--new-name", help="New name for the role")
    update_role_parser.add_argument("--description", help="New description")
    update_role_parser.add_argument("--permissions", help="Comma-separated list of permissions")
    update_role_parser.set_defaults(func=update_role)
    
    # List roles
    list_roles_parser = role_subparsers.add_parser("list", help="List roles")
    list_roles_parser.set_defaults(func=list_roles)
    
    # Delete role
    delete_role_parser = role_subparsers.add_parser("delete", help="Delete a role")
    delete_role_parser.add_argument("--id", type=int, help="Role ID")
    delete_role_parser.add_argument("--name", help="Role name (used for lookup if ID not provided)")
    delete_role_parser.add_argument("--force", action="store_true", help="Force delete even if role is assigned to users")
    delete_role_parser.set_defaults(func=delete_role)
    
    # Token commands
    token_parser = subparsers.add_parser("token", help="Token management commands")
    token_subparsers = token_parser.add_subparsers(title="token commands", dest="token_command")
    
    # List tokens
    list_tokens_parser = token_subparsers.add_parser("list", help="List blacklisted tokens")
    list_tokens_parser.add_argument("--active-only", action="store_true", help="Show only active tokens")
    list_tokens_parser.add_argument("--expired-only", action="store_true", help="Show only expired tokens")
    list_tokens_parser.add_argument("--sort-by", choices=["id", "expiry"], default="id", help="Sort results by")
    list_tokens_parser.set_defaults(func=list_tokens)
    
    # Blacklist token
    blacklist_token_parser = token_subparsers.add_parser("blacklist", help="Blacklist a token")
    blacklist_token_parser.add_argument("--token", required=True, help="Token to blacklist")
    blacklist_token_parser.add_argument("--expires", help="Expiry time in ISO format (YYYY-MM-DDTHH:MM:SS)")
    blacklist_token_parser.set_defaults(func=blacklist_token)
    
    # Remove token from blacklist
    remove_token_parser = token_subparsers.add_parser("remove", help="Remove a token from the blacklist")
    remove_token_parser.add_argument("--id", type=int, required=True, help="Token ID")
    remove_token_parser.set_defaults(func=remove_token)
    
    # Purge expired tokens
    purge_tokens_parser = token_subparsers.add_parser("purge", help="Purge expired tokens from the blacklist")
    purge_tokens_parser.set_defaults(func=purge_expired_tokens)
    
    # Validate token
    validate_token_parser = token_subparsers.add_parser("validate", help="Validate a token")
    validate_token_parser.add_argument("--token", required=True, help="Token to validate")
    validate_token_parser.set_defaults(func=validate_token)
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database maintenance commands")
    db_subparsers = db_parser.add_subparsers(title="db commands", dest="db_command")
    
    # Database stats
    stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=stats)
    
    # Database vacuum
    vacuum_parser = db_subparsers.add_parser("vacuum", help="Perform database maintenance")
    vacuum_parser.add_argument("--full", action="store_true", help="Perform full VACUUM ANALYZE (PostgreSQL only)")
    vacuum_parser.set_defaults(func=vacuum)
    
    # Service commands
    serve_parser = subparsers.add_parser("serve", help="Run the authentication service")
    serve_parser.add_argument("--host", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=serve)
    
    return parser


async def validate_token(args):
    """Validate a token and show its payload."""
    engine, session = await setup_db_session()
    
    try:
        # Import JWT for validation
        import jwt
        
        # Check if token is blacklisted
        token_id = hash(args.token)
        stmt = sa.select(TokenBlacklist).where(TokenBlacklist.token_id == token_id)
        result = await session.execute(stmt)
        blacklisted = result.scalar_one_or_none()
        
        if blacklisted:
            if blacklisted.expires_at > datetime.utcnow():
                console.print("[bold red]Token is blacklisted[/bold red]")
            else:
                console.print("[bold yellow]Token is blacklisted but expired[/bold yellow]")
        
        # Try to decode the token
        try:
            # Since we're validating directly, we need to use the project's JWT settings
            secret_key = settings.JWT_SECRET_KEY
            algorithm = settings.JWT_ALGORITHM
            
            # Decode without verification first to show payload
            payload = jwt.decode(args.token, options={"verify_signature": False})
            
            # Create a table for the payload
            table = Table(title="Token Payload")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in payload.items():
                # Format timestamps
                if key in ["exp", "iat"]:
                    try:
                        value = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                table.add_row(key, str(value))
            
            console.print(table)
            
            # Now verify signature
            try:
                jwt.decode(args.token, secret_key, algorithms=[algorithm])
                console.print("[bold green]Token signature is valid[/bold green]")
            except jwt.InvalidSignatureError:
                console.print("[bold red]Token signature is invalid[/bold red]")
            except Exception as e:
                console.print(f"[bold red]Token validation error:[/bold red] {str(e)}")
            
            # Check expiration
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"])
                now = datetime.utcnow()
                
                if exp_time < now:
                    console.print(f"[bold red]Token expired at {exp_time}[/bold red]")
                else:
                    console.print(f"[bold green]Token valid until {exp_time}[/bold green]")
                    console.print(f"[bold green]Time remaining: {exp_time - now}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error decoding token:[/bold red] {str(e)}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    finally:
        await close_db_session(engine, session)


# Main function
async def main_async():
    parser = get_parser()
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    if args.command == "serve":
        args.func(args)
    else:
        await args.func(args)


def main():
    if sys.version_info >= (3, 7):
        asyncio.run(main_async())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async())


if __name__ == "__main__":
    main()

