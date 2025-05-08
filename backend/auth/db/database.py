"""
Authentication Database Connection

This module handles database connection setup, session management,
and migration configuration for the backend auth module.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

from backend.app.core.config import settings

logger = logging.getLogger("backend.auth.db")

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Check connection before using from pool
    echo=settings.DEBUG,  # Log SQL queries if in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()


def init_db():
    """Initialize database tables and run migrations."""
    try:
        # Import models to register them with Base
        from backend.auth.db.models import Tenant, User, TenantUser
        
        # Create all tables (only in development)
        if settings.DEBUG:
            Base.metadata.create_all(bind=engine)
            logger.info("Created database tables")
        else:
            # In production, use Alembic migrations
            run_migrations()
            logger.info("Applied database migrations")
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def run_migrations():
    """Run Alembic migrations."""
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        raise


def get_db_engine():
    """Get database engine."""
    return engine


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

