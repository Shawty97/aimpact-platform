"""
Authentication Service Database Connection

This module handles database connection setup, session management,
and migration configuration.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

from .config import settings

logger = logging.getLogger("auth_service.database")

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
        from .models import UserDB, TokenBlacklist, user_roles
        
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


def create_test_db():
    """Create test database for testing."""
    # Import models to register them with Base
    from .models import UserDB, TokenBlacklist, user_roles
    
    # Create all tables in test database
    Base.metadata.create_all(bind=engine)
    return engine

