"""
Database engine, session factory, and table creation.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.config.settings import DATABASE_URL
from investbot.data.models import Base

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Yield a new session (use as context manager)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
