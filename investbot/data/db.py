"""
Database engine, session factory, and table creation.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.config.settings import DATABASE_URL
from investbot.data.migrations.financials_fy_end import migrate_financials_fiscal_year_end
from investbot.data.migrations.financials_pit import migrate_financials_point_in_time
from investbot.data.models import Base

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """Create all tables if they don't exist, then apply the idempotent migrations."""
    Base.metadata.create_all(bind=engine)
    migrate_financials_point_in_time(engine)
    migrate_financials_fiscal_year_end(engine)


def get_session():
    """Yield a new session (use as context manager)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
