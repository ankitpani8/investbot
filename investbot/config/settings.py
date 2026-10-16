"""
Global configuration loaded from .env and defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = parent of investbot package (on Google Drive)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Database ──────────────────────────────────────────────
# IMPORTANT: SQLite DB is stored OUTSIDE Google Drive.
# Drive's real-time file sync corrupts SQLite files while they are open.
# Code lives on Drive (version-controlled via git); data lives locally.
#
# Default: ~/investbot_data/investbot.db (user's home folder, local disk)
# Override via DATABASE_URL in .env if desired.
DEFAULT_DATA_DIR = Path.home() / "investbot_data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "investbot.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Data Ingestion ────────────────────────────────────────
INTRADAY_INTERVAL = "15m"
INTRADAY_PERIOD = "60d"
DAILY_PERIOD = "5y"

# ── Scheduler Intervals (minutes) ────────────────────────
EQUITY_SCORING_INTERVAL_MIN = 30
CRYPTO_SCORING_INTERVAL_MIN = 5

# ── Market Hours (IST) ───────────────────────────────────
NSE_MARKET_OPEN = "09:15"
NSE_MARKET_CLOSE = "15:30"
TIMEZONE = "Asia/Kolkata"

# ── Retry Config ──────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
