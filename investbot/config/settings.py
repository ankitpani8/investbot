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
# "max" (Yahoo has NSE daily history back to the 1990s-2000s for most names): the evaluation
# harness needs decades, not 5 years, to tell a real signal from noise (decided 2026-10-20).
DAILY_PERIOD = "max"

# Yahoo revises recent bars, and a bar fetched mid-session is partial. Every run
# re-fetches and upserts this trailing window; older history stays insert-only.
DAILY_REFRESH_SESSIONS = 5       # last N trading sessions in the fetched daily data
INTRADAY_REFRESH_DAYS = 2        # last N trading days in the fetched intraday data
DAILY_BAR_FINAL_AFTER = "16:00"  # IST; today's daily bar is not stored before this

# An asset that already has stored bars only downloads the last stored bar date minus this
# overlap, instead of DAILY_PERIOD / INTRADAY_PERIOD. The refresh window is applied to the
# last N sessions *in the download*, so the overlap must hold at least that many sessions;
# ingestion checks this and falls back to the full period when it doesn't.
# 14 calendar days = 10 weekdays, so even a long holiday weekend (Diwali / Holi / Christmas
# with the surrounding weekends, 4-5 closed days) leaves >= 5 sessions (DAILY_REFRESH_SESSIONS).
DAILY_INCREMENTAL_OVERLAP_DAYS = 14
# 5 calendar days always contain >= 2 trading days (INTRADAY_REFRESH_DAYS) unless the market
# is shut for 4+ days in a row on top of the weekend; the session-count guard covers that.
INTRADAY_INCREMENTAL_OVERLAP_DAYS = 5
# yfinance only serves 15m bars from the last 60 days. A start date at exactly 60 days back
# is rejected (midnight of that date is a few hours more than 60 days ago), so the largest
# usable start is one day closer.
INTRADAY_MAX_LOOKBACK_DAYS = 60

# ── Scheduler Intervals (minutes) ────────────────────────
EQUITY_SCORING_INTERVAL_MIN = 30
CRYPTO_SCORING_INTERVAL_MIN = 5

# ── Market Hours (IST) ───────────────────────────────────
NSE_MARKET_OPEN = "09:15"
NSE_MARKET_CLOSE = "15:30"
TIMEZONE = "Asia/Kolkata"

# ── Portfolio Constraints ────────────────────────────────
# Pending decision (docs/DECISIONS_PENDING.md, task 130): the plan's user-profile
# defaults are 20% / 40%; these system-wide caps are deliberately tighter.
MAX_POSITION_WEIGHT = 0.15  # max weight of one asset in a target portfolio
MAX_SECTOR_WEIGHT = 0.35    # max combined weight of one sector

# ── Retry Config ──────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
