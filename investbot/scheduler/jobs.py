"""
Scheduled jobs for data ingestion.
Uses APScheduler to run:
- Intraday price refresh every 15-30 min during market hours
- Daily price update after market close
- Quarterly financial statement check
"""
import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from investbot.config.settings import TIMEZONE
from investbot.data.db import SessionLocal
from investbot.data.ingestion.prices import ingest_all_prices
from investbot.data.ingestion.financials import ingest_all_financials

logger = logging.getLogger(__name__)
tz = pytz.timezone(TIMEZONE)


def _job_intraday_refresh():
    """Refresh intraday prices for all equity assets during market hours."""
    logger.info("SCHEDULER: Intraday price refresh triggered")
    session = SessionLocal()
    try:
        from investbot.data.ingestion.prices import ingest_intraday_prices
        from investbot.data.models import Asset
        assets = session.query(Asset).filter_by(is_active=True, asset_type="stock").all()
        for asset in assets:
            try:
                ingest_intraday_prices(session, asset)
            except Exception as e:
                logger.error(f"Intraday refresh failed for {asset.ticker}: {e}")
    finally:
        session.close()


def _job_daily_prices():
    """Update daily prices after market close."""
    logger.info("SCHEDULER: Daily price update triggered")
    session = SessionLocal()
    try:
        ingest_all_prices(session, asset_type="stock")
    except Exception as e:
        logger.error(f"Daily price update failed: {e}")
    finally:
        session.close()


def _job_quarterly_financials():
    """Check for new quarterly financial filings."""
    logger.info("SCHEDULER: Quarterly financial check triggered")
    session = SessionLocal()
    try:
        ingest_all_financials(session, asset_type="stock")
    except Exception as e:
        logger.error(f"Financial ingestion failed: {e}")
    finally:
        session.close()


def create_scheduler() -> BackgroundScheduler:
    """
    Create and configure the scheduler with all jobs.
    Jobs are NOT started — call scheduler.start() separately.
    """
    scheduler = BackgroundScheduler(timezone=tz)

    # Intraday price refresh: every 30 min, Mon-Fri, 9:15 AM - 3:30 PM IST
    scheduler.add_job(
        _job_intraday_refresh,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="0,30",
            timezone=tz,
        ),
        id="intraday_refresh",
        name="Intraday Price Refresh (30 min)",
        replace_existing=True,
    )

    # Daily price update: Mon-Fri at 4:00 PM IST (after market close)
    scheduler.add_job(
        _job_daily_prices,
        CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=0,
            timezone=tz,
        ),
        id="daily_prices",
        name="Daily Price Update (post-close)",
        replace_existing=True,
    )

    # Quarterly financial check: 1st of every month at 6:00 AM IST
    scheduler.add_job(
        _job_quarterly_financials,
        CronTrigger(
            day=1,
            hour=6,
            minute=0,
            timezone=tz,
        ),
        id="quarterly_financials",
        name="Quarterly Financial Statement Check",
        replace_existing=True,
    )

    logger.info("Scheduler configured with 3 jobs")
    return scheduler
