"""
Scheduled jobs for data ingestion.
Uses APScheduler to run:
- Intraday price refresh every 15-30 min during market hours
- Daily post-close pipeline: price ingestion, data-quality audit, technical scoring
- Quarterly financial statement check
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from investbot.config.settings import TIMEZONE
from investbot.data.db import SessionLocal
from investbot.data.ingestion.prices import ingest_all_prices
from investbot.data.ingestion.financials import ingest_all_financials
from investbot.data import quality
from investbot.scoring.technical import scorer

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


@dataclass
class PostCloseResult:
    """Outcome of one post-close pipeline run."""
    ingested: dict | None = None         # whatever ingest_all_prices returned; None if it failed
    bad_rows: int | None = None          # None if the audit never ran
    scored: dict[str, int] | None = None  # status counts; None if scoring was skipped
    skipped_reason: str | None = None    # why scoring (or a later step) did not run


def _log_bad_rows(bad: list) -> None:
    """One ERROR line per (table, ticker) with reason counts, as `main.py dq` prints them."""
    by_key = defaultdict(list)
    for b in bad:
        by_key[(b.table, b.ticker)].append(b)
    for (table, ticker), rows in sorted(by_key.items()):
        reasons = dict(Counter(r.reason for r in rows))
        logger.error(
            f"SCHEDULER: dq {table} {ticker}: {len(rows)} bad row(s) {reasons} "
            f"{rows[0].when} -> {rows[-1].when}"
        )


def run_post_close(session) -> PostCloseResult:
    """
    Post-close pipeline: ingest daily prices, audit them, then refresh daily technical scores.

    Bad rows are logged, never purged, and block scoring. A failing step is logged
    with its name and the later steps are skipped; this function does not raise.
    Intraday scoring is left out on purpose (15m history is too thin for MTF).
    """
    result = PostCloseResult()

    try:
        result.ingested = ingest_all_prices(session, asset_type="stock")
    except Exception:
        logger.exception("SCHEDULER: post-close step 'ingest' failed; audit and scoring skipped")
        result.skipped_reason = "ingest failed"
        return result

    try:
        bad = quality.audit_prices(session)
    except Exception:
        logger.exception("SCHEDULER: post-close step 'audit' failed; scoring skipped")
        result.skipped_reason = "audit failed"
        return result
    result.bad_rows = len(bad)
    if bad:
        _log_bad_rows(bad)
        logger.error(
            f"SCHEDULER: dq found {len(bad)} bad price row(s); scoring skipped. "
            "Run `python main.py dq --purge` and re-ingest."
        )
        result.skipped_reason = f"dq failed: {len(bad)} bad price row(s)"
        return result

    try:
        df = scorer.score_universe(asset_type="stock", timeframe="1d", session=session)
        result.scored = dict(Counter(df["status"]))
    except Exception:
        logger.exception("SCHEDULER: post-close step 'score' failed")
        result.skipped_reason = "score failed"
        return result

    counts = {k: result.scored.get(k, 0)
              for k in ("ok", "warmup", "insufficient_data", "no_data", "error")}
    counts.update({k: v for k, v in result.scored.items() if k not in counts})
    logger.info("SCHEDULER: daily scoring done: " +
                " ".join(f"{k}={v}" for k, v in counts.items()))
    return result


def _job_daily_prices():
    """Post-close pipeline: ingest daily prices, audit them, then score."""
    logger.info("SCHEDULER: Post-close pipeline triggered")
    try:
        session = SessionLocal()
    except Exception:
        logger.exception("SCHEDULER: post-close pipeline could not open a session")
        return
    try:
        run_post_close(session)
    except Exception:
        logger.exception("SCHEDULER: post-close pipeline failed")
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
