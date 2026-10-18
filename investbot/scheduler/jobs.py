"""
Scheduled jobs for data ingestion.
Uses APScheduler to run:
- Intraday price refresh every 15-30 min during market hours
- Daily post-close pipeline: price ingestion, data-quality audit, technical scoring
- Quarterly financial statement check
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from investbot.config.settings import TIMEZONE
from investbot.data.db import SessionLocal
from investbot.data.ingestion.prices import ingest_all_prices
from investbot.data.ingestion.financials import ingest_all_financials
from investbot.data import quality
from investbot.data.models import Asset
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
    bad_rows: int | None = None          # bad rows in the scoring gate; None if the gate never ran
    excluded: list[str] = field(default_factory=list)  # tickers left out of scoring by the gate
    report_only_bad_rows: int = 0        # bad rows outside the gate (old daily, intraday); never block
    split_issues: int = 0                # unadjusted splits found; never block
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


def _log_report_only(bad: list) -> None:
    """One WARNING line per (table, ticker) for bad rows that do not block scoring."""
    by_key = defaultdict(list)
    for b in bad:
        by_key[(b.table, b.ticker)].append(b)
    for (table, ticker), rows in sorted(by_key.items()):
        reasons = dict(Counter(r.reason for r in rows))
        logger.warning(
            f"SCHEDULER: dq (report only, not blocking) {table} {ticker}: "
            f"{len(rows)} bad row(s) {reasons} {rows[0].when} -> {rows[-1].when}"
        )


def run_post_close(session) -> PostCloseResult:
    """
    Post-close pipeline: ingest daily prices, audit them, then refresh daily technical scores.

    Only what daily scoring reads gates it: the daily bars inside each active stock's
    scoring lookback. An asset with a bad row there is excluded from scoring (one ERROR
    per asset); the rest are scored. Bad rows anywhere else (older daily bars, all 15m
    bars) and unadjusted splits are logged and never block. Nothing is purged.
    A failing step is logged with its name and the later steps are skipped; this
    function does not raise. Intraday scoring is left out on purpose (15m history is
    too thin for MTF).
    """
    result = PostCloseResult()

    try:
        result.ingested = ingest_all_prices(session, asset_type="stock")
    except Exception:
        logger.exception("SCHEDULER: post-close step 'ingest' failed; audit and scoring skipped")
        result.skipped_reason = "ingest failed"
        return result

    try:
        assets = dict(session.execute(
            select(Asset.id, Asset.ticker)
            .where(Asset.is_active.is_(True), Asset.asset_type == "stock")
            .order_by(Asset.ticker)
        ).all())
        since = quality.scoring_window_start(session, assets, scorer.DAILY_LOOKBACK_BARS)
        gate = quality.audit_prices(session, tables=("price_daily",),
                                    asset_ids=list(assets), since=since)
    except Exception:
        logger.exception("SCHEDULER: post-close step 'audit' failed; scoring skipped")
        result.skipped_reason = "audit failed"
        return result
    result.bad_rows = len(gate)
    result.excluded = sorted({b.ticker for b in gate})
    if gate:
        _log_bad_rows(gate)
        logger.error(
            f"SCHEDULER: dq found {len(gate)} bad row(s) in the scoring window; excluded from "
            f"scoring: {', '.join(result.excluded)}. Run `python main.py dq --purge` and re-ingest."
        )

    try:
        gated = {(b.table, b.row_id) for b in gate}
        report_only = [b for b in quality.audit_prices(session)
                       if (b.table, b.row_id) not in gated]
        result.report_only_bad_rows = len(report_only)
        _log_report_only(report_only)
        splits = quality.audit_splits(session)
        result.split_issues = len(splits)
        for i in splits:
            logger.error(
                f"SCHEDULER: dq split {i.ticker} ex {i.ex_date}: history looks unadjusted "
                f"(factor {i.factor:g}, observed ratio {i.observed_ratio:.3g}); not blocking"
            )
    except Exception:
        logger.exception("SCHEDULER: post-close step 'report-only audit' failed (not blocking)")

    to_score = [t for t in assets.values() if t not in result.excluded]
    if not to_score:
        logger.error("SCHEDULER: no asset to score; daily scoring skipped")
        result.skipped_reason = ("dq failed: every asset has bad rows in the scoring window"
                                 if assets else "no active stocks")
        return result

    try:
        result.scored = dict(Counter(
            scorer.score(t, "1d", session=session).status for t in to_score
        ))
    except Exception:
        logger.exception("SCHEDULER: post-close step 'score' failed")
        result.scored = None
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
