"""
Price data ingestion via yfinance.
Handles daily (5 years) and intraday (60 days of 15-min candles).
Includes retry logic, gap detection, and upsert behavior.
"""
import logging
import time as time_module
from datetime import datetime, date

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import select

from investbot.config.settings import (
    DAILY_PERIOD, INTRADAY_INTERVAL, INTRADAY_PERIOD,
    MAX_RETRIES, RETRY_DELAY_SECONDS,
)
from investbot.data.models import Asset, PriceDaily, PriceIntraday
from investbot.data.quality import validate_price_row

logger = logging.getLogger(__name__)


def _retry(func, *args, max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS, **kwargs):
    """Retry wrapper for flaky API calls."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time_module.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                raise


def _download_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download price data from yfinance with retry."""
    def _do_download():
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            timeout=30,
        )
        if data.empty:
            logger.warning(f"No data returned for {ticker} ({interval})")
        return data

    return _retry(_do_download)


def ingest_daily_prices(session: Session, asset: Asset):
    """
    Pull daily OHLCV for an asset. Upserts — skips existing dates.
    """
    ticker = asset.ticker
    logger.info(f"Ingesting daily prices for {ticker}...")

    try:
        df = _download_prices(ticker, DAILY_PERIOD, "1d")
    except Exception as e:
        logger.error(f"Failed to download daily prices for {ticker}: {e}")
        return 0

    if df.empty:
        return 0

    # Handle MultiIndex columns from yfinance (if single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Get existing dates to avoid duplicates
    existing_dates = set(
        row[0] for row in session.execute(
            select(PriceDaily.date).where(PriceDaily.asset_id == asset.id)
        ).all()
    )

    rows_added = 0
    rejected: list[tuple[date, str]] = []
    for idx, row in df.iterrows():
        row_date = idx.date() if hasattr(idx, 'date') else idx

        if row_date in existing_dates:
            continue

        o, h, l, c = (_safe_float(row.get(k)) for k in ("Open", "High", "Low", "Close"))
        volume = _safe_int(row.get("Volume"))
        reason = validate_price_row(o, h, l, c, volume, daily=True)
        if reason:
            # Not stored, so the next ingest run fetches this date again.
            rejected.append((row_date, reason))
            continue

        session.add(PriceDaily(
            asset_id=asset.id,
            date=row_date,
            open=o, high=h, low=l, close=c,
            adj_close=_safe_float(row.get("Adj Close")),
            volume=volume,
        ))
        rows_added += 1

    _log_rejected(ticker, "daily", rejected)
    if rows_added > 0:
        session.commit()
    logger.info(f"{ticker}: added {rows_added} daily price rows")
    return rows_added


def ingest_intraday_prices(session: Session, asset: Asset):
    """
    Pull 15-min intraday OHLCV for an asset (last 60 days).
    Upserts — skips existing timestamps.
    """
    ticker = asset.ticker
    logger.info(f"Ingesting intraday prices for {ticker}...")

    try:
        df = _download_prices(ticker, INTRADAY_PERIOD, INTRADAY_INTERVAL)
    except Exception as e:
        logger.error(f"Failed to download intraday prices for {ticker}: {e}")
        return 0

    if df.empty:
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Get existing timestamps
    existing_ts = set(
        row[0] for row in session.execute(
            select(PriceIntraday.timestamp).where(
                PriceIntraday.asset_id == asset.id,
                PriceIntraday.interval == INTRADAY_INTERVAL,
            )
        ).all()
    )

    rows_added = 0
    rejected: list[tuple[datetime, str]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        # Make timezone-naive for SQLite storage
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)

        if ts in existing_ts:
            continue

        o, h, l, c = (_safe_float(row.get(k)) for k in ("Open", "High", "Low", "Close"))
        volume = _safe_int(row.get("Volume"))
        reason = validate_price_row(o, h, l, c, volume, daily=False)
        if reason:
            rejected.append((ts, reason))
            continue

        session.add(PriceIntraday(
            asset_id=asset.id,
            timestamp=ts,
            interval=INTRADAY_INTERVAL,
            open=o, high=h, low=l, close=c,
            volume=volume,
        ))
        rows_added += 1

    _log_rejected(ticker, INTRADAY_INTERVAL, rejected)
    if rows_added > 0:
        session.commit()
    logger.info(f"{ticker}: added {rows_added} intraday price rows")
    return rows_added


def ingest_all_prices(session: Session, asset_type: str = "stock"):
    """
    Ingest daily + intraday prices for all active assets of a given type.
    """
    assets = session.query(Asset).filter_by(is_active=True, asset_type=asset_type).all()
    logger.info(f"Starting price ingestion for {len(assets)} {asset_type} assets")

    stats = {"daily_total": 0, "intraday_total": 0, "errors": []}
    for i, asset in enumerate(assets, 1):
        logger.info(f"[{i}/{len(assets)}] Processing {asset.ticker}...")
        try:
            stats["daily_total"] += ingest_daily_prices(session, asset)
            stats["intraday_total"] += ingest_intraday_prices(session, asset)
        except Exception as e:
            logger.error(f"Error processing {asset.ticker}: {e}")
            stats["errors"].append(asset.ticker)

        # Small delay to avoid rate-limiting
        if i < len(assets):
            time_module.sleep(1)

    logger.info(
        f"Price ingestion complete: {stats['daily_total']} daily rows, "
        f"{stats['intraday_total']} intraday rows, {len(stats['errors'])} errors"
    )
    if stats["errors"]:
        logger.warning(f"Failed tickers: {stats['errors']}")
    return stats


def detect_gaps(session: Session, asset: Asset, max_gap_days: int = 5) -> list[tuple]:
    """
    Detect gaps in daily price data larger than max_gap_days (excluding weekends).
    Returns list of (gap_start, gap_end, gap_business_days).
    """
    prices = session.query(PriceDaily.date).filter_by(
        asset_id=asset.id
    ).order_by(PriceDaily.date).all()

    if len(prices) < 2:
        return []

    gaps = []
    dates = [p[0] for p in prices]
    for i in range(1, len(dates)):
        bdays = pd.bdate_range(dates[i-1], dates[i]).shape[0] - 1
        if bdays > max_gap_days:
            gaps.append((dates[i-1], dates[i], bdays))

    return gaps


# ── Helpers ───────────────────────────────────────────────

def _log_rejected(ticker: str, label: str, rejected: list[tuple]) -> None:
    """One warning per asset: counts by reason plus the first few timestamps."""
    if not rejected:
        return
    by_reason: dict[str, int] = {}
    for _, reason in rejected:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    sample = ", ".join(str(when) for when, _ in rejected[:5])
    more = f" (+{len(rejected) - 5} more)" if len(rejected) > 5 else ""
    logger.warning(
        f"{ticker}: rejected {len(rejected)} {label} bar(s) {by_reason}: {sample}{more}"
    )


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert to int, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else int(f)
    except (ValueError, TypeError):
        return None
