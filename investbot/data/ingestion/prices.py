"""
Price data ingestion via yfinance.
Handles daily (5 years) and intraday (60 days of 15-min candles).
Includes retry logic, gap detection, and a trailing-window upsert
(see DAILY_REFRESH_SESSIONS / INTRADAY_REFRESH_DAYS in settings).
"""
import logging
import time as time_module
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import select

from investbot.config.settings import (
    DAILY_BAR_FINAL_AFTER, DAILY_PERIOD, DAILY_REFRESH_SESSIONS,
    INTRADAY_INTERVAL, INTRADAY_PERIOD, INTRADAY_REFRESH_DAYS,
    MAX_RETRIES, RETRY_DELAY_SECONDS, TIMEZONE,
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


def ingest_daily_prices(session: Session, asset: Asset, now: datetime | None = None) -> int:
    """
    Pull daily OHLCV for an asset and store it.

    The last DAILY_REFRESH_SESSIONS sessions in the download are upserted, so a
    revised or partial bar heals on the next run. Older history is insert-only.
    Today's bar (IST) is not stored before DAILY_BAR_FINAL_AFTER, since the
    session is not final yet. `now` overrides the clock (tests).

    Returns the number of rows written (inserted + updated).
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

    now_ist = _now_ist(now)
    today_unfinished = now_ist.time() < _parse_hhmm(DAILY_BAR_FINAL_AFTER)

    bars: dict[date, tuple[dict, str | None]] = {}
    skipped = 0
    for idx, row in df.iterrows():
        row_date = idx.date() if hasattr(idx, 'date') else idx
        if row_date > now_ist.date() or (row_date == now_ist.date() and today_unfinished):
            skipped += 1
            continue

        o, h, l, c = (_safe_float(row.get(k)) for k in ("Open", "High", "Low", "Close"))
        volume = _safe_int(row.get("Volume"))
        reason = validate_price_row(o, h, l, c, volume, daily=True)
        fields = dict(open=o, high=h, low=l, close=c,
                      adj_close=_safe_float(row.get("Adj Close")), volume=volume)
        bars[row_date] = (fields, reason)       # a duplicated index keeps the last bar

    refresh_from = _refresh_start(sorted(bars), DAILY_REFRESH_SESSIONS, key=lambda d: d)
    counts = _upsert_bars(
        session, PriceDaily, PriceDaily.date, bars, refresh_from,
        filters=[PriceDaily.asset_id == asset.id],
        new_row=lambda when, f: PriceDaily(asset_id=asset.id, date=when, **f),
    )
    _log_counts(ticker, "daily", counts, skipped, "unfinished session")
    return counts.inserted + counts.updated


def ingest_intraday_prices(session: Session, asset: Asset, now: datetime | None = None) -> int:
    """
    Pull 15-min intraday OHLCV for an asset (last 60 days).

    Bars on the last INTRADAY_REFRESH_DAYS trading days in the download are
    upserted; older bars are insert-only. The still-forming candle (end time in
    the future) is dropped. Timestamps are stored as naive IST wall time.
    `now` overrides the clock (tests).

    Returns the number of rows written (inserted + updated).
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

    now_naive = _now_ist(now).replace(tzinfo=None)
    bar_length = pd.Timedelta(INTRADAY_INTERVAL).to_pytimedelta()

    bars: dict[datetime, tuple[dict, str | None]] = {}
    skipped = 0
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx

        # Naive IST wall time for SQLite storage
        if getattr(ts, 'tzinfo', None) is not None:
            ts = ts.astimezone(_IST).replace(tzinfo=None)

        if ts + bar_length > now_naive:
            skipped += 1                          # candle still forming
            continue

        o, h, l, c = (_safe_float(row.get(k)) for k in ("Open", "High", "Low", "Close"))
        volume = _safe_int(row.get("Volume"))
        reason = validate_price_row(o, h, l, c, volume, daily=False)
        bars[ts] = (dict(open=o, high=h, low=l, close=c, volume=volume), reason)

    refresh_from = _refresh_start(sorted(bars), INTRADAY_REFRESH_DAYS, key=lambda t: t.date())
    counts = _upsert_bars(
        session, PriceIntraday, PriceIntraday.timestamp, bars, refresh_from,
        filters=[PriceIntraday.asset_id == asset.id, PriceIntraday.interval == INTRADAY_INTERVAL],
        new_row=lambda when, f: PriceIntraday(
            asset_id=asset.id, timestamp=when, interval=INTRADAY_INTERVAL, **f
        ),
    )
    _log_counts(ticker, INTRADAY_INTERVAL, counts, skipped, "forming candle")
    return counts.inserted + counts.updated


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

_IST = ZoneInfo(TIMEZONE)


@dataclass
class UpsertCounts:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    rejected: list[tuple] = field(default_factory=list)   # (key, reason)


def _now_ist(now: datetime | None) -> datetime:
    """Current time in IST. A naive `now` is taken to be IST already."""
    if now is None:
        return datetime.now(_IST)
    return now.replace(tzinfo=_IST) if now.tzinfo is None else now.astimezone(_IST)


def _parse_hhmm(value: str) -> dt_time:
    hh, mm = value.split(":")
    return dt_time(int(hh), int(mm))


def _refresh_start(keys: list, n_sessions: int, key):
    """
    First key inside the refresh window: the last `n_sessions` distinct sessions
    (`key(k)` maps a bar key to its session date) among the fetched bars.
    Returns None when there is nothing to refresh.
    """
    if n_sessions <= 0 or not keys:
        return None
    sessions = sorted({key(k) for k in keys})
    first_session = sessions[-min(n_sessions, len(sessions))]
    return next(k for k in keys if key(k) >= first_session)


def _upsert_bars(session, model, key_col, bars: dict, refresh_from, *, filters, new_row) -> UpsertCounts:
    """
    Write validated bars. Keys >= `refresh_from` are upserted; older keys are
    insert-only. A bar that fails validation is never written, so it can't
    replace a good stored row. Only rows whose values differ are updated, so a
    repeated run with the same download changes nothing.
    """
    counts = UpsertCounts()
    if not bars:
        return counts

    existing_keys = set(session.execute(select(key_col).where(*filters)).scalars())
    in_window: dict = {}
    if refresh_from is not None:
        in_window = {
            getattr(r, key_col.key): r
            for r in session.execute(
                select(model).where(*filters, key_col >= refresh_from)
            ).scalars()
        }

    for when, (fields, reason) in bars.items():
        if reason:
            # Not written: the stored row (if any) stays, and a missing date is re-fetched next run.
            counts.rejected.append((when, reason))
            continue
        stored = in_window.get(when)
        if stored is not None:
            if any(getattr(stored, k) != v for k, v in fields.items()):
                for k, v in fields.items():
                    setattr(stored, k, v)
                counts.updated += 1
            else:
                counts.unchanged += 1
        elif when in existing_keys:
            counts.unchanged += 1                  # outside the refresh window: insert-only
        else:
            session.add(new_row(when, fields))
            counts.inserted += 1

    if counts.inserted or counts.updated:
        session.commit()
    return counts


def _log_counts(ticker: str, label: str, counts: UpsertCounts, skipped: int, skip_reason: str) -> None:
    _log_rejected(ticker, label, counts.rejected)
    extra = f", skipped {skipped} ({skip_reason})" if skipped else ""
    logger.info(
        f"{ticker} {label}: inserted {counts.inserted}, updated {counts.updated}, "
        f"rejected {len(counts.rejected)}{extra}"
    )


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
