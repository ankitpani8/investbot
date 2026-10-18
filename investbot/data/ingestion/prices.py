"""
Price data ingestion via yfinance.
Handles daily (5 years) and intraday (60 days of 15-min candles); assets with stored
bars only download the overlap window (see DAILY_INCREMENTAL_OVERLAP_DAYS).
Includes retry logic, gap detection, and a trailing-window upsert
(see DAILY_REFRESH_SESSIONS / INTRADAY_REFRESH_DAYS in settings).
"""
import logging
import time as time_module
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import func, select

from investbot.config.settings import (
    DAILY_BAR_FINAL_AFTER, DAILY_INCREMENTAL_OVERLAP_DAYS, DAILY_PERIOD, DAILY_REFRESH_SESSIONS,
    INTRADAY_INCREMENTAL_OVERLAP_DAYS, INTRADAY_INTERVAL, INTRADAY_MAX_LOOKBACK_DAYS,
    INTRADAY_PERIOD, INTRADAY_REFRESH_DAYS,
    MAX_RETRIES, RETRY_DELAY_SECONDS, TIMEZONE,
)
from investbot.data.models import Asset, Base, CorporateAction, IndexMembership, PriceDaily, PriceIntraday
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


def _download_prices(
    ticker: str, period: str, interval: str, actions: bool = False, start: date | None = None,
) -> pd.DataFrame:
    """
    Download price data from yfinance with retry.

    `actions=True` adds the `Dividends` and `Stock Splits` columns to the same request.
    With `start` the request is `start=` (through today) and `period` is ignored.
    """
    def _do_download():
        window = {"start": start} if start is not None else {"period": period}
        data = yf.download(
            ticker,
            **window,
            interval=interval,
            progress=False,
            auto_adjust=False,
            actions=actions,
            timeout=30,
        )
        if data.empty:
            logger.warning(f"No data returned for {ticker} ({interval})")
        return data

    return _retry(_do_download)


def _download_start(
    session: Session, asset: Asset, interval: str, now: datetime | None = None,
) -> date | None:
    """
    First date to download for an asset that already has history, or None for the full period.

    The last stored bar date minus the overlap (see settings). None when the asset has no
    rows for this table/interval. For intraday also None when that start would fall outside
    yfinance's 15m lookback (INTRADAY_MAX_LOOKBACK_DAYS), which Yahoo rejects.
    """
    if interval == "1d":
        last = session.execute(
            select(func.max(PriceDaily.date)).where(PriceDaily.asset_id == asset.id)
        ).scalar()
        overlap = DAILY_INCREMENTAL_OVERLAP_DAYS
    else:
        last = session.execute(
            select(func.max(PriceIntraday.timestamp)).where(
                PriceIntraday.asset_id == asset.id, PriceIntraday.interval == interval)
        ).scalar()
        overlap = INTRADAY_INCREMENTAL_OVERLAP_DAYS
    if last is None:
        return None

    start = (last.date() if isinstance(last, datetime) else last) - timedelta(days=overlap)
    if interval != "1d" and (_now_ist(now).date() - start).days > INTRADAY_MAX_LOOKBACK_DAYS - 1:
        logger.info(f"{asset.ticker}: {interval} overlap start {start} is outside the "
                    f"{INTRADAY_MAX_LOOKBACK_DAYS}-day limit, downloading {INTRADAY_PERIOD}")
        return None
    return start


@dataclass
class UpsertCounts:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    rejected: list[tuple] = field(default_factory=list)   # (key, reason)


@dataclass
class IngestResult:
    """What one asset's daily or intraday ingestion did (`actions` is daily only)."""
    counts: UpsertCounts = field(default_factory=UpsertCounts)
    actions: UpsertCounts = field(default_factory=UpsertCounts)
    full_downloads: int = 0           # yfinance requests for the full period
    incremental_downloads: int = 0    # yfinance requests with a start date


def _fetch(
    result: IngestResult, ticker: str, period: str, interval: str, start: date | None,
    **kwargs,
) -> pd.DataFrame | None:
    """One counted download with flattened columns; None if it failed after the retries."""
    try:
        if start is None:
            df = _download_prices(ticker, period, interval, **kwargs)
            result.full_downloads += 1
        else:
            df = _download_prices(ticker, period, interval, start=start, **kwargs)
            result.incremental_downloads += 1
    except Exception as e:
        logger.error(f"Failed to download {interval} prices for {ticker}: {e}")
        return None
    if isinstance(df.columns, pd.MultiIndex):       # yfinance, single ticker
        df.columns = df.columns.get_level_values(0)
    return df


def ingest_daily_prices(session: Session, asset: Asset, now: datetime | None = None) -> int:
    """
    Pull daily OHLCV for an asset and store it.

    The last DAILY_REFRESH_SESSIONS sessions in the download are upserted, so a
    revised or partial bar heals on the next run. Older history is insert-only.
    Today's bar (IST) is not stored before DAILY_BAR_FINAL_AFTER, since the
    session is not final yet. `now` overrides the clock (tests).

    Returns the number of rows written (inserted + updated).
    """
    counts = _ingest_daily(session, asset, now).counts
    return counts.inserted + counts.updated


def _ingest_daily(
    session: Session, asset: Asset, now: datetime | None, *, full: bool = False,
) -> IngestResult:
    """
    Daily ingestion with its counts. An asset with stored bars downloads only the overlap
    window; it falls back to the full DAILY_PERIOD (one extra request) when that window holds
    fewer than DAILY_REFRESH_SESSIONS sessions, and re-downloads the full period for its
    corporate actions when the window shows a split that is not stored yet.

    `full=True` downloads DAILY_PERIOD in one request whatever is stored, so neither fallback
    can fire. Storage is unchanged: only the refresh window is upserted, older bars and actions
    are insert-only (a missing old bar or dividend is added, a stored one is never rewritten
    except a dividend whose value differs, as in any run), so history is not re-adjusted.
    """
    ticker = asset.ticker
    logger.info(f"Ingesting daily prices for {ticker}...")
    result = IngestResult()
    now_ist = _now_ist(now)

    start = None if full else _download_start(session, asset, "1d", now)
    df = _fetch(result, ticker, DAILY_PERIOD, "1d", start, actions=True)
    if df is None:
        return result
    bars, skipped = _daily_bars(df, now_ist)

    if start is not None and len(bars) < DAILY_REFRESH_SESSIONS:
        logger.info(f"{ticker}: incremental download from {start} holds {len(bars)} session(s), "
                    f"fewer than the {DAILY_REFRESH_SESSIONS}-session refresh window; "
                    f"downloading {DAILY_PERIOD}")
        start = None
        df = _fetch(result, ticker, DAILY_PERIOD, "1d", None, actions=True)
        if df is None:
            return result
        bars, skipped = _daily_bars(df, now_ist)

    if not bars and df.empty:
        return result

    refresh_from = _refresh_start(sorted(bars), DAILY_REFRESH_SESSIONS, key=lambda d: d)
    result.counts = _upsert_bars(
        session, PriceDaily, PriceDaily.date, bars, refresh_from,
        filters=[PriceDaily.asset_id == asset.id],
        new_row=lambda when, f: PriceDaily(asset_id=asset.id, date=when, **f),
    )
    _log_counts(ticker, "daily", result.counts, skipped, "unfinished session")

    fetched_at = now_ist.astimezone(timezone.utc)
    incoming = _incoming_actions(df, now_ist.date())
    if start is not None:
        new_splits = _new_splits(session, asset, incoming)
        if new_splits:
            logger.info(f"{ticker}: new split(s) {sorted(d.isoformat() for d in new_splits)} in "
                        f"the incremental download; downloading {DAILY_PERIOD} to re-state "
                        f"dividends")
            full = _fetch(result, ticker, DAILY_PERIOD, "1d", None, actions=True)
            if full is not None:
                incoming = _incoming_actions(full, now_ist.date())
            else:
                # Not stored, so the next run sees the split as new and retries the re-download.
                incoming = {k: v for k, v in incoming.items() if k[0] not in new_splits}
    result.actions = _upsert_incoming(session, asset, incoming, fetched_at)
    _log_counts(ticker, "actions", result.actions, 0, "")
    return result


def _daily_bars(df: pd.DataFrame, now_ist: datetime) -> tuple[dict, int]:
    """Validated daily bars by date, plus the number of unfinished (today / future) bars skipped."""
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
    return bars, skipped


def ingest_intraday_prices(session: Session, asset: Asset, now: datetime | None = None) -> int:
    """
    Pull 15-min intraday OHLCV for an asset (last 60 days on the first run, then the overlap).

    Bars on the last INTRADAY_REFRESH_DAYS trading days in the download are
    upserted; older bars are insert-only. The still-forming candle (end time in
    the future) is dropped. Timestamps are stored as naive IST wall time.
    `now` overrides the clock (tests).

    Returns the number of rows written (inserted + updated).
    """
    counts = _ingest_intraday(session, asset, now).counts
    return counts.inserted + counts.updated


def _ingest_intraday(
    session: Session, asset: Asset, now: datetime | None, *, full: bool = False,
) -> IngestResult:
    """Intraday ingestion with its counts; `full=True` downloads INTRADAY_PERIOD in one request."""
    ticker = asset.ticker
    logger.info(f"Ingesting intraday prices for {ticker}...")
    result = IngestResult()
    now_naive = _now_ist(now).replace(tzinfo=None)

    start = None if full else _download_start(session, asset, INTRADAY_INTERVAL, now)
    df = _fetch(result, ticker, INTRADAY_PERIOD, INTRADAY_INTERVAL, start)
    if df is None:
        return result
    bars, skipped = _intraday_bars(df, now_naive)

    if start is not None and len({t.date() for t in bars}) < INTRADAY_REFRESH_DAYS:
        logger.info(f"{ticker}: incremental download from {start} holds fewer than "
                    f"{INTRADAY_REFRESH_DAYS} trading day(s); downloading {INTRADAY_PERIOD}")
        df = _fetch(result, ticker, INTRADAY_PERIOD, INTRADAY_INTERVAL, None)
        if df is None:
            return result
        bars, skipped = _intraday_bars(df, now_naive)

    if not bars and df.empty:
        return result

    refresh_from = _refresh_start(sorted(bars), INTRADAY_REFRESH_DAYS, key=lambda t: t.date())
    result.counts = _upsert_bars(
        session, PriceIntraday, PriceIntraday.timestamp, bars, refresh_from,
        filters=[PriceIntraday.asset_id == asset.id, PriceIntraday.interval == INTRADAY_INTERVAL],
        new_row=lambda when, f: PriceIntraday(
            asset_id=asset.id, timestamp=when, interval=INTRADAY_INTERVAL, **f
        ),
    )
    _log_counts(ticker, INTRADAY_INTERVAL, result.counts, skipped, "forming candle")
    return result


def _intraday_bars(df: pd.DataFrame, now_naive: datetime) -> tuple[dict, int]:
    """Validated intraday bars by naive IST timestamp, plus the number of forming candles skipped."""
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
    return bars, skipped


_INTERVAL_INGESTS = {"daily": _ingest_daily, "intraday": _ingest_intraday}


def member_assets(
    session: Session, index_code: str = "NIFTY50", asset_type: str = "stock",
) -> list[Asset]:
    """
    Inactive assets with at least one `index_membership` row for `index_code`, by ticker.

    These are the past (and not yet seeded as active) members that the normal ingest skips.
    Active members are covered by `ingest_all_prices` already.
    """
    return list(session.execute(
        select(Asset)
        .join(IndexMembership, IndexMembership.asset_id == Asset.id)
        .where(IndexMembership.index_code == index_code,
               Asset.is_active.is_(False), Asset.asset_type == asset_type)
        .distinct().order_by(Asset.ticker)
    ).scalars())


def ingest_all_prices(
    session: Session, asset_type: str = "stock", *, full: bool = False,
    assets: list[Asset] | None = None, intervals: tuple[str, ...] = ("daily", "intraday"),
):
    """
    Ingest daily + intraday prices for all active assets of a given type.

    `full=True` forces the full-period download (one request per asset and interval).
    `assets` replaces the active-asset selection (`asset_type` is then unused), e.g.
    `member_assets(...)`; `intervals` picks "daily" and/or "intraday". `is_active` is never changed.

    `daily_total` / `intraday_total` are rows written (inserted + updated). The
    `*_inserted`, `*_updated`, `*_rejected` keys split them, `actions_*` count
    corporate-action rows, and `full_downloads` / `incremental_downloads` count
    yfinance price requests by kind. `no_data` lists tickers that have no `price_daily`
    rows after a daily ingest (empty or failed download).
    """
    unknown = set(intervals) - set(_INTERVAL_INGESTS)
    if unknown:
        raise ValueError(f"unknown interval(s) {sorted(unknown)}; use 'daily' and/or 'intraday'")
    Base.metadata.create_all(session.get_bind())    # corporate_actions may be missing on an existing DB
    if assets is None:
        assets = session.query(Asset).filter_by(is_active=True, asset_type=asset_type).all()
    logger.info(f"Starting price ingestion for {len(assets)} {asset_type} assets")

    stats = {
        "daily_total": 0, "intraday_total": 0, "errors": [],
        "daily_inserted": 0, "daily_updated": 0, "daily_rejected": 0,
        "intraday_inserted": 0, "intraday_updated": 0, "intraday_rejected": 0,
        "actions_inserted": 0, "actions_updated": 0,
        "full_downloads": 0, "incremental_downloads": 0, "no_data": [],
    }
    for i, asset in enumerate(assets, 1):
        logger.info(f"[{i}/{len(assets)}] Processing {asset.ticker}...")
        try:
            for label in ("daily", "intraday"):
                if label not in intervals:
                    continue
                res = _INTERVAL_INGESTS[label](session, asset, None, full=full)
                stats[f"{label}_inserted"] += res.counts.inserted
                stats[f"{label}_updated"] += res.counts.updated
                stats[f"{label}_rejected"] += len(res.counts.rejected)
                stats[f"{label}_total"] += res.counts.inserted + res.counts.updated
                stats["actions_inserted"] += res.actions.inserted
                stats["actions_updated"] += res.actions.updated
                stats["full_downloads"] += res.full_downloads
                stats["incremental_downloads"] += res.incremental_downloads
            if "daily" in intervals and not session.execute(
                select(func.count()).select_from(PriceDaily).where(PriceDaily.asset_id == asset.id)
            ).scalar():
                stats["no_data"].append(asset.ticker)
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


_ACTION_COLUMNS = (("Dividends", "dividend"), ("Stock Splits", "split"))


def _incoming_actions(df: pd.DataFrame, today: date) -> dict[tuple[date, str], float]:
    """Every non-zero dividend and split of a download by (ex_date, type); later ex-dates are skipped."""
    incoming: dict[tuple[date, str], float] = {}
    for column, action_type in _ACTION_COLUMNS:
        if column not in df.columns:
            continue
        for idx, raw in df[column].items():
            value = _safe_float(raw)
            if value is None or value <= 0:
                continue
            ex_date = idx.date() if hasattr(idx, "date") else idx
            if ex_date > today:
                continue
            incoming[(ex_date, action_type)] = value       # a duplicated index keeps the last row
    return incoming


def _new_splits(session: Session, asset: Asset, incoming: dict[tuple[date, str], float]) -> set[date]:
    """Ex-dates of splits in `incoming` that have no stored (ex_date, 'split') row."""
    splits = {d for d, action_type in incoming if action_type == "split"}
    if not splits:
        return set()
    stored = set(session.execute(
        select(CorporateAction.ex_date).where(
            CorporateAction.asset_id == asset.id, CorporateAction.action_type == "split")
    ).scalars())
    return splits - stored


def _upsert_actions(session: Session, asset: Asset, df: pd.DataFrame, now: datetime) -> UpsertCounts:
    """
    Upsert every non-zero dividend and split row of a daily download.

    Unlike OHLCV there is no refresh window: actions are sparse, and yfinance
    re-states dividend amounts after a later split, so any row may change. An
    existing (ex_date, action_type) row gets its `value` and `fetched_at`
    updated only when the value differs. Rows dated after today (IST) are
    skipped. `now` is the fetch time; a naive value is taken to be UTC.
    """
    now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now
    return _upsert_incoming(session, asset, _incoming_actions(df, _now_ist(now).date()), now)


def _upsert_incoming(
    session: Session, asset: Asset, incoming: dict[tuple[date, str], float], now: datetime,
) -> UpsertCounts:
    """Write `_incoming_actions` rows; `now` is the fetch time (aware, or naive UTC)."""
    counts = UpsertCounts()
    if not incoming:
        return counts
    fetched_at = (now if now.tzinfo is None else now.astimezone(timezone.utc)).replace(tzinfo=None)

    stored = {
        (r.ex_date, r.action_type): r
        for r in session.execute(
            select(CorporateAction).where(CorporateAction.asset_id == asset.id)
        ).scalars()
    }
    for (ex_date, action_type), value in incoming.items():
        row = stored.get((ex_date, action_type))
        if row is None:
            session.add(CorporateAction(
                asset_id=asset.id, ex_date=ex_date, action_type=action_type,
                value=value, source="yfinance", fetched_at=fetched_at,
            ))
            counts.inserted += 1
        elif row.value != value:
            row.value = value
            row.fetched_at = fetched_at
            counts.updated += 1
        else:
            counts.unchanged += 1

    if counts.inserted or counts.updated:
        session.commit()
    return counts


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
