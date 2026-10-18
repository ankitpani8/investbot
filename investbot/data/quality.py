"""
Price-row data quality: one set of rules shared by ingestion and `main.py dq`.

A row that fails validation is never stored. Ingestion skips it and logs the
reason; because the date is not stored, the next ingest run fetches it again.
Indicator code relies on this: a single NaN close poisons every running-state
TA-Lib indicator (SMA, EMA, MACD, Bollinger) for the rest of the series and
turns others (RSI, HT_TRENDLINE) into plausible-looking wrong numbers.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from investbot.data.corporate_actions import load_actions
from investbot.data.models import Asset, CorporateAction, PriceDaily, PriceIntraday

logger = logging.getLogger(__name__)

# Rejection reasons (stable strings: they are printed by `dq` and asserted in tests).
MISSING_OHLC = "missing_ohlc"
NON_POSITIVE_PRICE = "non_positive_price"
INCONSISTENT_OHLC = "inconsistent_ohlc"
MISSING_VOLUME = "missing_volume"
NEGATIVE_VOLUME = "negative_volume"
ZERO_VOLUME_FLAT = "zero_volume_flat"

# Relative slack for the high/low envelope check; prices come from float32 sources.
_ENVELOPE_TOL = 1e-6


def _is_missing(x) -> bool:
    """True for None, NaN, +/-inf, pd.NA and anything non-numeric."""
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def validate_price_row(o, h, l, c, volume, *, daily: bool) -> str | None:
    """
    Return the rejection reason for one bar, or None if the bar is storable.

    `daily=True` additionally rejects zero-volume bars with O=H=L=C. yfinance
    emits those as carry-forward placeholders for sessions with no real data
    (holidays, and days it failed to populate); they are not trades. Intraday
    bars do not get this rule, a quiet 15m bar can legitimately be flat.
    """
    prices = (o, h, l, c)
    if any(_is_missing(p) for p in prices):
        return MISSING_OHLC
    o, h, l, c = (float(p) for p in prices)
    if min(o, h, l, c) <= 0:
        return NON_POSITIVE_PRICE
    if h < max(o, l, c) * (1 - _ENVELOPE_TOL) or l > min(o, h, c) * (1 + _ENVELOPE_TOL):
        return INCONSISTENT_OHLC

    if _is_missing(volume):
        return MISSING_VOLUME
    if volume < 0:
        return NEGATIVE_VOLUME
    if daily and volume == 0 and o == h == l == c:
        return ZERO_VOLUME_FLAT
    return None


# ──────────────────────────────────────────────────────────
# Database audit
# ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BadRow:
    table: str                       # 'price_daily' | 'price_intraday'
    row_id: int
    asset_id: int
    ticker: str
    when: date | datetime
    reason: str


def audit_prices(
    session: Session,
    *,
    tables: Collection[str] = ("price_daily", "price_intraday"),
    asset_ids: Collection[int] | None = None,
    since: dict[int, date] | None = None,
) -> list[BadRow]:
    """
    Run the ingestion rules over stored price rows. Read-only.

    The defaults audit every row of both tables. `tables` picks `price_daily`
    and/or `price_intraday`; `asset_ids` restricts to those assets; `since[asset_id]`
    restricts that asset's *daily* rows to `date >= since[asset_id]` (assets
    missing from `since`, and intraday rows, are not restricted).
    """
    tickers = dict(session.execute(select(Asset.id, Asset.ticker)).all())
    bad: list[BadRow] = []

    if "price_daily" in tables:
        daily = select(
            PriceDaily.id, PriceDaily.asset_id, PriceDaily.date,
            PriceDaily.open, PriceDaily.high, PriceDaily.low, PriceDaily.close,
            PriceDaily.volume,
        ).order_by(PriceDaily.asset_id, PriceDaily.date)
        if asset_ids is not None:
            daily = daily.where(PriceDaily.asset_id.in_(list(asset_ids)))
        for rid, aid, when, o, h, l, c, v in session.execute(daily).yield_per(5000):
            if since is not None and aid in since and when < since[aid]:
                continue
            reason = validate_price_row(o, h, l, c, v, daily=True)
            if reason:
                bad.append(BadRow("price_daily", rid, aid, tickers.get(aid, str(aid)), when, reason))

    if "price_intraday" in tables:
        intraday = select(
            PriceIntraday.id, PriceIntraday.asset_id, PriceIntraday.timestamp,
            PriceIntraday.open, PriceIntraday.high, PriceIntraday.low, PriceIntraday.close,
            PriceIntraday.volume,
        ).order_by(PriceIntraday.asset_id, PriceIntraday.timestamp)
        if asset_ids is not None:
            intraday = intraday.where(PriceIntraday.asset_id.in_(list(asset_ids)))
        for rid, aid, when, o, h, l, c, v in session.execute(intraday).yield_per(5000):
            reason = validate_price_row(o, h, l, c, v, daily=False)
            if reason:
                bad.append(BadRow("price_intraday", rid, aid, tickers.get(aid, str(aid)), when, reason))

    return bad


def scoring_window_start(
    session: Session, asset_ids: Collection[int], lookback_bars: int,
) -> dict[int, date]:
    """
    Per asset, the date of its `lookback_bars`-th most recent stored daily bar,
    or its first bar if it has fewer. Assets without daily bars are omitted.
    One indexed query per asset; nothing is loaded beyond a single date.
    """
    if lookback_bars < 1:
        raise ValueError("lookback_bars must be >= 1")
    out: dict[int, date] = {}
    for aid in asset_ids:
        start = session.execute(
            select(PriceDaily.date).where(PriceDaily.asset_id == aid)
            .order_by(PriceDaily.date.desc()).offset(lookback_bars - 1).limit(1)
        ).scalar()
        if start is None:
            start = session.execute(
                select(func.min(PriceDaily.date)).where(PriceDaily.asset_id == aid)
            ).scalar()
        if start is not None:
            out[aid] = start
    return out


def purge_bad_rows(session: Session, bad: list[BadRow]) -> int:
    """Delete the given audited rows. Returns the number of rows deleted."""
    ids: dict[str, list[int]] = defaultdict(list)
    for b in bad:
        ids[b.table].append(b.row_id)

    deleted = 0
    for model in (PriceDaily, PriceIntraday):
        row_ids = ids.get(model.__tablename__, [])
        for i in range(0, len(row_ids), 500):          # stay under SQLite's variable limit
            chunk = row_ids[i:i + 500]
            deleted += session.execute(delete(model).where(model.id.in_(chunk))).rowcount
    session.commit()
    return deleted


# ──────────────────────────────────────────────────────────
# Split consistency
# ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitIssue:
    ticker: str
    ex_date: date
    factor: float                    # new/old shares from the corporate action
    observed_ratio: float            # close[prev bar] / open[ex bar]


def detect_unadjusted_splits(
    bars: pd.DataFrame, actions: pd.DataFrame, tol: float = 0.15, ticker: str = "",
) -> list[SplitIssue]:
    """
    Splits whose pre-ex-date history was not re-adjusted.

    For each split with an ex-date inside the stored bars, compare
    `close[prev bar] / open[ex bar]` with the factor. Adjusted history gives a
    ratio near 1; unadjusted history gives a ratio near the factor. A ratio
    within `tol` (relative) of the factor is flagged, and a ratio nearer 1 is not,
    so an ordinary gap day never matches a split of factor 1.5 or more.
    `bars` has a date index and `open`/`close` columns; `actions` is the
    `load_actions` frame. Pure; no NaN dropping (a NaN price is simply not flagged).
    """
    if len(bars) == 0 or len(actions) == 0:
        return []
    splits = actions[actions["action_type"] == "split"]
    if splits.empty:
        return []
    idx = pd.DatetimeIndex(bars.index).normalize()
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    issues: list[SplitIssue] = []
    for ex, factor in zip(pd.DatetimeIndex(splits.index).normalize(), splits["value"].to_numpy(dtype=float)):
        if not factor > 0 or factor == 1.0:
            continue
        i = int(idx.searchsorted(ex, side="left"))
        if i == 0 or i >= len(idx):
            continue                                  # ex-date not inside the stored bars
        o, c = opens[i], closes[i - 1]
        if not (o > 0 and c > 0):
            continue
        ratio = c / o
        if abs(ratio - factor) <= tol * factor:
            issues.append(SplitIssue(ticker, ex.date(), float(factor), float(ratio)))
    return issues


def audit_splits(session: Session) -> list[SplitIssue]:
    """Run `detect_unadjusted_splits` for every asset that has splits, one at a time. Read-only."""
    issues: list[SplitIssue] = []
    rows = session.execute(
        select(Asset.id, Asset.ticker).where(
            Asset.id.in_(select(CorporateAction.asset_id).where(CorporateAction.action_type == "split"))
        ).order_by(Asset.ticker)
    ).all()
    for asset_id, ticker in rows:
        bars = pd.read_sql(
            select(PriceDaily.date, PriceDaily.open, PriceDaily.close)
            .where(PriceDaily.asset_id == asset_id).order_by(PriceDaily.date),
            session.connection(), index_col="date", parse_dates=["date"],
        )
        issues.extend(detect_unadjusted_splits(bars, load_actions(session, ticker), ticker=ticker))
    return issues
