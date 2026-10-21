"""
Backtest inputs from the database: the wide frames ``simulator.simulate`` takes.

This is the only backtest module that touches the DB. It reads through the evaluation
loader (one asset at a time: 8 GB RAM), ``corporate_actions`` and the index-membership
tables, and builds sessions x asset_id frames (``open_``, ``close``, ``dividends``,
``membership``) plus the 31-Jan-2018 FMV per asset. Every other step is a pure function.

It only exposes data. Prices are not adjusted, masked or dropped: share-basis problems
(``detect_unadjusted_splits``) are reported in ``split_issues``, and stale marks, delisting
exits and price anomalies are left to the simulator and DECISIONS_PENDING. No ``dropna``:
a missing bar stays NaN.

``end`` is applied in SQL (a load with ``end=s`` reads nothing after ``s``), ``start`` in
pandas after loading. Dividends are placed on the full bar index before the ``start`` cut,
so a dividend that went ex before ``start`` is never credited to the first bar in range.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.data.corporate_actions import dividends_on_bars
from investbot.data.index_membership import COMPLETE_FROM
from investbot.data.models import Asset
from investbot.data.quality import SplitIssue, detect_unadjusted_splits
from investbot.evaluation import loader
from investbot.trading.grandfathering import fmv_31_jan_2018

log = logging.getLogger(__name__)

UNIVERSES = ("index", "active")


@dataclass
class BacktestInputs:
    open_: pd.DataFrame                      # sessions x asset_id, NaN where the asset has no bar
    close: pd.DataFrame
    dividends: pd.DataFrame                  # gross cash per share on the ex-date bar; 0.0 on other bars
    membership: pd.DataFrame                 # bool, member of the index on that session
    tickers: dict[int, str]
    fmv_2018_01_31: dict[int, float]         # assets without FMV are absent
    split_issues: list[SplitIssue]           # reported only; prices are untouched
    fingerprint: dict
    complete_from: date                      # membership is known only from this date
    missing: list[str] = field(default_factory=list)   # universe tickers with no price rows (no column)


@dataclass(frozen=True)
class AssetColumns:
    """One asset's contribution to the wide frames (all indexed by its own bars)."""
    open_: pd.Series
    close: pd.Series
    dividends: pd.Series
    fmv: float | None
    split_issues: list[SplitIssue]


# ──────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────

def union_sessions(indexes: Iterable[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Sorted union of the given session indexes (an empty DatetimeIndex for none)."""
    out = pd.DatetimeIndex([])
    for ix in indexes:
        out = out.union(pd.DatetimeIndex(ix))
    out.name = None
    return out


def wide_frame(per_asset: dict[int, pd.Series]) -> pd.DataFrame:
    """Sessions x asset_id frame from one Series per asset, with one ``pd.concat``.

    The index is the sorted union of the series' indexes, the columns are the
    ``asset_id`` keys in ascending order; a session an asset has no value for is NaN.
    """
    ids = sorted(per_asset)
    index = union_sessions(per_asset[a].index for a in ids)
    if not ids:
        return pd.DataFrame(index=index, columns=pd.Index([], dtype="int64", name="asset_id"),
                            dtype=float)
    out = pd.concat({int(a): per_asset[a].astype(float) for a in ids}, axis=1).reindex(index)
    out.columns = pd.Index([int(a) for a in ids], dtype="int64", name="asset_id")
    return out


def asset_columns(bars: pd.DataFrame, actions: pd.DataFrame, start: date | None = None,
                  ticker: str = "") -> AssetColumns:
    """Cut one asset's raw frames into its wide-frame columns. Pure.

    ``bars`` (date index; ``open``, ``high``, ``close``) and ``actions`` are what the
    loaders return, already truncated at ``end``. FMV uses all bars, including those
    before ``start``; dividends are placed on the full bar index and then cut at ``start``;
    split issues are looked for in the bars kept.
    """
    fmv = fmv_31_jan_2018(bars)
    divs = pd.Series(dividends_on_bars(bars.index, actions), index=bars.index, dtype=float)
    if start is not None:
        keep = bars.index >= pd.Timestamp(start)
        bars, divs = bars[keep], divs[keep]
    issues = detect_unadjusted_splits(bars, actions, ticker=ticker)
    return AssetColumns(bars["open"], bars["close"], divs, fmv, issues)


# ──────────────────────────────────────────────────────────
# DB reading
# ──────────────────────────────────────────────────────────

def _asset_ids(session: Session, tickers: list[str]) -> dict[str, int]:
    rows = session.execute(
        select(Asset.ticker, Asset.id).where(Asset.ticker.in_(list(tickers)))).all()
    return {t: int(i) for t, i in rows}


def load_backtest_inputs(
    session: Session,
    *,
    index_code: str = "NIFTY50",
    universe: str = "index",
    start: date | None = None,
    end: date | None = None,
) -> BacktestInputs:
    """Build the ``simulate`` inputs for a universe over [start, end] (open-ended if None).

    ``universe="index"``: every ticker with a stint in ``index_code`` overlapping the
    window, past members included. ``universe="active"``: the active stocks, with
    ``membership`` all True. Tickers with no price rows are listed in ``missing`` (and
    logged) and get no column.
    """
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}, got {universe!r}")
    if universe == "index":
        names = loader.universe_tickers(session, index_code, start, end)
    else:
        names = loader.active_tickers(session)
    ids = _asset_ids(session, names)

    opens: dict[int, pd.Series] = {}
    closes: dict[int, pd.Series] = {}
    divs: dict[int, pd.Series] = {}
    fmv: dict[int, float] = {}
    issues: list[SplitIssue] = []
    tickers: dict[int, str] = {}
    missing: list[str] = []
    for ticker in names:
        bars = loader.load_daily_history(session, ticker, end)
        if len(bars) == 0 or ticker not in ids:
            missing.append(ticker)
            continue
        actions = loader.load_actions(session, ticker, end)
        asset_id = ids[ticker]
        cols = asset_columns(bars, actions, start, ticker)
        del bars, actions                                  # one raw frame in memory at a time
        opens[asset_id], closes[asset_id], divs[asset_id] = cols.open_, cols.close, cols.dividends
        if cols.fmv is not None:
            fmv[asset_id] = cols.fmv
        issues.extend(cols.split_issues)
        tickers[asset_id] = ticker
    if missing:
        log.warning("%d universe ticker(s) have no price rows and get no column: %s",
                    len(missing), ", ".join(missing))

    open_, close, dividends = wide_frame(opens), wide_frame(closes), wide_frame(divs)
    sessions = open_.index
    if universe == "index":
        by_ticker = loader.membership_for(session, index_code, list(tickers.values()), sessions)
        membership = by_ticker.rename(columns={t: a for a, t in tickers.items()})
        membership = membership.reindex(columns=open_.columns).astype(bool)
        complete_from = COMPLETE_FROM
    else:
        membership = pd.DataFrame(True, index=sessions, columns=open_.columns)
        complete_from = sessions[0].date() if len(sessions) else COMPLETE_FROM
    membership.columns.name = "asset_id"

    return BacktestInputs(
        open_=open_, close=close, dividends=dividends, membership=membership,
        tickers=tickers, fmv_2018_01_31=fmv, split_issues=issues,
        fingerprint=loader.db_fingerprint(session), complete_from=complete_from,
        missing=missing,
    )
