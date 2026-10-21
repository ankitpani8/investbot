"""
Report-only audit of suspicious daily prices (`python main.py dq`, section "Price anomalies").

`quality.py` checks single rows and unadjusted splits. This module looks at the price *path* of
one asset: bad ticks that revert, unexplained level breaks and frozen closes. It only reports.
It does not purge, mask or change any harness input; what to do with a flagged bar is a
separate decision (docs/DECISIONS_PENDING.md).

**Offline data inspection only.** `spike_reversal` uses bar t+1 to judge bar t, so it looks
one bar into the future by construction. It must never be used as a real-time feature, nor as
a backtest mask or label filter, without an explicit decision: masking bars by their own
future would itself be lookahead.

Kinds (stable strings, asserted in tests):

- `spike_reversal`: close-to-close log return at bar t of at least `SPIKE_MIN_ABS_LOG_RET`
  whose next bar's return has the opposite sign and at least `SPIKE_REVERT_SHARE` of its
  magnitude. Flagged at bar t (the bad tick). Corporate actions do not suppress it: a split
  does not come back the next day.
- `extreme_move`: log return of at least `EXTREME_ABS_LOG_RET` that is not a spike_reversal,
  is not the reverting bar of one (that bar belongs to the spike), has no corporate action
  (`split`, `bonus` if stored, `dividend`) with an ex-date within `ACTION_WINDOW_SESSIONS`
  sessions of the bar, and is not an unadjusted split found by `quality.detect_unadjusted_splits`.
- `stale_run`: `STALE_RUN_MIN_SESSIONS` or more consecutive bars with the same close (relative
  difference to the previous bar at most `STALE_RUN_REL_TOL`). `sessions` counts the bars
  sharing the frozen close, the first included; `start`/`end` are the first and last of them.

Market-wide tag (still report-only): for the return kinds, `market_share` is the share of the
*other* priced tickers that moved at least `MARKET_MOVE_ABS_LOG_RET` the same way on the anomaly's
date, and `market_wide` is set when that share is at least `MARKET_WIDE_MIN_SHARE` and at least
`MARKET_WIDE_MIN_TICKERS` other tickers were priced. It separates real events (2004-05-17, the
election-result crash) from single-name bad data; it removes nothing and it is a same-day
cross-section, so it adds no lookahead beyond the next-bar rule of `spike_reversal`. `stale_run`
keeps NaN / False. The "other tickers" are the assets the audit loaded, so `tickers=` narrows them.

A return needs two valid closes (finite and positive). A NaN or non-positive close gives no
flag, ends any stale run, and is never dropped from the frame (no NaN dropping here).

Pure detector first (`detect_price_anomalies`, `daily_move_counts`, `tag_market_wide`,
`summarise_anomalies`); the DB loader
(`audit_price_anomalies`) is at the bottom of the file.
"""
from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.data.corporate_actions import load_actions
from investbot.data.models import Asset, PriceDaily
from investbot.data.quality import detect_unadjusted_splits

# A close-to-close log return at least this large (absolute) is a spike candidate.
SPIKE_MIN_ABS_LOG_RET: float = 0.20
# The spike is a `spike_reversal` when the next return has the opposite sign and at least this
# share of the spike's magnitude.
SPIKE_REVERT_SHARE: float = 0.75
# A log return at least this large (absolute), not a spike_reversal and without a corporate
# action nearby, is an `extreme_move`.
EXTREME_ABS_LOG_RET: float = 0.40
# A corporate action explains a move when its ex-date bar is within this many sessions of it.
ACTION_WINDOW_SESSIONS: int = 1
# This many or more consecutive bars with an unchanged close make a `stale_run`.
STALE_RUN_MIN_SESSIONS: int = 5
# Relative difference up to which two closes count as unchanged.
STALE_RUN_REL_TOL: float = 1e-9

# A ticker "moved hard" on a day when its close-to-close |log return| is at least this.
MARKET_MOVE_ABS_LOG_RET: float = 0.10
# A return anomaly is market-wide when at least this share of the other priced tickers moved hard
# the same way that day...
MARKET_WIDE_MIN_SHARE: float = 0.20
# ...and at least this many other tickers were priced (a share of 3 tickers means nothing).
MARKET_WIDE_MIN_TICKERS: int = 5

ANOMALY_KINDS = ("spike_reversal", "extreme_move", "stale_run")
_RETURN_KINDS = ("spike_reversal", "extreme_move")
_EXPLAINING_ACTIONS = ("split", "bonus", "dividend")
SUMMARY_TOTAL_ROW = "total"
SUMMARY_TICKERS_ROW = "tickers"


@dataclass(frozen=True)
class PriceAnomaly:
    ticker: str
    kind: str                        # one of ANOMALY_KINDS
    start: date
    end: date                        # == start for the return kinds
    sessions: int                    # 1 for the return kinds, bars with the frozen close for stale_run
    value: float                     # triggering log return; frozen close for stale_run
    # Set by tag_market_wide; NaN / False until then and always for stale_run.
    market_share: float = math.nan   # same-direction hard movers among the other priced tickers
    market_wide: bool = False


def _log_returns(close: np.ndarray) -> np.ndarray:
    """Log return of each bar over the previous one; NaN unless both closes are finite and > 0."""
    out = np.full(len(close), np.nan)
    if len(close) < 2:
        return out
    prev, cur = close[:-1], close[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
    out[1:][ok] = np.log(cur[ok] / prev[ok])
    return out


def _explained_by_action(idx: pd.DatetimeIndex, actions: pd.DataFrame) -> np.ndarray:
    """True for bars within ACTION_WINDOW_SESSIONS bars of an action's ex-date bar."""
    explained = np.zeros(len(idx), dtype=bool)
    if len(idx) == 0 or actions is None or len(actions) == 0:
        return explained
    ex = actions.loc[actions["action_type"].isin(_EXPLAINING_ACTIONS)].index
    if len(ex) == 0:
        return explained
    # An ex-date without a stored bar goes to the first bar on or after it (as dividends_on_bars).
    pos = idx.searchsorted(pd.DatetimeIndex(ex).normalize(), side="left")
    for p in pos[pos < len(idx)]:
        explained[max(0, p - ACTION_WINDOW_SESSIONS):p + ACTION_WINDOW_SESSIONS + 1] = True
    return explained


def _stale_runs(close: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) bar positions of every run of >= STALE_RUN_MIN_SESSIONS equal valid closes."""
    runs: list[tuple[int, int]] = []
    n = len(close)
    valid = np.isfinite(close) & (close > 0)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and valid[j + 1]
               and abs(close[j + 1] - close[j]) <= STALE_RUN_REL_TOL * abs(close[j])):
            j += 1
        if j - i + 1 >= STALE_RUN_MIN_SESSIONS:
            runs.append((i, j))
        i = j + 1
    return runs


def detect_price_anomalies(
    bars: pd.DataFrame, actions: pd.DataFrame, ticker: str = "",
) -> list[PriceAnomaly]:
    """
    Anomalies in the daily close path of one asset, ordered by `start`. Pure; inputs are not modified.

    `bars` has a date index and `open/high/low/close/volume` columns (only `close`, plus `open`
    for the split cross-check, is read); `actions` is the `corporate_actions.load_actions`
    frame for the asset and may be empty. See the module docstring for the rules and for why
    this is offline inspection only.
    """
    if len(bars) == 0:
        return []
    idx = pd.DatetimeIndex(bars.index).normalize()
    close = bars["close"].to_numpy(dtype=float)
    ret = _log_returns(close)
    mag = np.abs(ret)
    n = len(close)

    with np.errstate(invalid="ignore"):
        candidate = mag >= SPIKE_MIN_ABS_LOG_RET           # NaN compares False
        reverts = np.zeros(n, dtype=bool)
        reverts[:-1] = (
            candidate[:-1]
            & (np.sign(ret[1:]) == -np.sign(ret[:-1]))
            & (mag[1:] >= SPIKE_REVERT_SHARE * mag[:-1])
        )
        spike_bar = reverts                                 # flagged at t, judged with t+1
        reverting_bar = np.zeros(n, dtype=bool)
        reverting_bar[1:] = spike_bar[:-1]
        extreme = (mag >= EXTREME_ABS_LOG_RET) & ~spike_bar & ~reverting_bar

    if extreme.any():
        extreme &= ~_explained_by_action(idx, actions)
        for issue in detect_unadjusted_splits(bars, actions, ticker=ticker):
            extreme[int(idx.searchsorted(pd.Timestamp(issue.ex_date), side="left"))] = False

    found: list[PriceAnomaly] = []
    for t in np.flatnonzero(spike_bar):
        found.append(PriceAnomaly(ticker, "spike_reversal", idx[t].date(), idx[t].date(), 1, float(ret[t])))
    for t in np.flatnonzero(extreme):
        found.append(PriceAnomaly(ticker, "extreme_move", idx[t].date(), idx[t].date(), 1, float(ret[t])))
    for first, last in _stale_runs(close):
        found.append(PriceAnomaly(ticker, "stale_run", idx[first].date(), idx[last].date(),
                                  last - first + 1, float(close[first])))
    found.sort(key=lambda a: (a.start, ANOMALY_KINDS.index(a.kind)))
    return found


def daily_move_counts(bars: pd.DataFrame) -> pd.DataFrame:
    """
    One asset's contribution to the cross-section, indexed by the normalised bar dates.

    Integer columns `priced` (valid close-to-close log return, the `_log_returns` rule), `up`
    (log return >= +MARKET_MOVE_ABS_LOG_RET) and `down` (<= -MARKET_MOVE_ABS_LOG_RET). A bar
    without a valid return contributes 0 to all three; no row is dropped. Sum the frames of
    all assets to get the `counts` argument of `tag_market_wide`. Pure.
    """
    idx = pd.DatetimeIndex(bars.index).normalize()
    ret = _log_returns(bars["close"].to_numpy(dtype=float))
    with np.errstate(invalid="ignore"):                    # NaN compares False
        up = ret >= MARKET_MOVE_ABS_LOG_RET
        down = ret <= -MARKET_MOVE_ABS_LOG_RET
    return pd.DataFrame(
        {"priced": np.isfinite(ret).astype(int), "up": up.astype(int), "down": down.astype(int)},
        index=idx,
    )


def tag_market_wide(
    anomalies: list[PriceAnomaly],
    counts: pd.DataFrame,
    own_moves: dict[tuple[str, date], tuple[int, int, int]] | None = None,
) -> list[PriceAnomaly]:
    """
    Copies of `anomalies` with `market_share` / `market_wide` set for the return kinds. Pure.

    `counts` is the per-date sum of `daily_move_counts` over all assets (columns `priced`, `up`,
    `down`). Self is removed with the ticker's own row, `own_moves[(ticker, start)] = (priced,
    up, down)`; when a key is missing the row is derived from the anomaly itself (priced, and a
    hard mover in the direction of `value`). `market_share` = (same-sign hard movers - self) /
    (priced - self), NaN when no other ticker is priced that day. `market_wide` needs the share
    at least MARKET_WIDE_MIN_SHARE and at least MARKET_WIDE_MIN_TICKERS other priced tickers.
    """
    out: list[PriceAnomaly] = []
    for a in anomalies:
        if a.kind not in _RETURN_KINDS:
            out.append(replace(a))
            continue
        key = pd.Timestamp(a.start).normalize()
        if key in counts.index:
            row = counts.loc[key]
            priced, same = int(row["priced"]), int(row["up" if a.value > 0 else "down"])
        else:
            priced, same = 0, 0
        own = (own_moves or {}).get((a.ticker, a.start))
        if own is None:
            own = (1, int(a.value > 0), int(a.value < 0))
        self_priced, self_same = int(own[0]), int(own[1] if a.value > 0 else own[2])
        others, movers = priced - self_priced, same - self_same
        if others <= 0:
            out.append(replace(a, market_share=math.nan, market_wide=False))
            continue
        share = movers / others
        out.append(replace(a, market_share=share,
                           market_wide=share >= MARKET_WIDE_MIN_SHARE and others >= MARKET_WIDE_MIN_TICKERS))
    return out


def summarise_anomalies(anomalies: list[PriceAnomaly]) -> pd.DataFrame:
    """
    Counts by calendar year of `start` (rows) and kind (columns, all of ANOMALY_KINDS).

    The last two rows are `total` (count per kind) and `tickers` (distinct tickers per kind);
    the year rows are ints in ascending order. With no anomalies only those two rows (zeros) remain.
    """
    years = sorted({a.start.year for a in anomalies})
    table = pd.DataFrame(0, index=years, columns=list(ANOMALY_KINDS), dtype=int)
    for a in anomalies:
        table.loc[a.start.year, a.kind] += 1
    total = table.sum(axis=0).astype(int)
    tickers = pd.Series(
        {k: len({a.ticker for a in anomalies if a.kind == k}) for k in ANOMALY_KINDS}, dtype=int,
    )
    out = pd.concat([table, total.to_frame(SUMMARY_TOTAL_ROW).T, tickers.to_frame(SUMMARY_TICKERS_ROW).T])
    out.index = pd.Index(list(years) + [SUMMARY_TOTAL_ROW, SUMMARY_TICKERS_ROW], dtype=object, name="year")
    return out.astype(int)


def largest_anomalies(anomalies: list[PriceAnomaly], n: int = 30) -> list[PriceAnomaly]:
    """
    The `n` most notable cases: return kinds by |log return| descending, then stale runs by
    `sessions` descending. (`value` of a stale run is a price level, not comparable with a return.)
    """
    returns = sorted((a for a in anomalies if a.kind in _RETURN_KINDS), key=lambda a: -abs(a.value))
    stale = sorted((a for a in anomalies if a.kind not in _RETURN_KINDS), key=lambda a: (-a.sessions, a.start))
    return (returns + stale)[:n]


# ──────────────────────────────────────────────────────────
# Database loader
# ──────────────────────────────────────────────────────────

def audit_price_anomalies(
    session: Session, *, tickers: Collection[str] | None = None,
) -> list[PriceAnomaly]:
    """
    `detect_price_anomalies` over the daily bars of every asset that has any, active or not (past
    index members are in the point-in-time universe). One asset's bars in memory at a time. Read-only.

    The same pass sums `daily_move_counts` over the assets and keeps each asset's own row for its
    anomaly dates; `tag_market_wide` then runs once at the end.
    """
    stmt = (
        select(Asset.id, Asset.ticker)
        .where(Asset.id.in_(select(PriceDaily.asset_id)))
        .order_by(Asset.ticker)
    )
    if tickers is not None:
        stmt = stmt.where(Asset.ticker.in_(list(tickers)))
    found: list[PriceAnomaly] = []
    counts: pd.DataFrame | None = None
    own_moves: dict[tuple[str, date], tuple[int, int, int]] = {}
    for asset_id, ticker in session.execute(stmt).all():
        bars = pd.read_sql(
            select(PriceDaily.date, PriceDaily.open, PriceDaily.high, PriceDaily.low,
                   PriceDaily.close, PriceDaily.volume)
            .where(PriceDaily.asset_id == asset_id).order_by(PriceDaily.date),
            session.connection(), index_col="date", parse_dates=["date"],
        )
        mine = detect_price_anomalies(bars, load_actions(session, ticker), ticker=ticker)
        found.extend(mine)
        own = daily_move_counts(bars)
        for a in mine:
            if a.kind in _RETURN_KINDS:
                own_moves[(ticker, a.start)] = tuple(int(v) for v in own.loc[pd.Timestamp(a.start)])
        counts = own if counts is None else counts.add(own, fill_value=0).astype(int)
    if counts is None:
        return found
    return tag_market_wide(found, counts, own_moves)
