"""
Price coverage of index membership stints (`python main.py coverage`). Read-only.

After `ingest --members --full` every past and present member has to be checked: did the
symbol return prices, and do they cover the stint the ticker was in the index? A symbol whose
bars stop before its stint ends is probably renamed, merged or wrong. This module answers that
per stint and per calendar year from the stored daily bar dates only (no OHLCV, no network,
no writes).

The market calendar has no source of its own: it is the union of every date with at least one
stored daily bar of any stock (`session_calendar`). A stint's sessions are the calendar dates
inside `[effective_from, effective_to]`; an open-ended stint counts up to `as_of` (default: the
last calendar date), and `as_of` also caps closed stints. Per-date counts use only the membership
and bars on that date, so truncating the bars after d leaves earlier years unchanged.

Pure functions on DataFrames come first; the DB loaders are at the bottom of the file.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.data.index_membership import membership_matrix
from investbot.data.models import Asset, IndexMembership, PriceDaily

# A stint is "holes" below this share of priced sessions.
COVERAGE_OK: float = 0.98
# More missing sessions than this at either edge of a stint make it late_start / early_end.
COVERAGE_EDGE_SESSIONS: int = 5

# Priority order of `status`: the first one that holds wins.
STATUS_ORDER = ("no_prices", "outside_stint", "early_end", "late_start", "holes", "ok")

STINT_COLUMNS = [
    "ticker", "effective_from", "effective_to", "sessions", "priced_sessions", "coverage",
    "first_bar", "last_bar", "missing_at_start", "missing_at_end", "late_start", "early_end",
    "status",
]
YEAR_COLUMNS = [
    "year", "sessions", "members_median", "members_min", "priced_median", "priced_min",
    "priced_share_mean",
]


def session_calendar(bar_dates: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted union of every date with at least one stored daily bar of any stock."""
    return pd.DatetimeIndex(pd.to_datetime(bar_dates["date"]).unique()).sort_values()


def _bars_by_ticker(bar_dates: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    dates = pd.to_datetime(bar_dates["date"])
    return {t: pd.DatetimeIndex(g.unique()).sort_values()
            for t, g in dates.groupby(bar_dates["ticker"], sort=False)}


def _horizon(calendar: pd.DatetimeIndex, as_of: date | None) -> pd.Timestamp | None:
    if as_of is not None:
        return pd.Timestamp(as_of)
    return calendar[-1] if len(calendar) else None


def _stint_bounds(stints: pd.DataFrame, horizon: pd.Timestamp | None
                  ) -> tuple[pd.Series, pd.Series]:
    """Start and end of each stint as Timestamps; an open end (or one past the horizon) is the horizon."""
    start = pd.to_datetime(stints["effective_from"])
    end = pd.to_datetime(stints["effective_to"])
    if horizon is not None:
        end = end.where(end.notna() & (end <= horizon), horizon)
    return start, end


def _status(first_bar_seen: bool, priced: int, sessions: int, after_horizon: bool,
            early_end: bool, late_start: bool, coverage: float) -> str:
    if sessions == 0 and after_horizon:
        return "ok"
    if not first_bar_seen:
        return "no_prices"
    if priced == 0:
        return "outside_stint"
    if early_end:
        return "early_end"
    if late_start:
        return "late_start"
    if coverage < COVERAGE_OK:
        return "holes"
    return "ok"


def stint_coverage(stints: pd.DataFrame, bar_dates: pd.DataFrame, calendar: pd.DatetimeIndex,
                   *, as_of: date | None = None) -> pd.DataFrame:
    """One row per stint (`STINT_COLUMNS`): how many of its sessions have a bar, and where the gaps are.

    `stints` has `ticker, effective_from, effective_to` (NULL/NaT = still a member, counted up to
    `as_of`, default the last calendar date); `bar_dates` has `ticker, date`. `first_bar` /
    `last_bar` are the ticker's overall first and last stored bar. A stint with no priced session
    has `missing_at_start == missing_at_end == sessions`. A stint that starts after the horizon has
    no sessions, NaN coverage and status "ok"; it is kept.
    """
    horizon = _horizon(calendar, as_of)
    start, end = _stint_bounds(stints, horizon)
    bars = _bars_by_ticker(bar_dates)
    cal = calendar.values

    cols: dict[str, list] = {c: [] for c in STINT_COLUMNS}
    for ticker, eff_from, eff_to, s, e in zip(stints["ticker"], stints["effective_from"],
                                              stints["effective_to"], start, end):
        has_bars = ticker in bars
        own = bars[ticker] if has_bars else pd.DatetimeIndex([])
        if pd.isna(e):  # no calendar and no as_of: nothing to count
            window = cal[:0]
        else:
            window = cal[np.searchsorted(cal, s.to_datetime64(), side="left"):
                         np.searchsorted(cal, e.to_datetime64(), side="right")]
        sessions = len(window)
        hit = np.flatnonzero(np.isin(window, own.values)) if sessions else np.array([], dtype=int)
        priced = len(hit)
        if priced:
            miss_start, miss_end = int(hit[0]), int(sessions - 1 - hit[-1])
        else:
            miss_start = miss_end = sessions
        coverage = priced / sessions if sessions else float("nan")
        late = miss_start > COVERAGE_EDGE_SESSIONS
        early = miss_end > COVERAGE_EDGE_SESSIONS
        after = horizon is not None and s > horizon

        cols["ticker"].append(ticker)
        cols["effective_from"].append(eff_from)
        cols["effective_to"].append(eff_to)
        cols["sessions"].append(sessions)
        cols["priced_sessions"].append(priced)
        cols["coverage"].append(coverage)
        cols["first_bar"].append(own[0] if has_bars else pd.NaT)
        cols["last_bar"].append(own[-1] if has_bars else pd.NaT)
        cols["missing_at_start"].append(miss_start)
        cols["missing_at_end"].append(miss_end)
        cols["late_start"].append(late)
        cols["early_end"].append(early)
        cols["status"].append(_status(has_bars, priced, sessions, after, early, late, coverage))

    out = pd.DataFrame(cols, columns=STINT_COLUMNS)
    out["first_bar"] = pd.to_datetime(out["first_bar"])
    out["last_bar"] = pd.to_datetime(out["last_bar"])
    out["coverage"] = out["coverage"].astype(float)
    return out


def coverage_by_year(stints: pd.DataFrame, bar_dates: pd.DataFrame, calendar: pd.DatetimeIndex,
                     *, as_of: date | None = None) -> pd.DataFrame:
    """One row per calendar year (`YEAR_COLUMNS`) of the per-session member / priced-member counts.

    Members on date d are the stints that contain d (inclusive bounds, `membership_matrix`);
    priced members are those of them with a bar on d. `priced_share_mean` is the mean over the
    year's sessions of priced / members (sessions with no member are skipped by the mean).
    """
    horizon = _horizon(calendar, as_of)
    dates = calendar if horizon is None else calendar[calendar <= horizon]
    start, end = _stint_bounds(stints, horizon)
    closed = pd.DataFrame({"ticker": stints["ticker"].to_numpy(),
                           "effective_from": start.to_numpy(), "effective_to": end.to_numpy()})
    tickers = list(dict.fromkeys(stints["ticker"]))

    member = membership_matrix(closed, dates, tickers)
    bars = _bars_by_ticker(bar_dates)
    priced = pd.DataFrame(
        {t: dates.isin(bars[t]) if t in bars else np.zeros(len(dates), dtype=bool)
         for t in tickers},
        index=dates, columns=tickers)
    priced &= member

    n_members = member.sum(axis=1)
    n_priced = priced.sum(axis=1)
    share = (n_priced / n_members.where(n_members > 0)).astype(float)
    year = dates.year
    g_members = n_members.groupby(year)
    g_priced = n_priced.groupby(year)

    cols = {
        "year": g_members.size().index.to_numpy(),
        "sessions": g_members.size().to_numpy(),
        "members_median": g_members.median().to_numpy(),
        "members_min": g_members.min().to_numpy(),
        "priced_median": g_priced.median().to_numpy(),
        "priced_min": g_priced.min().to_numpy(),
        "priced_share_mean": share.groupby(year).mean().to_numpy(),
    }
    return pd.DataFrame(cols, columns=YEAR_COLUMNS)


def _day(ts) -> str:
    return "-" if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def format_coverage(stints_df: pd.DataFrame, years_df: pd.DataFrame, *, max_rows: int = 60) -> list[str]:
    """Printable lines: counts by status, every non-ok stint (cut at `max_rows`), the per-year table."""
    lines = [f"Price coverage of {len(stints_df)} membership stints "
             f"(ok = at least {COVERAGE_OK:.0%} of sessions priced, edges within "
             f"{COVERAGE_EDGE_SESSIONS} sessions):"]
    counts = stints_df["status"].value_counts() if len(stints_df) else pd.Series(dtype=int)
    lines += [f"  {s:<14s} {int(counts.get(s, 0)):>4d}" for s in STATUS_ORDER]

    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    bad = stints_df[stints_df["status"] != "ok"]
    bad = bad.assign(_rank=bad["status"].map(rank)).sort_values(
        ["_rank", "effective_from", "ticker"], kind="stable")
    lines.append("")
    if bad.empty:
        lines.append("No stint with a coverage problem.")
    else:
        lines.append(f"Stints with a coverage problem ({len(bad)}):")
        lines.append(f"  {'ticker':<18s}{'stint':<25s}{'coverage':>9s}  {'first bar':<11s}"
                     f"{'last bar':<11s}status")
        for r in bad.head(max_rows).itertuples(index=False):
            stint = f"{_day(r.effective_from)}..{_day(r.effective_to) if pd.notna(r.effective_to) else 'open'}"
            cov = "-" if pd.isna(r.coverage) else f"{100 * r.coverage:.1f}%"
            lines.append(f"  {r.ticker:<18s}{stint:<25s}{cov:>9s}  {_day(r.first_bar):<11s}"
                         f"{_day(r.last_bar):<11s}{r.status}")
        if len(bad) > max_rows:
            lines.append(f"  … and {len(bad) - max_rows} more")

    lines.append("")
    lines.append("Members with a bar, per calendar year:")
    lines.append(f"  {'year':<6s}{'sessions':>9s}{'members med':>13s}{'min':>6s}"
                 f"{'priced med':>12s}{'min':>6s}{'priced share':>14s}")
    for r in years_df.itertuples(index=False):
        share = "-" if pd.isna(r.priced_share_mean) else f"{100 * r.priced_share_mean:.1f}%"
        lines.append(f"  {int(r.year):<6d}{int(r.sessions):>9d}{r.members_median:>13.1f}"
                     f"{int(r.members_min):>6d}{r.priced_median:>12.1f}{int(r.priced_min):>6d}"
                     f"{share:>14s}")
    return lines


# ──────────────────────────────────────────────────────────
# DB side (read-only)
# ──────────────────────────────────────────────────────────

def load_stints(session: Session, index_code: str = "NIFTY50") -> pd.DataFrame:
    """`ticker, effective_from, effective_to` of every stint of `index_code` (NULL end = NaT)."""
    rows = session.execute(
        select(Asset.ticker, IndexMembership.effective_from, IndexMembership.effective_to)
        .join(Asset, Asset.id == IndexMembership.asset_id)
        .where(IndexMembership.index_code == index_code)
        .order_by(IndexMembership.effective_from, Asset.ticker)
    ).all()
    df = pd.DataFrame(rows, columns=["ticker", "effective_from", "effective_to"])
    df["effective_from"] = pd.to_datetime(df["effective_from"])
    df["effective_to"] = pd.to_datetime(df["effective_to"])
    return df


def load_bar_dates(session: Session) -> pd.DataFrame:
    """`ticker, date` of every stored daily stock bar. Dates only: never an OHLCV column."""
    rows = session.execute(
        select(Asset.ticker, PriceDaily.date)
        .join(PriceDaily, PriceDaily.asset_id == Asset.id)
        .where(Asset.asset_type == "stock")
    ).all()
    df = pd.DataFrame(rows, columns=["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df
