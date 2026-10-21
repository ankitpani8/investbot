"""
Point-in-time index membership (NIFTY50 first, the table is index-agnostic).

Source of truth is `config/index_history/nifty50_changes.csv`: one row per
change (`add` / `remove`), `effective_date` = first session the change applies
to, `source` = the NSE Indices press release it was read from. Only changes
read in an official source are in the file.

Intervals are rebuilt by walking *backwards* from today's list
(`NIFTY50_TICKERS`, as of `CURRENT_AS_OF`) and reversing each change:
  - reversing an `add` closes the stint's start (effective_from = that date);
  - reversing a `remove` re-opens a stint that ended the day before.
`effective_to` is the calendar day before a removal (queries are `from <= d <= to`,
so it behaves as "last session" on every trading day).

Completeness: `COMPLETE_FROM` is the first date from which the whole change
chain is sourced. Members already in the index at the start of the chain have an
unsourced start (`effective_from = HISTORY_START`, `complete=False`,
`verified=False`). Dates before `COMPLETE_FROM` are returned as-is from those
intervals and must not be treated as known; check `is_complete`.

Temporary members: when a NIFTY50 constituent demerges, the spun-off company is
added at zero price under a dummy symbol and dropped a few sessions after it
lists (JIOFIN 2023, ITCHOTELS 2025, TMCV 2025, KWIL 2025-26). The index then
holds 51 names; those windows are marked `temporary` in the CSV note column.
The same holds for 2016-04-01 .. 2017-09-28, when the Tata Motors DVR share class
(TATAMTRDVR) was an additional security: the release says the index "shall have
51 securities". It is a separate ticker and is marked `temporary` for the same reason.

Source chain: every NIFTY 50 change is read from the NSE Indices press releases (niftyindices.com
`Press_Release/*.pdf`; before the 2015 rebrand the index is "CNX Nifty" or "S&P CNX Nifty"). The chain
runs from the 1998-09-02 release (effective 1998-10-07, the oldest release the archive lists) to the
current list and is gap-free: the reverse walk from today's list never contradicts a change and gives
50 names (51 in the `temporary` windows) on every day from 1998-10-07. The 1998-10-07 change is the
first one in the archive, so the 50 names before it are the reverse-walk result, not sourced.
Releases before about 2012 give company names only (no symbol column); where no release read gives
the symbol, the CSV note says so and the symbol is the NSE trading symbol, assumed. A company that
was renamed keeps one ticker where the reverse walk requires it (the note says so).

Symbols are the press-release symbol plus `.NS`; one company keeps one ticker
(ZOMATO is ETERNAL.NS; TATAMOTORS.NS is TMPV.NS, which never left the index).

Pure functions take frames and return frames or sets; only `sync_index_membership`
and `membership_summary` touch the DB. No price rows are fetched here.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from investbot.config.tickers import NIFTY50_TICKERS
from investbot.data.models import Asset, Exchange, IndexMembership, PriceDaily

logger = logging.getLogger(__name__)

CHANGES_DIR = Path(__file__).resolve().parents[1] / "config" / "index_history"
NIFTY50_CHANGES_CSV = CHANGES_DIR / "nifty50_changes.csv"

# NIFTY50_TICKERS is the list as of this date; bump it with the list and append the changes.
CURRENT_AS_OF = date(2026, 10, 18)
# Start of the studied window (the first read release, 1998-09-02, rounded down to the 1st of its
# month); the start of stints that began earlier is unknown.
HISTORY_START = date(1998, 9, 1)
# Earliest date from which every later change is sourced (the first sourced change).
COMPLETE_FROM = date(1998, 10, 7)

CURRENT_LIST_SOURCE = "investbot/config/tickers.py NIFTY50_TICKERS (NSE ind_nifty50list.csv, 2026-10-18)"

INTERVAL_COLUMNS = ["ticker", "effective_from", "effective_to", "source", "complete"]

# index_code -> (exchange code, changes CSV)
INDEXES = {"NIFTY50": ("NSE", NIFTY50_CHANGES_CSV)}


def load_changes(path: Path | str = NIFTY50_CHANGES_CSV) -> pd.DataFrame:
    """Read a change list; `effective_date` becomes `datetime.date`."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["effective_date"] = [date.fromisoformat(d) for d in df["effective_date"]]
    return df


def current_tickers() -> list[str]:
    return [t for t, *_ in NIFTY50_TICKERS]


def reconstruct_intervals(
    current: list[str],
    changes: pd.DataFrame,
    as_of: date,
    *,
    history_start: date = HISTORY_START,
) -> pd.DataFrame:
    """Membership stints (`ticker, effective_from, effective_to, source, complete`).

    `current` is the member list on `as_of`; changes dated after `as_of` are ignored.
    `effective_to` is None for a current member. `complete` is True when the start
    of the stint is a sourced `add`; members already in at the start of the chain
    start at `history_start` with `complete=False`. Raises ValueError when a change
    contradicts the walk (an add of a name that is not a member afterwards, a
    removal of a name that is).
    """
    state = set(current)
    if len(state) != len(current):
        raise ValueError("duplicate tickers in the current list")
    ends: dict[str, date | None] = {t: None for t in state}
    end_source: dict[str, str] = {t: CURRENT_LIST_SOURCE for t in state}
    rows: list[dict] = []

    ch = changes[changes["effective_date"] <= as_of]
    has_source = "source" in ch.columns
    for d, group in sorted(ch.groupby("effective_date"), key=lambda kv: kv[0], reverse=True):
        for _, c in group.iterrows():
            t, action = c["ticker"], c["action"]
            src = c["source"] if has_source else ""
            if action == "add":
                if t not in state:
                    raise ValueError(f"{d}: add {t} but it is not a member from that date")
                state.remove(t)
                rows.append({"ticker": t, "effective_from": d, "effective_to": ends.pop(t),
                             "source": src, "complete": True})
                end_source.pop(t, None)
            elif action == "remove":
                if t in state:
                    raise ValueError(f"{d}: remove {t} but it is still a member from that date")
                state.add(t)
                ends[t] = d - timedelta(days=1)
                end_source[t] = src
            else:
                raise ValueError(f"{d}: unknown action {action!r} for {t}")

    for t in sorted(state):
        rows.append({"ticker": t, "effective_from": history_start, "effective_to": ends[t],
                     "source": end_source[t], "complete": False})

    out = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    return out.sort_values(["effective_from", "ticker"]).reset_index(drop=True)


def _bounds(intervals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    start = pd.to_datetime(intervals["effective_from"])
    end = pd.to_datetime(intervals["effective_to"])  # None -> NaT
    return start, end


def members_as_of(intervals: pd.DataFrame, d: date) -> set[str]:
    """Tickers that are members on `d`; uses only stints that started on or before `d`."""
    start, end = _bounds(intervals)
    ts = pd.Timestamp(d)
    mask = (start <= ts) & (end.isna() | (end >= ts))
    return set(intervals.loc[mask, "ticker"])


def membership_matrix(intervals: pd.DataFrame, dates, tickers: list[str]) -> pd.DataFrame:
    """Boolean frame (rows = `dates`, columns = `tickers`): member on that date."""
    idx = pd.DatetimeIndex(dates)
    start, end = _bounds(intervals)
    cols: dict[str, pd.Series] = {}
    for t in tickers:
        m = pd.Series(False, index=idx)
        for s, e in zip(start[intervals["ticker"] == t], end[intervals["ticker"] == t]):
            m |= (idx >= s) & (idx <= e if pd.notna(e) else True)
        cols[t] = m
    return pd.DataFrame(cols, index=idx, columns=list(tickers))


def is_complete(d: date, complete_from: date = COMPLETE_FROM) -> bool:
    """False before the change chain is fully sourced: membership there is not known."""
    return d >= complete_from


def complete_mask(dates, complete_from: date = COMPLETE_FROM) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    return pd.Series(idx >= pd.Timestamp(complete_from), index=idx)


def nifty50_intervals() -> pd.DataFrame:
    """Intervals for the repo's NIFTY50 list and change file."""
    return reconstruct_intervals(current_tickers(), load_changes(NIFTY50_CHANGES_CSV), CURRENT_AS_OF)


# ──────────────────────────────────────────────────────────
# DB side
# ──────────────────────────────────────────────────────────

def sync_index_membership(session: Session, index_code: str = "NIFTY50") -> dict[str, int]:
    """Upsert the reconstructed intervals into `index_membership` (flush, no commit).

    Stale rows of this index (a stint that no longer exists) are deleted: the table
    is derived from the repo files. Members missing from `assets` are created with
    is_active=False and never activated; existing is_active flags are untouched.
    No prices are fetched. Returns counts of inserted / updated / deleted rows and
    created assets.
    """
    if index_code not in INDEXES:
        raise ValueError(f"no change list for index {index_code!r}")
    exchange_code, csv_path = INDEXES[index_code]
    exchange = session.query(Exchange).filter_by(code=exchange_code).one()

    changes = load_changes(csv_path)
    intervals = reconstruct_intervals(current_tickers(), changes, CURRENT_AS_OF)

    names = {t: n for t, n, _sector, _industry in NIFTY50_TICKERS}
    names.update(dict(zip(changes["ticker"], changes["name"])))  # latest row per ticker wins

    stats = {"inserted": 0, "updated": 0, "deleted": 0, "assets_created": 0}
    asset_ids: dict[str, int] = {}
    for t in intervals["ticker"].unique():
        asset = session.query(Asset).filter_by(exchange_id=exchange.id, ticker=t).first()
        if asset is None:
            asset = Asset(exchange_id=exchange.id, ticker=t, name=names.get(t, t), asset_type="stock",
                          sector=None, industry=None, is_active=False)
            session.add(asset)
            session.flush()
            stats["assets_created"] += 1
        asset_ids[t] = asset.id

    existing = {
        (r.asset_id, r.effective_from): r
        for r in session.query(IndexMembership).filter_by(index_code=index_code)
    }
    wanted: set[tuple[int, date]] = set()
    for row in intervals.itertuples(index=False):
        key = (asset_ids[row.ticker], row.effective_from)
        wanted.add(key)
        to = row.effective_to if pd.notna(row.effective_to) else None
        rec = existing.get(key)
        if rec is None:
            session.add(IndexMembership(
                index_code=index_code, asset_id=key[0], effective_from=row.effective_from,
                effective_to=to, source=row.source, verified=bool(row.complete)))
            stats["inserted"] += 1
        elif (rec.effective_to, rec.source, rec.verified) != (to, row.source, bool(row.complete)):
            rec.effective_to, rec.source, rec.verified = to, row.source, bool(row.complete)
            stats["updated"] += 1
    for key, rec in existing.items():
        if key not in wanted:
            session.delete(rec)
            stats["deleted"] += 1
    session.flush()
    logger.info("Index membership %s: %s", index_code, stats)
    return stats


def membership_summary(session: Session) -> dict[str, int]:
    """Row counts for `main.py status`: stints, distinct members, members without daily price rows."""
    n_rows = session.query(IndexMembership).count()
    n_members = session.query(func.count(func.distinct(IndexMembership.asset_id))).scalar() or 0
    no_prices = session.execute(
        select(func.count(func.distinct(IndexMembership.asset_id))).where(
            ~exists().where(PriceDaily.asset_id == IndexMembership.asset_id))
    ).scalar() or 0
    return {"rows": n_rows, "members": n_members, "members_without_prices": no_prices}
