"""
Read-only check that `index_membership` matches the repo's change list.

`sync_index_membership` derives the table from `config/index_history/*.csv` and
`NIFTY50_TICKERS`; a DB that was synced before the files changed (or never) is
stale, and nothing else says so. This module compares the two and reports the
difference. It only reads: no writes, no network, no prices fetched.

A stint matches on `(ticker, effective_from, effective_to)`, an open-ended
`effective_to` being NULL. Missing prices are reported but do not make the table
out of sync: that is an ingest question.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from investbot.data.index_membership import INDEXES, nifty50_intervals
from investbot.data.models import Asset, Exchange, IndexMembership, PriceDaily

FIX = "python main.py setup"

Stint = tuple[str, object, object]


@dataclass(frozen=True)
class MembershipSync:
    index_code: str
    in_sync: bool
    db_rows: int
    expected_rows: int
    missing: tuple[str, ...]                  # expected stints absent from the DB
    extra: tuple[str, ...]                    # DB stints that are not expected
    members_without_asset: tuple[str, ...]    # change-list tickers with no `assets` row
    members_without_prices: tuple[str, ...]   # member tickers with an asset row but no daily prices


def expected_intervals(index_code: str = "NIFTY50") -> pd.DataFrame:
    """Stints the repo files imply for `index_code` (the same frame `sync_index_membership` writes)."""
    if index_code not in INDEXES:
        raise ValueError(f"no change list for index {index_code!r}")
    return nifty50_intervals()


def _stint_key(ticker: str, start, end) -> Stint:
    return (ticker, start, end if pd.notna(end) else None)


def _label(key: Stint) -> str:
    ticker, start, end = key
    return f"{ticker} {start}..{end if end is not None else 'open'}"


def check_membership_sync(session: Session, index_code: str = "NIFTY50") -> MembershipSync:
    """Compare `index_membership` for `index_code` with the repo's change list. Read-only."""
    expected = expected_intervals(index_code)
    exchange_code = INDEXES[index_code][0]

    want = [_stint_key(r.ticker, r.effective_from, r.effective_to)
            for r in expected.itertuples(index=False)]
    have_rows = session.execute(
        select(Asset.ticker, IndexMembership.effective_from, IndexMembership.effective_to)
        .join(Asset, Asset.id == IndexMembership.asset_id)
        .where(IndexMembership.index_code == index_code)
    ).all()
    have = [_stint_key(t, s, e) for t, s, e in have_rows]

    missing = sorted(set(want) - set(have), key=lambda k: (k[1], k[0]))
    extra = sorted(set(have) - set(want), key=lambda k: (k[1], k[0]))

    tickers = sorted(set(expected["ticker"]))
    assets = dict(session.execute(
        select(Asset.ticker, Asset.id)
        .join(Exchange, Exchange.id == Asset.exchange_id)
        .where(Exchange.code == exchange_code, Asset.ticker.in_(tickers))
    ).all())
    without_asset = tuple(t for t in tickers if t not in assets)
    priced = set(session.execute(
        select(Asset.ticker).where(
            Asset.id.in_(list(assets.values())),
            exists().where(PriceDaily.asset_id == Asset.id))
    ).scalars())
    without_prices = tuple(t for t in tickers if t in assets and t not in priced)

    return MembershipSync(
        index_code=index_code,
        in_sync=not missing and not extra and not without_asset,
        db_rows=len(have_rows),
        expected_rows=len(want),
        missing=tuple(_label(k) for k in missing),
        extra=tuple(_label(k) for k in extra),
        members_without_asset=without_asset,
        members_without_prices=without_prices,
    )


def _items(title: str, items: tuple[str, ...], max_items: int) -> list[str]:
    if not items:
        return []
    lines = [f"  {title} ({len(items)}):"]
    lines += [f"    {i}" for i in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"    … and {len(items) - max_items} more")
    return lines


def format_sync(s: MembershipSync, max_items: int = 10) -> list[str]:
    """Human-readable report lines; long lists are cut at `max_items` with "… and N more"."""
    if s.in_sync:
        head = f"{s.index_code} membership is in sync with the change list ({s.db_rows} stints)."
    else:
        head = (f"{s.index_code} membership is OUT OF SYNC with the change list: "
                f"{s.db_rows} rows in the DB, {s.expected_rows} expected.")
    lines = [head]
    lines += _items("Expected stints missing from the DB", s.missing, max_items)
    lines += _items("DB stints not in the change list", s.extra, max_items)
    lines += _items("Change-list tickers with no assets row", s.members_without_asset, max_items)
    lines += _items("Members without daily prices", s.members_without_prices, max_items)
    if not s.in_sync:
        lines.append(f"  Fix: run `{FIX}` (local seed, no network).")
    return lines
