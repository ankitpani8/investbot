"""
Read-only check of `NIFTY50_TICKERS` against NSE's current constituent CSV.

NSE reconstitutes NIFTY50 in March and September, and mergers and demergers change it in
between. Nothing else notices when the repo's list stops matching, so `python main.py
constituents` and the weekly `constituents_check` scheduler job run this.

The check only reads: it never edits `NIFTY50_TICKERS`, `nifty50_changes.csv` or the DB. A
human applies the change (append rows to `nifty50_changes.csv`, update `NIFTY50_TICKERS`,
bump `CURRENT_AS_OF` in `index_membership.py`).

One HTTP GET per call, no retries, no other NSE endpoints. Source used on 2026-10-18; the
niftyindices.com URL timed out. NSE rejects bare clients, hence the browser-like User-Agent.
During a demerger the index briefly holds 51 names, so the accepted count is 45-55.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from investbot.data.index_membership import current_tickers, members_as_of, nifty50_intervals

logger = logging.getLogger(__name__)

NIFTY50_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")

# Plausible size of the list; anything else is an error page or a truncated file, not a change.
MIN_CONSTITUENTS = 45
MAX_CONSTITUENTS = 55

# NSE symbol -> our ticker, for the day the two differ (e.g. NSE renames a symbol but we keep
# the Yahoo symbol, or Yahoo uses a different suffix). Empty today: every symbol is f"{symbol}.NS".
# Add an entry only after checking by hand that both names are the same company; the value is
# the full ticker as it appears in NIFTY50_TICKERS, e.g. {"NEWSYM": "OLDSYM.NS"}.
SYMBOL_OVERRIDES: dict[str, str] = {}


class ConstituentFetchError(Exception):
    """The CSV could not be downloaded (network, HTTP status, timeout)."""


class ConstituentParseError(Exception):
    """The body is not a plausible NIFTY50 list (HTML page, empty, no Symbol column, odd count)."""


@dataclass(frozen=True)
class ConstituentDiff:
    added: list[str]               # in NSE's list, not in ours
    removed: list[str]             # in ours, not in NSE's list
    matches: bool                  # no added and no removed
    members_disagree: list[str]    # NIFTY50_TICKERS vs the index_membership chain today; should be []


def _urllib_fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8-sig", errors="replace")


def fetch_index_csv(url: str = NIFTY50_LIST_URL, *,
                    fetcher: Callable[[str], str] | None = None, timeout: float = 20) -> str:
    """Download the CSV text with one request. `fetcher(url) -> text` is injected by tests.

    Any failure raises ConstituentFetchError; the caller decides how loudly to log it.
    """
    try:
        if fetcher is not None:
            return fetcher(url)
        return _urllib_fetch(url, timeout)
    except Exception as e:
        raise ConstituentFetchError(f"could not fetch {url}: {type(e).__name__}: {e}") from e


def parse_constituents(csv_text: str) -> list[str]:
    """yfinance tickers (`SYMBOL.NS`) from the `Symbol` column, in file order.

    Raises ConstituentParseError for a missing `Symbol` column (an HTML error page lands
    here), an empty file, duplicate symbols, or a count outside 45-55.
    """
    text = csv_text.lstrip("﻿")
    if not text.strip():
        raise ConstituentParseError("empty body")
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip(): f for f in (reader.fieldnames or [])}
    if "Symbol" not in fields:
        raise ConstituentParseError(
            f"no 'Symbol' column (columns: {list(fields)[:6]}); not the NSE constituent CSV")
    col = fields["Symbol"]

    symbols = []
    for row in reader:
        sym = (row.get(col) or "").strip()
        if sym:
            symbols.append(sym)
    if not MIN_CONSTITUENTS <= len(symbols) <= MAX_CONSTITUENTS:
        raise ConstituentParseError(
            f"{len(symbols)} symbols, expected {MIN_CONSTITUENTS}-{MAX_CONSTITUENTS}")
    if len(set(symbols)) != len(symbols):
        raise ConstituentParseError("duplicate symbols in the CSV")
    return [SYMBOL_OVERRIDES.get(s, f"{s}.NS") for s in symbols]


def compare_constituents(nse: list[str], ours: list[str],
                         as_of_members: set[str] | None = None) -> ConstituentDiff:
    """Set differences, sorted. `as_of_members` is the index_membership chain today (optional)."""
    added = sorted(set(nse) - set(ours))
    removed = sorted(set(ours) - set(nse))
    disagree = sorted(set(ours) ^ as_of_members) if as_of_members is not None else []
    return ConstituentDiff(added=added, removed=removed,
                           matches=not added and not removed, members_disagree=disagree)


def check_nifty50(*, fetcher: Callable[[str], str] | None = None,
                  today: date | None = None) -> ConstituentDiff:
    """Fetch NSE's list and compare it with the repo. No DB; raises ConstituentFetchError / ParseError."""
    nse = parse_constituents(fetch_index_csv(fetcher=fetcher))
    ours = current_tickers()
    chain = members_as_of(nifty50_intervals(), today or date.today())
    return compare_constituents(nse, ours, chain)
