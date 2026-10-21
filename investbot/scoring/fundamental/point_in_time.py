"""
Point-in-time availability of financial statements.

A filing's numbers may only be used from the day they were public. yfinance gives
period-end dates and no publication date, so when the real filing date is unknown
the conservative fallback is the latest date SEBI LODR Reg. 33 allows:

    quarterly results   period end + 45 days
    annual results      period end + 60 days
    Q4 (the quarter ending in a fiscal-year-end month) is published together with
    the audited annual results, so it gets the 60-day annual lag too.

Fiscal-year-end months are per company: ``fiscal_year_end_months`` takes the months of
the company's annual period ends and always adds March (``FISCAL_YEAR_END_MONTH``).
The union with March is the conservative choice: a quarter ending in March never gets
a shorter lag than under the old March-only rule, a December-FY company's December
quarter gets 60 days instead of 45, and a company that changed its fiscal year gets
60 days on both months. Every date is the same as or later than before, never earlier.

Pure module: no DB I/O. The ingestion layer stores ``available_from`` /
``available_from_source`` on every ``financials`` row; consumers filter through
``financials_as_of``.
"""
from collections.abc import Collection, Iterable
from datetime import date, datetime, timedelta

import pandas as pd

QUARTERLY_LAG_DAYS = 45       # SEBI LODR Reg. 33(3)(a): quarterly results within 45 days
ANNUAL_LAG_DAYS = 60          # SEBI LODR Reg. 33(3)(d): audited annual results within 60 days
FISCAL_YEAR_END_MONTH = 3     # Indian fiscal year ends 31 March

SOURCE_PUBLISHED = "published"
SOURCE_LODR_LAG = "lodr_lag"


def _as_date(d) -> date:
    if isinstance(d, datetime):     # includes pd.Timestamp
        return d.date()
    return d


def fiscal_year_end_months(annual_period_ends: Iterable[date]) -> frozenset[int]:
    """Months whose quarter-end gets the annual lag: March plus every annual period end's month.

    March is always included, so the result never gives a shorter lag than the old
    March-only rule.
    """
    return frozenset({FISCAL_YEAR_END_MONTH}) | {_as_date(d).month for d in annual_period_ends}


def available_from(
    period_end: date,
    period_type: str,
    filing_date: date | None = None,
    *,
    fiscal_year_end_months: Collection[int] = frozenset({FISCAL_YEAR_END_MONTH}),
) -> tuple[date, str]:
    """First date a period's numbers may be used, and where that date came from.

    Returns ``(filing_date, "published")`` when the publication date is known,
    otherwise ``(period_end + lag, "lodr_lag")``. A quarterly row whose period end
    falls in one of ``fiscal_year_end_months`` (see ``fiscal_year_end_months()``) is a
    Q4 and gets the annual lag. Raises ``ValueError`` for an unknown ``period_type``
    or a ``filing_date`` before ``period_end``.
    """
    period_end = _as_date(period_end)
    if period_type == "annual":
        lag = ANNUAL_LAG_DAYS
    elif period_type == "quarterly":
        lag = ANNUAL_LAG_DAYS if period_end.month in fiscal_year_end_months else QUARTERLY_LAG_DAYS
    else:
        raise ValueError(f"unknown period_type {period_type!r} (expected 'annual' or 'quarterly')")

    if filing_date is not None:
        filing_date = _as_date(filing_date)
        if filing_date < period_end:
            raise ValueError(f"filing_date {filing_date} is before period_end {period_end}")
        return filing_date, SOURCE_PUBLISHED
    return period_end + timedelta(days=lag), SOURCE_LODR_LAG


def financials_as_of(fin: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Rows of ``fin`` that were public on ``as_of``, sorted by ``period_end``.

    ``fin`` needs ``available_from`` and ``period_end`` columns (dates, Timestamps
    or datetime64). A row with a missing ``available_from`` is never public.
    Columns, dtypes and index labels are kept; rows with NaN line items stay in.
    """
    avail = pd.to_datetime(fin["available_from"])
    visible = fin.loc[(avail <= pd.Timestamp(as_of)).to_numpy()]
    order = pd.to_datetime(visible["period_end"]).argsort(kind="stable")
    return visible.iloc[order.to_numpy()]
