"""
Grandfathered cost of acquisition for listed equity bought before 1 February 2018.

Pure compute: no DB, no network, no mutation of inputs. Section 55(2)(ac) of the Income-tax
Act (inserted by the Finance Act 2018, read with s.112A): for a long-term capital asset
acquired before 1 February 2018 and transferred on or after 1 April 2018, the cost of
acquisition is the higher of

- the actual cost, and
- the lower of the fair market value on 31 January 2018 and the full value of the sale
  consideration.

So the step-up never creates a loss (cost is capped at the sale value) and never lowers the
cost (a FMV below the actual cost leaves the actual cost). For a listed share the FMV is the
**highest price quoted on 31 January 2018**, or, if it did not trade that day, the highest
price on the last day before it on which it traded (Explanation to s.55(2)(ac)). Source
for the worked examples: CBDT FAQ on the LTCG tax (Circular/FAQ of 4 February 2018).

Not handled here: unlisted shares, units of equity funds and business trusts (same rule,
different FMV source), and the NAV-based FMV of funds.

Unit basis (what the caller must do)
Stored daily prices come from ``yf.download(auto_adjust=False)``: ``open/high/low/close``
are quoted in today's share units (Yahoo restates them for later splits and bonuses, but
not for dividends), ``adj_close`` is also dividend-adjusted and must never be used here.
A lot's ``buy_price`` and ``quantity`` are in the units of the day of purchase. A split or
bonus after the purchase changes the share count, so the caller has to put the lots and
``fmv_31_jan_2018`` on **one share basis** before calling ``sell_fifo``: either restate the
lots (quantity x ratio, price / ratio) or divide the stored FMV back by the cumulative
split ratio after 2018-01-31. Because stored history is restated at refresh time, a bar
fetched before a split can sit on the old basis (``detect_unadjusted_splits`` finds those).
This module does no corporate-action adjustment (BACKLOG follow-up).

Lookahead: only bars dated on or before ``GF_FMV_DATE`` are read.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

GF_ACQUIRED_BEFORE = date(2018, 2, 1)  # a lot bought on or after this date gets no step-up
GF_FMV_DATE = date(2018, 1, 31)
GF_TRANSFER_FROM = date(2018, 4, 1)  # only sales on or after this date use the grandfathered cost


def fmv_31_jan_2018(bars: pd.DataFrame) -> float | None:
    """Highest price (``high``) of 2018-01-31, else the ``high`` of the last bar before it.

    ``bars`` has a date-like index and a ``high`` column (the ``_load_daily`` frame works).
    Bars after 2018-01-31 are ignored, so appending later data never changes the result.
    Returns None when no bar exists on or before the date, or the high is NaN or <= 0.
    The input is not modified; NaN rows are not dropped (a NaN high on the chosen bar gives
    None rather than a silent fall back to an older bar).
    """
    if len(bars) == 0:
        return None
    idx = pd.DatetimeIndex(bars.index).normalize()
    eligible = np.flatnonzero(idx <= pd.Timestamp(GF_FMV_DATE))
    if eligible.size == 0:
        return None
    # the index need not be sorted: take the latest eligible bar by date
    last = eligible[np.argmax(idx[eligible].to_numpy())]
    high = float(bars["high"].iloc[last])
    if math.isnan(high) or high <= 0:
        return None
    return high


def grandfathered_cost(actual_cost: float, fmv: float, sale_value: float) -> float:
    """``max(actual_cost, min(fmv, sale_value))``, all on the same quantity (totals)."""
    return max(actual_cost, min(fmv, sale_value))


def applies(buy_date: date, sell_date: date, term: str) -> bool:
    """True for a long-term lot bought before 2018-02-01 and sold on or after 2018-04-01."""
    return term == "LTCG" and buy_date < GF_ACQUIRED_BEFORE and sell_date >= GF_TRANSFER_FROM
