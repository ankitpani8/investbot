"""
Walk-forward split utilities (expanding train, fixed-length test blocks, embargo).

Nothing is fitted yet: the harness reports every metric on the test blocks
only, so the numbers are what an expanding-window model would have seen out of
sample. The ML phase reuses the same splits.

    train  = every date from the first date up to (test_start - embargo - 1 bars)
    gap    = ``embargo`` bars right before the test block, in neither set
    test   = one calendar block of ``test_months`` months (Jan–Jun / Jul–Dec
             for 6), the last one possibly partial

With ``embargo = max(horizons)``, no training label (which looks h bars ahead)
can overlap a test bar, so purging and embargo are the same gap here.
The first test block is the first calendar block that starts after
``min_train`` bars plus the embargo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Length of one out-of-sample test block.
TEST_MONTHS: int = 6
#: Bars of history before the first test block (one trading year).
MIN_TRAIN_BARS: int = 252


@dataclass(frozen=True)
class Split:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def label(self) -> str:
        return f"{self.test[0]:%Y-%m-%d}..{self.test[-1]:%Y-%m-%d}"


def _block_start(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Start of the calendar block of ``months`` months that contains ``ts``."""
    m0 = (ts.month - 1) // months * months + 1
    return pd.Timestamp(year=ts.year, month=m0, day=1)


def walk_forward_splits(
    dates,
    *,
    embargo: int,
    test_months: int = TEST_MONTHS,
    min_train: int = MIN_TRAIN_BARS,
) -> list[Split]:
    """Expanding-window splits over the sorted unique ``dates`` (a trading calendar)."""
    if embargo < 0 or min_train < 1 or test_months < 1 or 12 % test_months:
        raise ValueError("need embargo >= 0, min_train >= 1 and test_months dividing 12")
    cal = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values()
    if len(cal) <= min_train + embargo:
        return []
    earliest = cal[min_train + embargo]
    first = _block_start(earliest, test_months)
    if first < earliest:
        first = first + pd.DateOffset(months=test_months)
    splits: list[Split] = []
    lo = first
    while lo <= cal[-1]:
        hi = lo + pd.DateOffset(months=test_months)
        test = cal[(cal >= lo) & (cal < hi)]
        if len(test):
            pos = int(np.searchsorted(cal, test[0]))       # index of the first test bar
            train = cal[: max(pos - embargo, 0)]
            splits.append(Split(train=train, test=test))
        lo = hi
    return splits


def oos_dates(splits: list[Split]) -> pd.DatetimeIndex:
    """Union of every test block."""
    if not splits:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([s.test.to_numpy() for s in splits]))
