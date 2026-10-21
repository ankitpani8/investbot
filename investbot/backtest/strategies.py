"""
Strategy-to-weights adapter: a point-in-time signal panel -> a target-weight schedule
for ``simulator.simulate``.

Pure compute: no DB, no network, inputs are never mutated. The module presumes no
decision horizon: the rebalance frequency is a required argument, so the same code runs a
daily technical strategy (``freq="D"``) and a monthly slow-factor one (``freq="M"``).

Timing (the lookahead contract)
    A weight row dated *t* is decided after the close of session *t*; the simulator
    executes it at the open of the next session. The row uses only ``signal.loc[t]`` and
    ``membership.loc[t]``, so it is causal as long as the signal at *t* uses data up to
    *t*. The harness features are (truncation-invariance tests in ``evaluation/panel.py``
    and ``tests/test_indicators.py``), and point-in-time membership is by construction.

Rebalance calendar (``rebalance_sessions``)
    ``"D"`` every session; ``"<n>D"`` every n-th session, counted from the first session
    of the index (so the phase depends on where the panel starts); ``"W"`` / ``"M"`` the
    last session of each ISO week / calendar month that is present in ``sessions``. A week
    or month counts as ended only once a later session in a later period exists in
    ``sessions``; the trailing period is therefore excluded. Without that rule its last
    *known* session would move as data arrives (a Wednesday row today, a Thursday row
    tomorrow), which is lookahead. Consequence: on data truncated at a period's last
    session *s*, the row dated *s* appears only once the next session is in the index;
    every row that is present is identical to the full-data row.

Selection (``top_quantile_weights``)
    On each rebalance session the eligible names are members with a finite signal. Fewer
    than ``min_names`` eligible: the row is all zero (cash). Otherwise the top
    ``ceil(quantile x n_eligible)`` names are selected (bottom, with
    ``higher_is_better=False``) and every name tied with the cutoff value is included too,
    so a row can hold more than ``ceil(quantile x n)`` names; weights are 1 / n_selected.
    Nothing is forward-filled between rebalance rows: the simulator holds positions
    between decision rows by itself, and a name that leaves the index gets weight 0 on the
    next rebalance (it stays held until then).
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd

from investbot.evaluation.metrics import MIN_NAMES

log = logging.getLogger(__name__)

#: Share of the eligible names that is selected (top quintile, as the harness).
DEFAULT_QUANTILE: float = 0.2

#: ``quantile x n`` is rounded up after subtracting this, so float noise such as
#: 0.1 x 30 = 3.0000000000000004 still selects 3 names, not 4.
_CEIL_TOL = 1e-9

_EVERY_N = re.compile(r"^([1-9][0-9]*)D$")


def rebalance_sessions(sessions: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Decision sessions of ``sessions`` for ``freq`` (see the module docstring).

    ``freq``: ``"D"``, ``"W"``, ``"M"`` or ``"<n>D"`` (n >= 1). ``sessions`` must be
    strictly increasing. The trailing, possibly unfinished, week or month is excluded.
    Raises ValueError for an unknown ``freq`` or an unsorted index.
    """
    idx = pd.DatetimeIndex(sessions)
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("sessions must be strictly increasing")
    if not isinstance(freq, str):
        raise ValueError(f"unknown rebalance freq {freq!r}")
    if freq == "D":
        return idx
    every = _EVERY_N.match(freq)
    if every:
        return idx[:: int(every.group(1))]
    if freq == "W":
        iso = idx.isocalendar()
        key = iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64)
    elif freq == "M":
        key = idx.year.to_numpy(dtype=np.int64) * 12 + idx.month.to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"unknown rebalance freq {freq!r} (expected 'D', 'W', 'M' or '<n>D')")
    # last session of a period that a later session (in a later period) closes
    ended = np.zeros(len(idx), dtype=bool)
    ended[:-1] = key[1:] != key[:-1]
    return idx[ended]


def _membership_mask(membership: pd.DataFrame, index: pd.Index, columns: pd.Index) -> np.ndarray:
    """Boolean array aligned to (index, columns); missing or non-True cells are False."""
    m = membership.reindex(index=index, columns=columns)
    return np.asarray(m.to_numpy() == True, dtype=bool)  # noqa: E712 (NaN -> False)


def _weights(
    values: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    freq: str,
    quantile: float,
    min_names: int,
    higher_is_better: bool,
    label: str,
) -> pd.DataFrame:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if min_names < 1:
        raise ValueError("min_names must be >= 1")
    dates = rebalance_sessions(pd.DatetimeIndex(values.index), freq)
    rows_idx = values.index.get_indexer(dates)
    sub = values.iloc[rows_idx]
    vals = sub.to_numpy(dtype=float, copy=True)
    member = _membership_mask(membership, sub.index, values.columns)

    rows: dict[pd.Timestamp, np.ndarray] = {}
    n_cash = 0
    for i, t in enumerate(sub.index):
        v = vals[i]
        eligible = member[i] & np.isfinite(v)
        n = int(eligible.sum())
        w = np.zeros(len(v))
        if n < min_names:
            n_cash += 1
        else:
            k = max(1, math.ceil(quantile * n - _CEIL_TOL))
            ranked = np.sort(v[eligible])
            if higher_is_better:
                chosen = eligible & (v >= ranked[n - k])
            else:
                chosen = eligible & (v <= ranked[k - 1])
            w[chosen] = 1.0 / int(chosen.sum())
        rows[t] = w
    if n_cash:
        log.info("%s: %d of %d rebalance rows have < %d eligible names (all cash)",
                 label, n_cash, len(rows), min_names)
    out = (pd.DataFrame.from_dict(rows, orient="index", columns=values.columns) if rows
           else pd.DataFrame(columns=values.columns, dtype=float))
    out.index = pd.DatetimeIndex(list(rows), name=values.index.name)
    return out


def top_quantile_weights(
    signal: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    freq: str,
    quantile: float = DEFAULT_QUANTILE,
    min_names: int = MIN_NAMES,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Equal-weight top-quantile target weights on the rebalance sessions of ``signal``.

    ``signal``: sessions x asset (any harness column, e.g. ``TECH_SCORE``, ``SIG_*``,
    ``BASE_*``; see ``signal_panel``). ``membership``: sessions x asset, True where the
    asset is an index member on that session; it is aligned to ``signal`` and a missing
    or non-True cell counts as not a member.

    Returns rebalance sessions (``rebalance_sessions(signal.index, freq)``) x
    ``signal.columns``. A row dated *t* is decided after the close of *t* and executed
    by the simulator at the next open, so ``signal.loc[t]`` must use data up to *t* only
    (the harness features do; see the module docstring). Eligible = member and finite
    signal; fewer than ``min_names`` eligible -> an all-zero (cash) row, counted in an
    INFO log; otherwise the best ``ceil(quantile x n_eligible)`` names plus every name
    tied with the cutoff, each weighted 1 / n_selected (row sum 1). With
    ``higher_is_better=False`` the lowest values are selected.
    """
    return _weights(signal, membership, freq=freq, quantile=quantile, min_names=min_names,
                    higher_is_better=higher_is_better, label="top_quantile_weights")


def equal_weight_universe(
    membership: pd.DataFrame,
    close: pd.DataFrame,
    *,
    freq: str,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Benchmark: equal weight over members with a finite close, on each rebalance session.

    The simulator counterpart of the harness's equal-weight universe baseline. Returns
    rebalance sessions of ``close.index`` x ``close.columns``; fewer than ``min_names``
    eligible names -> an all-zero (cash) row. Same timing contract as
    ``top_quantile_weights``.
    """
    return _weights(close, membership, freq=freq, quantile=1.0, min_names=min_names,
                    higher_is_better=True, label="equal_weight_universe")


def signal_panel(
    features: pd.DataFrame, column: str, ids: dict[str, int] | None = None
) -> pd.DataFrame:
    """Wide sessions x asset frame of one column of the harness's stacked feature frame.

    ``features`` is indexed by (``date``, ``ticker``) as ``evaluation.panel.build_panel``
    returns it. With ``ids`` (ticker -> asset_id) the columns are asset_ids (``asset_id``,
    ascending), as the simulator's price frames; a ticker missing from ``ids`` raises
    ValueError. Without ``ids`` the columns are tickers. Dates where a ticker has no row
    are NaN; no row is dropped.
    """
    if column not in features.columns:
        raise KeyError(f"{column!r} is not a column of the feature panel")
    if list(features.index.names) != ["date", "ticker"]:
        raise ValueError("features must be indexed by (date, ticker)")
    wide = features[column].unstack("ticker").sort_index()
    wide.index = pd.DatetimeIndex(wide.index, name="date")
    if ids is None:
        return wide.sort_index(axis=1)
    unmapped = sorted(set(wide.columns) - set(ids))
    if unmapped:
        raise ValueError(f"tickers with no asset_id: {unmapped}")
    wide.columns = pd.Index([int(ids[t]) for t in wide.columns], dtype="int64", name="asset_id")
    return wide.sort_index(axis=1)
