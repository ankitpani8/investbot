"""
Walk-forward evaluation of every signal × horizon on a (features, labels) panel.

Pure compute: takes the two frames from ``panel.build_dataset`` and returns a
long metrics table, one row per (signal, label, horizon, subset):

    label     "open"   FWD_RET_h    (entry at the next bar's open; the headline)
              "close"  FWD_CC_RET_h (close-to-close; comparison only, OOS subset)
    subset    "oos"           every walk-forward test block (the headline)
              "all"           every date, train periods included (context only)
              "block:<a..b>"  one test block (IC columns only)
              "regime:<r>"    OOS dates, names whose REGIME on that date is r; these rows
                              also carry the pooled IC (``pooled_*``, NaN elsewhere)
              "year:<YYYY>"   OOS dates of one calendar year

Then, on the headline rows (label "open", subset "oos") only:
    ic_q            Benjamini–Hochberg q-value across all those tests
    block_sign_share share of test blocks whose mean IC has the sign of the OOS mean IC
    promising       ic_q < PROMISING_Q and block_sign_share >= PROMISING_SIGN_SHARE

Every row with quintile metrics also has the open-to-open held portfolio
(``held_*``, from ``FWD_OO_RET_h``) next to the open→close ``lo_*`` check.

Universe masking: when the features carry ``IN_UNIVERSE`` (``universe="index"``),
every signal cell outside the universe on its date becomes NaN (``where``, no
dropna) before IC, buckets and portfolios, baselines included, and so does the
label used there, so the equal-weight universe benchmark is the members on the
rebalance date. A member's label still reaches past its exit date. Without the
column nothing is masked (``universe="active"``).

Nothing is fitted, and a signal's polarity is used as designed: a significant
negative IC is reported as such (``direction``), not flipped.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import REGIME_NAMES
from investbot.evaluation.labels import HORIZONS
from investbot.evaluation.metrics import (
    MIN_NAMES,
    ROUND_TRIP_COST,
    daily_ic,
    ic_summary,
    pooled_ic,
    signal_metrics,
)
from investbot.evaluation.panel import UNIVERSE_COLUMN, signal_columns
from investbot.evaluation.stats import bh_adjust
from investbot.evaluation.walkforward import Split, oos_dates, walk_forward_splits

#: BH q-value below which a test counts as significant.
PROMISING_Q: float = 0.10
#: Share of test blocks in which the IC sign must hold.
PROMISING_SIGN_SHARE: float = 0.70
#: Column whose cross-section defines the evaluation calendar (first usable date).
CALENDAR_COLUMN: str = "TECH_SCORE"


def universe_mask(features: pd.DataFrame) -> pd.Series | None:
    """``IN_UNIVERSE`` as a bool Series (missing cells = False), or None without the column."""
    if UNIVERSE_COLUMN not in features.columns:
        return None
    return features[UNIVERSE_COLUMN].eq(True)


def evaluation_calendar(features: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.DatetimeIndex:
    """Dates from the first one where ``CALENDAR_COLUMN`` has >= min_names values
    (inside the universe when ``IN_UNIVERSE`` is present)."""
    col = features[CALENDAR_COLUMN]
    mask = universe_mask(features)
    if mask is not None:
        col = col.where(mask)
    counts = col.notna().groupby(level="date").sum()
    dates = pd.DatetimeIndex(counts.index)
    usable = counts.to_numpy() >= min_names
    if not usable.any():
        return pd.DatetimeIndex([])
    return dates[int(np.argmax(usable)):]


def _wide(series: pd.Series, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    return series.unstack("ticker").reindex(index=dates, columns=tickers)


def universe_frame(features: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame | None:
    """Wide (dates × tickers) bool ``IN_UNIVERSE`` (absent cells False), or None without it."""
    mask = universe_mask(features)
    if mask is None:
        return None
    return _wide(mask.astype(np.float64), dates, tickers).eq(1.0)


def apply_universe(frame: pd.DataFrame, U: pd.DataFrame | None) -> pd.DataFrame:
    """Cells outside the universe become NaN (``where``, nothing dropped); no-op if U is None."""
    return frame if U is None else frame.where(U)


def evaluate(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizons=HORIZONS,
    *,
    signals: list[str] | None = None,
    splits: list[Split] | None = None,
    min_names: int = MIN_NAMES,
    cost: float = ROUND_TRIP_COST,
) -> tuple[pd.DataFrame, list[Split]]:
    """(metrics table, walk-forward splits). See the module docstring for the rows."""
    horizons = tuple(sorted(int(h) for h in horizons))
    cal = evaluation_calendar(features, min_names)
    if splits is None:
        splits = walk_forward_splits(cal, embargo=max(horizons))
    oos = oos_dates(splits)
    if not oos.isin(cal).all():        # splits from another universe's calendar
        cal = cal.union(oos)
    tickers = sorted(features.index.get_level_values("ticker").unique())
    signals = signals if signals is not None else signal_columns(features)

    regime = _wide(features["REGIME"], cal, tickers)
    U = universe_frame(features, cal, tickers)

    def in_universe(frame: pd.DataFrame) -> pd.DataFrame:
        return apply_universe(frame, U)

    ys = {(kind, h): in_universe(_wide(labels[f"{col}_{h}"], cal, tickers))
          for h in horizons for kind, col in (("open", "FWD_RET"), ("close", "FWD_CC_RET"),
                                              ("oo", "FWD_OO_RET"))}
    years = sorted(set(oos.year))

    rows: list[dict] = []
    for sig in signals:
        S = in_universe(_wide(features[sig], cal, tickers))
        if not np.isfinite(S.to_numpy(dtype=np.float64)).any():
            continue                       # e.g. TECH_SCORE_MTF on daily-only frames
        for h in horizons:
            Y = ys[("open", h)]
            Yo = ys[("oo", h)]

            def add(subset, label, metrics):
                rows.append({"signal": sig, "label": label, "horizon": h,
                             "subset": subset, **metrics})

            add("oos", "open", signal_metrics(S.loc[oos], Y.loc[oos], h, min_names, cost,
                                              Yo.loc[oos]))
            add("all", "open", signal_metrics(S, Y, h, min_names, cost, Yo))
            Yc = ys[("close", h)]
            add("oos", "close", signal_metrics(S.loc[oos], Yc.loc[oos], h, min_names, cost,
                                               Yo.loc[oos]))

            ic = daily_ic(S, Y, min_names)
            for sp in splits:
                add(f"block:{sp.label}", "open", ic_summary(ic.loc[sp.test], h))
            for code, name in REGIME_NAMES.items():
                Sr = S.loc[oos].where(regime.loc[oos] == code)
                add(f"regime:{name}", "open",
                    {**signal_metrics(Sr, Y.loc[oos], h, min_names, cost, Yo.loc[oos]),
                     **pooled_ic(Sr, Y.loc[oos], h)})
            for yr in years:
                d = oos[oos.year == yr]
                add(f"year:{yr}", "open",
                    signal_metrics(S.loc[d], Y.loc[d], h, min_names, cost, Yo.loc[d]))

    table = pd.DataFrame(rows)
    if table.empty:
        return table, splits
    return _add_multiple_testing(table), splits


def _add_multiple_testing(table: pd.DataFrame) -> pd.DataFrame:
    head = (table["label"] == "open") & (table["subset"] == "oos")
    table["ic_q"] = np.nan
    table.loc[head, "ic_q"] = bh_adjust(table.loc[head, "ic_p"].to_numpy())

    blocks = table[(table["label"] == "open") & table["subset"].str.startswith("block:")]
    oos_sign = table[head].set_index(["signal", "horizon"])["ic_mean"].apply(np.sign)
    share = {}
    for (sig, h), grp in blocks.groupby(["signal", "horizon"]):
        means = grp["ic_mean"].to_numpy(dtype=np.float64)
        means = means[np.isfinite(means)]
        target = oos_sign.get((sig, h), math.nan)
        share[(sig, h)] = (float((np.sign(means) == target).mean())
                           if means.size and np.isfinite(target) and target != 0 else math.nan)
    table["block_sign_share"] = np.nan
    table.loc[head, "block_sign_share"] = [share.get((s, h), math.nan) for s, h in
                                           zip(table.loc[head, "signal"], table.loc[head, "horizon"])]
    table["promising"] = False
    table.loc[head, "promising"] = ((table.loc[head, "ic_q"] < PROMISING_Q)
                                    & (table.loc[head, "block_sign_share"] >= PROMISING_SIGN_SHARE))
    table["direction"] = ""
    table.loc[head, "direction"] = np.where(table.loc[head, "ic_mean"] >= 0, "as designed", "inverted")
    return table


def headline(table: pd.DataFrame) -> pd.DataFrame:
    """The BH-tested rows: label "open", subset "oos"."""
    return table[(table["label"] == "open") & (table["subset"] == "oos")].copy()
