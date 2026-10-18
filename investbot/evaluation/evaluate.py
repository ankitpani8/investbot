"""
Walk-forward evaluation of every signal × horizon on a (features, labels) panel.

Pure compute: takes the two frames from ``panel.build_dataset`` and returns a
long metrics table, one row per (signal, label, horizon, subset):

    label     "open"   next-open label of ``PRIMARY_RETURN`` (the headline):
                       FWD_TR_RET_h (total return, dividends on their ex-dates) for "tr",
                       FWD_RET_h for "price"
              "close"  close-to-close label of the same return (FWD_TR_CC_RET_h /
                       FWD_CC_RET_h; comparison only, OOS subset)
              "price"  FWD_RET_h with the FWD_OO_RET_h held portfolio: the price-only
                       comparison (subsets "oos" and "block:*"; only when the primary
                       return is not "price")
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

The "price" OOS rows get the same three columns from their own BH family
(the price-only labels, same tests), so a verdict can be compared across the
two returns; the headline family is unchanged.

Every row with quintile metrics also has the open-to-open held portfolio
(``held_*``, from ``FWD_TR_OO_RET_h``, or ``FWD_OO_RET_h`` on the "price" rows
and when the primary return is "price") next to the open→close ``lo_*`` check.

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

#: Label stems per return definition: next-open ("open"), close-to-close ("close") and
#: the open-to-open held portfolio ("oo").
RETURN_LABELS: dict[str, dict[str, str]] = {
    "tr": {"open": "FWD_TR_RET", "close": "FWD_TR_CC_RET", "oo": "FWD_TR_OO_RET"},
    "price": {"open": "FWD_RET", "close": "FWD_CC_RET", "oo": "FWD_OO_RET"},
}
#: The return every headline metric uses (IC, BH, quintiles, cost check, held portfolio and
#: the universe benchmark): total return. Price-only is evaluated next to it as "price".
PRIMARY_RETURN: str = "tr"
#: The comparison return and its ``label`` value in the metrics table.
COMPARISON_RETURN: str = "price"
#: ``label`` values that carry their own BH family (headline first).
BH_LABELS: tuple[str, ...] = ("open", COMPARISON_RETURN)


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
    primary: str = PRIMARY_RETURN,
) -> tuple[pd.DataFrame, list[Split]]:
    """(metrics table, walk-forward splits). See the module docstring for the rows.

    ``primary="price"`` gives the pre-task-200 table (no "price" comparison rows)."""
    if primary not in RETURN_LABELS:
        raise ValueError(f"primary must be one of {tuple(RETURN_LABELS)}, not {primary!r}")
    compare = primary != COMPARISON_RETURN
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

    stems = {kind: RETURN_LABELS[primary][kind] for kind in ("open", "close", "oo")}
    if compare:
        stems.update({("price", "open"): RETURN_LABELS[COMPARISON_RETURN]["open"],
                      ("price", "oo"): RETURN_LABELS[COMPARISON_RETURN]["oo"]})
    ys = {(kind, h): in_universe(_wide(labels[f"{col}_{h}"], cal, tickers))
          for h in horizons for kind, col in stems.items()}
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
            if compare:
                Yp, Yop = ys[(("price", "open"), h)], ys[(("price", "oo"), h)]
                add("oos", COMPARISON_RETURN,
                    signal_metrics(S.loc[oos], Yp.loc[oos], h, min_names, cost, Yop.loc[oos]))
                icp = daily_ic(S, Yp, min_names)
                for sp in splits:
                    add(f"block:{sp.label}", COMPARISON_RETURN, ic_summary(icp.loc[sp.test], h))
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
    """BH q, block sign share, promising and direction on the OOS rows of each ``BH_LABELS``
    label, each label its own family (the headline family is the "open" rows)."""
    for col, empty in (("ic_q", np.nan), ("block_sign_share", np.nan), ("promising", False),
                       ("direction", "")):
        table[col] = empty
    for label in BH_LABELS:
        head = (table["label"] == label) & (table["subset"] == "oos")
        if not head.any():
            continue
        table.loc[head, "ic_q"] = bh_adjust(table.loc[head, "ic_p"].to_numpy())
        blocks = table[(table["label"] == label) & table["subset"].str.startswith("block:")]
        oos_sign = table[head].set_index(["signal", "horizon"])["ic_mean"].apply(np.sign)
        share = {}
        for (sig, h), grp in blocks.groupby(["signal", "horizon"]):
            means = grp["ic_mean"].to_numpy(dtype=np.float64)
            means = means[np.isfinite(means)]
            target = oos_sign.get((sig, h), math.nan)
            share[(sig, h)] = (float((np.sign(means) == target).mean())
                               if means.size and np.isfinite(target) and target != 0
                               else math.nan)
        table.loc[head, "block_sign_share"] = [
            share.get((s, h), math.nan)
            for s, h in zip(table.loc[head, "signal"], table.loc[head, "horizon"])]
        table.loc[head, "promising"] = (
            (table.loc[head, "ic_q"] < PROMISING_Q)
            & (table.loc[head, "block_sign_share"] >= PROMISING_SIGN_SHARE))
        table.loc[head, "direction"] = np.where(table.loc[head, "ic_mean"] >= 0, "as designed",
                                                "inverted")
    return table


def headline(table: pd.DataFrame, label: str = "open") -> pd.DataFrame:
    """The BH-tested rows: label "open" (the primary return), subset "oos".
    ``label=COMPARISON_RETURN`` gives the price-only comparison rows."""
    return table[(table["label"] == label) & (table["subset"] == "oos")].copy()
