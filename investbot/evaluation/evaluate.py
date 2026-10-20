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
              "oos:inverted"  descriptive, not a test: for each headline row that is promising with
                              ``direction == "inverted"``, the same metrics (IC, quintiles, cost
                              check, held portfolio, bootstrap) on the negated signal, same
                              labels. Selected after seeing the IC, so its numbers are conditional
                              on that selection; ``ic_q`` and ``block_sign_share`` are NaN,
                              ``promising`` False, ``direction`` "negated", and the rows never
                              enter a BH family

Then, on the headline rows (label "open", subset "oos") only:
    ic_q            Benjamini–Hochberg q-value across all those tests
    block_sign_share share of test blocks whose mean IC has the sign of the OOS mean IC
    promising       ic_q < PROMISING_Q and block_sign_share >= PROMISING_SIGN_SHARE

The "price" OOS rows get the same three columns from their own BH family
(the price-only labels, same tests), so a verdict can be compared across the
two returns; the headline family is unchanged.

Robust IC inference (task 340, pre-registered; a check, not a new rule): on the OOS rows of
each ``BH_LABELS`` label only, ``metrics.ic_robust`` of the same per-date IC series the row's
``ic_p`` uses adds ``ROBUST_KEYS`` (IC autocorrelations, an automatic Andrews lag with its
Newey-West t and p, a stationary block bootstrap p), then, within the same BH family as
``ic_q``:
    ic_q_auto / ic_q_boot           BH q-values of ``ic_p_auto`` / ``ic_p_boot``
    promising_auto / promising_boot the promising rule with that q and the same
                                    ``block_sign_share``
Every other row (``INVERTED_SUBSET`` included) has NaN there and False for the flags. These
columns come after every existing column, which stays bit-identical; ``robust=False`` skips
the step (all NaN / False).

Survivorship split (task 420, descriptive; no test): on the headline rows (label "open", subset
"oos") only, ``metrics.ic_complete_split`` of the same per-date IC series adds
``SURVIVORSHIP_KEYS`` (mean IC and number of dates before / from ``complete_from``, default
``index_membership.COMPLETE_FROM``). They come after the robust columns; every other row has NaN
there, and so does every row when ``complete_from=None``. No existing column or verdict changes.

Every row with quintile metrics also has the open-to-open held portfolio
(``held_*``, from ``FWD_TR_OO_RET_h``, or ``FWD_OO_RET_h`` on the "price" rows
and when the primary return is "price") next to the open→close ``lo_*`` check.

Bootstrap intervals: the held-portfolio block-bootstrap intervals (``held_*_lo/_hi``,
``held_ir_net_share_pos``, ``held_boot_mean_block``) are computed only on the rows listed in
``BOOTSTRAP_ROWS``, the (label, subset) pairs the report reads (today the headline rows,
"open" / "oos"). Every other row passes ``bootstrap=False`` and carries NaN there; all other
columns are unchanged. A report section that starts reading intervals on other rows must add
its (label, subset) to ``BOOTSTRAP_ROWS``.

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
import time

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import REGIME_NAMES
from investbot.data.index_membership import COMPLETE_FROM
from investbot.evaluation.labels import HORIZONS
from investbot.evaluation.metrics import (
    MIN_NAMES,
    ROBUST_KEYS,
    ROUND_TRIP_COST,
    daily_ic,
    SURVIVORSHIP_KEYS,
    ic_complete_split,
    ic_robust,
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
#: (label, subset) pairs whose held portfolio gets bootstrap intervals; the report's
#: "Held portfolio: bootstrap intervals" table reads the headline rows and nothing else does.
BOOTSTRAP_ROWS: tuple[tuple[str, str], ...] = (("open", "oos"),)
#: ``subset`` of the descriptive rows for promising-inverted signals (negated signal, no q-value).
INVERTED_SUBSET: str = "oos:inverted"
#: ``direction`` of those rows.
NEGATED_DIRECTION: str = "negated"
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
#: Columns ``_add_multiple_testing`` appends after ``ROBUST_KEYS``, with their empty value.
ROBUST_VERDICT_COLUMNS: tuple[tuple[str, object], ...] = (
    ("ic_q_auto", np.nan), ("ic_q_boot", np.nan),
    ("promising_auto", False), ("promising_boot", False))
#: Every column the robust step appends, in table order.
ROBUST_COLUMNS: tuple[str, ...] = (*ROBUST_KEYS, *(c for c, _ in ROBUST_VERDICT_COLUMNS))
#: ``timings`` key of the robust IC step (seconds, summed over calls).
ROBUST_TIMING_KEY: str = "robust_ic"


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


def signal_matrix(features: pd.DataFrame, sig: str, dates: pd.DatetimeIndex, tickers,
                  U: pd.DataFrame | None) -> pd.DataFrame:
    """Wide (dates × tickers) matrix of one signal, cells outside the universe NaN."""
    return apply_universe(_wide(features[sig], dates, tickers), U)


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
    robust: bool = True,
    complete_from=COMPLETE_FROM,
    timings: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[Split]]:
    """(metrics table, walk-forward splits). See the module docstring for the rows.

    ``primary="price"`` gives the pre-task-200 table (no "price" comparison rows).
    ``robust=False`` skips the robust IC step (its columns stay NaN / False).
    ``complete_from``: date splitting the headline OOS ICs into the survivorship columns
    (``SURVIVORSHIP_KEYS``); None leaves them NaN.
    ``timings``: if given, the seconds of the robust step are added under
    ``ROBUST_TIMING_KEY``."""
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
    #: row position -> ``ic_robust`` of a BH row (OOS, label in BH_LABELS)
    robust_rows: dict[int, dict[str, float]] = {}
    robust_s = 0.0
    #: row position -> ``ic_complete_split`` of a headline row
    split_rows: dict[int, dict[str, float]] = {}

    def add_robust(ic_oos: pd.Series, h: int) -> None:
        nonlocal robust_s
        if robust:
            t0 = time.monotonic()
            robust_rows[len(rows) - 1] = ic_robust(ic_oos, h)
            robust_s += time.monotonic() - t0

    for sig in signals:
        S = signal_matrix(features, sig, cal, tickers, U)
        if not np.isfinite(S.to_numpy(dtype=np.float64)).any():
            continue                       # e.g. TECH_SCORE_MTF on daily-only frames
        for h in horizons:
            Y = ys[("open", h)]
            Yo = ys[("oo", h)]

            def boot(label, subset):
                return (label, subset) in BOOTSTRAP_ROWS

            def add(subset, label, metrics):
                rows.append({"signal": sig, "label": label, "horizon": h,
                             "subset": subset, **metrics})

            add("oos", "open", signal_metrics(S.loc[oos], Y.loc[oos], h, min_names, cost,
                                              Yo.loc[oos], boot("open", "oos")))
            ic = daily_ic(S, Y, min_names)
            add_robust(ic.loc[oos], h)     # per-date, so equal to the OOS row's own IC series
            if complete_from is not None:
                split_rows[len(rows) - 1] = ic_complete_split(ic.loc[oos], complete_from)
            add("all", "open", signal_metrics(S, Y, h, min_names, cost, Yo,
                                              boot("open", "all")))
            Yc = ys[("close", h)]
            add("oos", "close", signal_metrics(S.loc[oos], Yc.loc[oos], h, min_names, cost,
                                               Yo.loc[oos], boot("close", "oos")))

            for sp in splits:
                add(f"block:{sp.label}", "open", ic_summary(ic.loc[sp.test], h))
            if compare:
                Yp, Yop = ys[(("price", "open"), h)], ys[(("price", "oo"), h)]
                add("oos", COMPARISON_RETURN,
                    signal_metrics(S.loc[oos], Yp.loc[oos], h, min_names, cost, Yop.loc[oos],
                                   boot(COMPARISON_RETURN, "oos")))
                icp = daily_ic(S, Yp, min_names)
                add_robust(icp.loc[oos], h)
                for sp in splits:
                    add(f"block:{sp.label}", COMPARISON_RETURN, ic_summary(icp.loc[sp.test], h))
            for code, name in REGIME_NAMES.items():
                Sr = S.loc[oos].where(regime.loc[oos] == code)
                add(f"regime:{name}", "open",
                    {**signal_metrics(Sr, Y.loc[oos], h, min_names, cost, Yo.loc[oos],
                                   boot("open", f"regime:{name}")),
                     **pooled_ic(Sr, Y.loc[oos], h)})
            for yr in years:
                d = oos[oos.year == yr]
                add(f"year:{yr}", "open",
                    signal_metrics(S.loc[d], Y.loc[d], h, min_names, cost, Yo.loc[d],
                                   boot("open", f"year:{yr}")))

    table = pd.DataFrame(rows)
    if table.empty:
        return table, splits
    if timings is not None:
        timings[ROBUST_TIMING_KEY] = timings.get(ROBUST_TIMING_KEY, 0.0) + robust_s
    table = _add_multiple_testing(table, pd.DataFrame.from_dict(robust_rows, orient="index"))
    table = _add_survivorship_split(table, split_rows)
    inverted = promising_inverted(table)
    extra = []
    for sig, h in inverted:
        S = signal_matrix(features, sig, cal, tickers, U)
        Y, Yo = ys[("open", h)], ys[("oo", h)]
        extra.append({"signal": sig, "label": "open", "horizon": h, "subset": INVERTED_SUBSET,
                      **signal_metrics(-S.loc[oos], Y.loc[oos], h, min_names, cost, Yo.loc[oos],
                                       True),
                      "ic_q": np.nan, "block_sign_share": np.nan, "promising": False,
                      "direction": NEGATED_DIRECTION,
                      **dict.fromkeys(ROBUST_KEYS, np.nan), **dict(ROBUST_VERDICT_COLUMNS),
                      **dict.fromkeys(SURVIVORSHIP_KEYS, np.nan)})
    if extra:
        table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    return table, splits


def promising_inverted(table: pd.DataFrame) -> list[tuple[str, int]]:
    """(signal, horizon) of every headline row that is promising with an inverted IC."""
    head = headline(table)
    sel = head[head["promising"].astype(bool) & (head["direction"] == "inverted")]
    return [(str(s), int(h)) for s, h in zip(sel["signal"], sel["horizon"])]


def _add_multiple_testing(table: pd.DataFrame,
                          robust: pd.DataFrame | None = None) -> pd.DataFrame:
    """BH q, block sign share, promising and direction on the OOS rows of each ``BH_LABELS``
    label, each label its own family (the headline family is the "open" rows).

    ``robust``: ``ic_robust`` per row position (index = ``table`` index, BH rows only); its
    ``ROBUST_KEYS`` and the ``ROBUST_VERDICT_COLUMNS`` are appended after ``direction``
    (NaN / False elsewhere), and ``ic_q_auto`` / ``ic_q_boot`` / ``promising_auto`` /
    ``promising_boot`` are filled within the same families as ``ic_q``. Without ``robust``,
    ``ROBUST_KEYS`` columns already in ``table`` are kept (re-running gives the same table)."""
    for col, empty in (("ic_q", np.nan), ("block_sign_share", np.nan), ("promising", False),
                       ("direction", "")):
        table[col] = empty
    if robust is None:
        robust = table[[k for k in ROBUST_KEYS if k in table.columns]]
    table = table.drop(columns=[c for c in (*ROBUST_KEYS, *dict(ROBUST_VERDICT_COLUMNS))
                                if c in table.columns])
    extra = {k: (robust[k].reindex(table.index).to_numpy(dtype=np.float64) if k in robust
                 else np.full(len(table), np.nan)) for k in ROBUST_KEYS}
    extra.update({col: np.full(len(table), empty, dtype=bool if empty is False else np.float64)
                  for col, empty in ROBUST_VERDICT_COLUMNS})
    table = pd.concat([table, pd.DataFrame(extra, index=table.index)], axis=1)
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
        for kind in ("auto", "boot"):
            table.loc[head, f"ic_q_{kind}"] = bh_adjust(table.loc[head, f"ic_p_{kind}"].to_numpy())
            table.loc[head, f"promising_{kind}"] = (
                (table.loc[head, f"ic_q_{kind}"] < PROMISING_Q)
                & (table.loc[head, "block_sign_share"] >= PROMISING_SIGN_SHARE))
        table.loc[head, "direction"] = np.where(table.loc[head, "ic_mean"] >= 0, "as designed",
                                                "inverted")
    return table


def _add_survivorship_split(table: pd.DataFrame,
                            split_rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Append ``SURVIVORSHIP_KEYS`` after every existing column: the ``ic_complete_split`` of
    the headline rows in ``split_rows`` (keyed by row position), NaN elsewhere."""
    cols = {k: np.full(len(table), np.nan) for k in SURVIVORSHIP_KEYS}
    for pos, vals in split_rows.items():
        for k in SURVIVORSHIP_KEYS:
            cols[k][pos] = vals[k]
    return pd.concat([table, pd.DataFrame(cols, index=table.index)], axis=1)


def headline(table: pd.DataFrame, label: str = "open") -> pd.DataFrame:
    """The BH-tested rows: label "open" (the primary return), subset "oos".
    ``label=COMPARISON_RETURN`` gives the price-only comparison rows."""
    return table[(table["label"] == label) & (table["subset"] == "oos")].copy()
