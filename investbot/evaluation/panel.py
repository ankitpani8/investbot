"""
Panel builder: (date, ticker) feature panel and a separate forward-return label panel.

For each asset, one at a time, the full daily history (up to ``end``) is
loaded and the feature frame computed **once** with
``scorer.compute_daily_frame``. That frame is truncation-invariant (a value at
bar t uses bars <= t only, tested per module and in
``tests/test_evaluation.py``), so the full-history value at t equals what was
knowable at t's close. Only the evaluated columns are kept:

    TECH_SCORE, TECH_SCORE_<FAMILY>     confluence outputs
    SIG_<NAME>                          every confluence signal (``compute_signals``),
                                        prefixed because some names (VP_POS) are
                                        also upstream column names
    VAR_*                               pre-registered polarity variants of TECH_SCORE
                                        and TECH_SCORE_MOMENTUM (``variants.py``), from
                                        the same frame (signal + aggregation steps only);
                                        only with ``variants=True`` (``run_evaluation``
                                        always sets it; off by default so the pre-180
                                        panel, and its golden tests, stay unchanged)
    REGIME, REGIME_CONFIDENCE           conditioning variables, not signals
    BASE_*                              baseline factors (``baselines.py``)

Labels (``labels.py``) are computed from the same bars into a different frame,
together with the asset's ``corporate_actions`` (``load_actions``, ex-date <=
``end``): cash dividends credited on their ex-date bars for the total-return
labels, and the ex-date bars of splits that ``quality.detect_unadjusted_splits``
flags, whose spanning label periods become NaN. Features never see either.
``start`` only trims the output; features always warm up on the full history.
Bars after ``end`` are never loaded, so labels near ``end`` are NaN rather
than reaching past it.

Universe (``build_dataset(universe=...)``)
    "active"  every active stock, no membership column (the pre-170 behaviour).
    "index"   every past or present member of ``index_code`` in [start, end]
              (``index_membership``), active or not. Features are still computed
              on each asset's **full** stored history (warmup before it joined the
              index was knowable at the time), and a boolean ``IN_UNIVERSE``
              column says whether it was a member on date t (changes effective
              <= t only). Before ``COMPLETE_FROM`` the change chain is not
              sourced, so those dates fall back to "active" membership (an asset
              is in the universe there iff it is active today).
              ``evaluate.py`` masks every signal (and the label used for the
              universe benchmark) with ``IN_UNIVERSE``. Labels are never masked
              here: a name that leaves at t+3 still earns its return for a
              position opened at t.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import FAMILIES
from investbot.data.index_membership import COMPLETE_FROM
from investbot.evaluation.baselines import compute_baselines
from investbot.data.corporate_actions import dividends_on_bars
from investbot.data.quality import SplitIssue, detect_unadjusted_splits
from investbot.evaluation.labels import HORIZONS, forward_returns
from investbot.evaluation.variants import compute_variants
from investbot.scoring.technical.confluence import compute_signals
from investbot.scoring.technical.scorer import MIN_BARS, compute_daily_frame

logger = logging.getLogger(__name__)

SIGNAL_PREFIX = "SIG_"
SCORE_COLUMNS: tuple[str, ...] = ("TECH_SCORE", *(f"TECH_SCORE_{f.upper()}" for f in FAMILIES))
#: Point-in-time index membership on date t (``universe="index"`` only).
UNIVERSE_COLUMN: str = "IN_UNIVERSE"
CONDITIONING_COLUMNS: tuple[str, ...] = ("REGIME", "REGIME_CONFIDENCE", UNIVERSE_COLUMN)
UNIVERSES: tuple[str, ...] = ("index", "active")


#: Key in the ``timings`` dict for the seconds spent in ``compute_variants``.
VARIANT_TIMING_KEY: str = "variants_s"


def asset_features(bars: pd.DataFrame, *, variants: bool = False,
                   timings: dict[str, float] | None = None) -> pd.DataFrame:
    """Evaluated feature columns for one asset's daily OHLCV (pure; no dropna).

    ``variants`` adds the ``VAR_*`` columns (``variants.compute_variants`` on the
    same frame). ``timings``, if given, accumulates that step's seconds under
    ``VARIANT_TIMING_KEY`` (the runtime the variants add to feature building).
    """
    frame = compute_daily_frame(bars)
    cols: dict[str, np.ndarray] = {c: frame[c].to_numpy(dtype=np.float64) for c in SCORE_COLUMNS}
    signals = compute_signals(frame)
    for name, values in signals.items():
        cols[SIGNAL_PREFIX + name] = values
    if variants:
        t0 = time.perf_counter()
        cols.update(compute_variants(frame, signals=signals))
        if timings is not None:
            timings[VARIANT_TIMING_KEY] = (timings.get(VARIANT_TIMING_KEY, 0.0)
                                           + time.perf_counter() - t0)
    for c in ("REGIME", "REGIME_CONFIDENCE"):
        cols[c] = frame[c].to_numpy(dtype=np.float64)
    cols.update(compute_baselines(bars))
    return pd.DataFrame(cols, index=bars.index)


def asset_labels(bars: pd.DataFrame, horizons=HORIZONS, dividends: np.ndarray | None = None,
                 invalid_from: np.ndarray | None = None) -> pd.DataFrame:
    return pd.DataFrame(forward_returns(bars, horizons, dividends, invalid_from), index=bars.index)


#: Label whose NaN'd periods are counted per split issue (the headline TR label).
SPLIT_COUNT_STEM: str = "FWD_TR_RET"


def split_flags(bars: pd.DataFrame, issues: list[SplitIssue]) -> np.ndarray:
    """Bool per bar: True on the ex-date bar of each flagged split (``invalid_from``)."""
    out = np.zeros(len(bars), dtype=bool)
    idx = pd.DatetimeIndex(bars.index).normalize()
    for issue in issues:
        i = int(idx.searchsorted(pd.Timestamp(issue.ex_date), side="left"))
        if i < len(idx):
            out[i] = True
    return out


def label_inputs(bars: pd.DataFrame, actions: pd.DataFrame, ticker: str = ""
                 ) -> tuple[np.ndarray, np.ndarray, list[SplitIssue]]:
    """(dividends per bar, ``invalid_from`` split flags, the split issues) for one asset."""
    issues = detect_unadjusted_splits(bars, actions, ticker=ticker)
    return dividends_on_bars(bars.index, actions), split_flags(bars, issues), issues


def nan_periods(bars: pd.DataFrame, horizons, keep: np.ndarray, issue: SplitIssue,
                dividends: np.ndarray | None = None) -> int:
    """(bar, horizon) periods of ``SPLIT_COUNT_STEM`` inside ``keep`` that ``issue``'s flag NaNs."""
    flag = split_flags(bars, [issue])
    base = forward_returns(bars, horizons, dividends)
    flagged = forward_returns(bars, horizons, dividends, flag)
    return int(sum((np.isfinite(base[f"{SPLIT_COUNT_STEM}_{h}"])
                    & ~np.isfinite(flagged[f"{SPLIT_COUNT_STEM}_{h}"]))[keep].sum()
                   for h in horizons))


def _stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    out = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    out.index = out.index.set_names(["date", "ticker"])
    return out


def point_in_time_membership(membership: pd.DataFrame, active: set[str] | list[str],
                             complete_from: date = COMPLETE_FROM) -> pd.DataFrame:
    """``membership`` (dates × tickers, bool) with dates before ``complete_from``
    replaced by "active today" (the fallback where the change chain is unsourced)."""
    out = membership.astype(bool).copy()
    early = out.index < pd.Timestamp(complete_from)
    if early.any():
        act = np.array([t in set(active) for t in out.columns], dtype=bool)
        out.loc[early, :] = np.broadcast_to(act, (int(early.sum()), act.size))
    return out


def universe_coverage(membership: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """
    Per date: ``members`` (True cells of ``membership``, dates × every member
    ticker, including names with no prices), ``covered`` (members whose cell in
    ``values`` is finite; a ticker missing from ``values`` is not covered) and
    ``coverage`` = covered / members (NaN on a date without members).
    """
    m = membership.to_numpy(dtype=bool).reshape(len(membership), membership.shape[1])
    v = values.reindex(index=membership.index, columns=membership.columns).to_numpy(
        dtype=np.float64).reshape(m.shape)
    members = m.sum(axis=1)
    covered = (m & np.isfinite(v)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(members > 0, covered / np.maximum(members, 1), np.nan)
    return pd.DataFrame({"members": members, "covered": covered, "coverage": cov}, index=membership.index)


def build_dataset(
    assets: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    horizons=HORIZONS,
    session=None,
    universe: str = "index",
    index_code: str = "NIFTY50",
    complete_from: date = COMPLETE_FROM,
    variants: bool = False,
    timings: dict[str, float] | None = None,
    label_info: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (features, labels), both indexed by (date, ticker) and aligned row for row.

    ``assets`` defaults to every active stock (``universe="active"``) or every
    member of ``index_code`` in [start, end] (``universe="index"``, which also
    adds ``IN_UNIVERSE``; see the module docstring). Assets with fewer than
    ``MIN_BARS`` bars (including none) are skipped (logged). ``variants``,
    ``timings``: see ``asset_features``. ``label_info``, if given, receives one
    entry per evaluated ticker: ``dividend_rows`` (dividend rows with an ex-date
    in the output window) and ``splits`` (``(SplitIssue, periods NaN'd)`` pairs,
    counted on ``SPLIT_COUNT_STEM`` over all horizons in the output window).
    """
    from investbot.evaluation.loader import (
        active_tickers,
        load_actions,
        load_daily_history,
        membership_for,
        universe_tickers,
    )

    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}, not {universe!r}")

    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    feats: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.DataFrame] = {}
    try:
        if universe == "index":
            active = set(active_tickers(session))
            tickers = (list(assets) if assets is not None
                       else universe_tickers(session, index_code, start, end))
        else:
            tickers = list(assets) if assets is not None else active_tickers(session)
        for ticker in tickers:
            bars = load_daily_history(session, ticker, end)
            if len(bars) < MIN_BARS:
                logger.info("evaluation: skipping %s (%d bars)", ticker, len(bars))
                continue
            f = asset_features(bars, variants=variants, timings=timings)
            if universe == "index":
                member = point_in_time_membership(
                    membership_for(session, index_code, [ticker], bars.index), active,
                    complete_from)
                f[UNIVERSE_COLUMN] = member[ticker].to_numpy(dtype=bool)
            actions = load_actions(session, ticker, end)
            dividends, invalid, issues = label_inputs(bars, actions, ticker)
            y = asset_labels(bars, horizons, dividends, invalid)
            keep = (bars.index >= pd.Timestamp(start) if start is not None
                    else np.ones(len(bars), dtype=bool))
            if start is not None:
                f, y = f[keep], y[keep]
            if label_info is not None:
                lo = bars.index[keep].min() if keep.any() else None
                div_dates = actions.index[actions["action_type"] == "dividend"]
                label_info[ticker] = {
                    "dividend_rows": (int(((div_dates >= lo) & (div_dates <= bars.index[-1]))
                                          .sum()) if lo is not None else 0),
                    "splits": [(i, nan_periods(bars, horizons, keep, i, dividends))
                               for i in issues],
                }
            feats[ticker] = f
            labels[ticker] = y
    finally:
        if own:
            session.close()
    return _stack(feats), _stack(labels)


def build_panel(assets: list[str] | None = None, start: date | None = None,
                end: date | None = None, **kw) -> pd.DataFrame:
    """Feature panel indexed by (date, ticker). Labels: see ``build_dataset``."""
    return build_dataset(assets, start, end, **kw)[0]


def signal_columns(features: pd.DataFrame) -> list[str]:
    """Columns to evaluate as signals: scores, SIG_*, VAR_*, baselines (not REGIME*, IN_UNIVERSE)."""
    return [c for c in features.columns if c not in CONDITIONING_COLUMNS]
