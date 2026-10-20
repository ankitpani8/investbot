"""
Regime study: does the real-time regime label predict the ex-post regime?

``REGIME`` and ``REGIME_CONFIDENCE`` (``scoring/technical/regime.py``) will
drive the Black-Litterman <-> risk-parity blend and the PID gain scheduling.
This module measures, out of sample, whether bars labelled *trend* are
followed by more persistent price paths than bars labelled *range*, whether
*chaos* is followed by volatility expansion and deeper drawdowns, and whether
a higher confidence means the label is more often right. It is a study only:
nothing here changes a threshold, span or weight.

Pure compute (no DB I/O). ``scripts/regime_study.py`` loads one asset at a
time and writes the report.

Pre-registered constants (fixed before any run; do not add or search over more)
    REGIME_HORIZONS       (20, 60) bars ahead
    EX_POST_HURST_WINDOW  regime.REGIME_WINDOW (100): the ex-post Hurst uses the
                          same DFA window as the real-time HURST_100
    CONFIDENCE_SPANS      (1, 3, 5, 10, 20); span 1 = REGIME_CONFIDENCE_RAW,
                          span 5 = production REGIME_CONFIDENCE
    NULL_SEED             20261018
    MIN_GROUP_NAMES       3 names per label and date for a conditional mean

Pre-registered hypotheses and expected signs (written before the first run)
    H1 persistence   trend - range > 0 for FWD_ER_H, FWD_AC1_H, FWD_VR5_H
                     (H = 20, 60) and FWD_HURST_100.
    H2 vol expansion chaos - (trend U range) > 0 for FWD_VOLX_H.
    H3 drawdown      chaos - (trend U range) < 0 for FWD_MDD_H (deeper).
    H4 rank          daily Spearman IC > 0 of REGIME_SCORE_TREND and HURST_100
                     vs FWD_ER_H and FWD_HURST_100; IC > 0 of REGIME_SCORE_CHAOS
                     and VOL_RATIO vs FWD_VOLX_H; IC < 0 of the same two vs
                     FWD_MDD_H.
    H5 calibration   daily IC > 0 of REGIME_CONFIDENCE_S<span> vs HIT_H, for
                     every span (higher confidence = the label is more right).

    Verdict rule (``verdict``): *supported* = the expected sign, BH q <
    ``evaluate.PROMISING_Q`` (0.10) over every test of H1-H5, and the expected
    sign in >= ``evaluate.PROMISING_SIGN_SHARE`` (70%) of the walk-forward test
    blocks; *wrong sign* = the opposite sign with q < 0.10; otherwise *not
    supported*. p-values are two-sided (normal approximation of the t-stat).

Pre-registered null-excess family H6 (task 330, written 2026-10-19 before any H6 run
on real data; H1-H5 above and their BH family are unchanged)
    Why: VOL_RATIO (and so REGIME_SCORE_CHAOS) and FWD_VOLX share the REALVOL_250
    denominator, and REALVOL_20 sits inside REALVOL_250's window, so the H2/H4
    volatility tests have a large mechanical part that the shuffled null shows
    too. H6 asks whether the real association exceeds that mechanical part.
    NULL_EXCESS_SEEDS     (20261018, 20261019, 20261020, 20261021, 20261022);
                          the first is NULL_SEED, so the existing null panel is
                          reused. No other seeds.
    Tests (6, all with expected sign +1: real exceeds the mechanical null)
        (i)   H2 chaos - rest          on FWD_VOLX_20, FWD_VOLX_60
        (ii)  H4 REGIME_SCORE_CHAOS    vs FWD_VOLX_20, FWD_VOLX_60
        (iii) H4 VOL_RATIO             vs FWD_VOLX_20, FWD_VOLX_60
        No MDD, no other targets.
    Statistic per test and OOS date d
        x_d = stat_real(d) - mean_k stat_null_k(d), where stat is the per-date
        paired group difference (``group_difference``, >= MIN_GROUP_NAMES names
        per label) for (i) and the per-date Spearman IC (``metrics.daily_ic``,
        >= ``metrics.MIN_NAMES`` names) for (ii)/(iii), computed on the real panel
        and on the panel of each seed k. The null mean is over the seeds whose
        value on d is finite; a date with no finite null value (or no finite
        real value) is skipped. Summary: ``_summary`` (time mean, Newey-West t
        with lag H, two-sided normal p, as H1-H4) and ``block_sign_share``.
    BH over the 6 H6 tests as their own family; the same ``verdict`` rule
    (q < 0.10 and the expected sign in >= 70% of the test blocks).

Pre-registered robust inference (task 350, written 2026-10-20 before any robust run on
DB data; H1-H6, their targets, thresholds, spans, seeds and the verdict rule are unchanged)
    Why: every verdict above rests on a Newey-West t at the fixed lag H. REALVOL_250
    keeps the H4/H6 per-date statistics persistent for ~250 bars, and task 330 found
    |t| ~ 4 at a true mean of 0 on synthetic iid panels of ~480 OOS dates, so the lag-H
    t over-rejects there. Two robust checks of task 340 are added next to the existing
    columns (``robust_summary``), on the **same per-date series and dates** as the
    existing summary of each test. Constants are imported from ``stats.py``, not
    redefined: ``ROBUST_BOOT_N`` = 2000, ``ROBUST_BOOT_SEED`` = 20261020,
    ``ROBUST_MAX_LAG_FRAC`` = 0.25, ``ROBUST_RHO_CLIP`` = 0.97.
        H          the lag each test uses today: ``_summary(x, h)`` and
                   ``ic_summary(ic, h + 1)`` both use lag h (H = 20, 60, or 100 for
                   FWD_HURST_100).
        acf1       ``stats.acf`` of the finite values at lag 1 (descriptive).
        lag_auto   max(H, ``andrews_bartlett_lag(x, min_lag=H)``). The outer max matters:
                   the cap floor(0.25 n) of ``andrews_bartlett_lag`` beats ``min_lag``, and
                   for the H6 chaos - rest tests (n ~ 45, H = 20) the cap is 11, so without
                   it the lag would be less conservative than the current one.
        t_auto     ``newey_west_tstat(x, lag_auto)``; p_auto = ``normal_pvalue(t_auto)``.
        boot_block max(H, lag_auto + 1, round(n ** (1/3))).
        p_boot     ``block_bootstrap_mean_pvalue(x, boot_block)``: centred series,
                   ``ROBUST_BOOT_N`` draws, seed ``ROBUST_BOOT_SEED``; NaN below
                   ``performance.BOOTSTRAP_MIN_PERIODS`` (20) finite values.
    q-values: ``q_auto`` / ``q_boot`` are BH of p_auto / p_boot within the **existing
    families**: H1-H5 is one family of 35 tests, H6 its own family of 6. ``verdict_auto``
    / ``verdict_boot`` are the unchanged ``verdict`` rule with q replaced, so the block
    share and the sign are the same as for ``verdict``. Bootstrap resolution: p_boot >=
    1/2001, so the smallest attainable q of the top test is 35/2001 ~ 0.017 (H1-H5) and
    6/2001 ~ 0.003 (H6). These are checks next to the pre-registered verdict, which is
    not replaced; the report lists every test whose verdict changes.
    Size check (run once, reported, not tuned against): seeded synthetic per-date series
    of ``SIZE_CHECK_N`` = 690 values with true mean 0, horizon ``SIZE_CHECK_HORIZON`` =
    20, ``SIZE_CHECK_RUNS`` = 200 seeds per DGP, run k of DGP d drawing from
    ``np.random.default_rng((SIZE_CHECK_SEED, d, k))``, ``SIZE_CHECK_SEED`` = 20261020:
        d = 1  AR(1), rho = ``SIZE_CHECK_AR_RHO`` = 0.99: z = standard_normal(n),
               x_0 = z_0 / sqrt(1 - rho^2), x_t = rho x_{t-1} + z_t.
        d = 2  overlap of a ``SIZE_CHECK_WINDOW`` = 250 window (mimics REALVOL_250):
               e = standard_normal(n + 249), then u = standard_normal(n);
               x_t = sqrt(250) * mean(e_t .. e_{t+249}) + u_t, so the rolling-mean part
               and the iid N(0, 1) noise have equal (unit) variance.
    Reported: the share of runs with two-sided p < ``SIZE_CHECK_LEVEL`` = 0.05 for the
    fixed lag H (``normal_pvalue(newey_west_tstat(x, H))``), p_auto and p_boot. If
    auto or boot still over-rejects badly, the report says so; no other method is added.

Features vs labels
    ``regime_conditioning`` returns features (bar t uses bars <= t only,
    tested for truncation invariance). ``forward_regime_targets`` returns
    labels: they use bars after t and must never be joined into a feature
    frame. The two are kept in separate frames, as in ``panel.build_dataset``.

Statistics (all on the harness's out-of-sample dates)
    Label base rates   share of bars per label and label flips per 1000 bars,
                       real vs the shuffled-returns null, per asset and pooled;
                       HURST_100 median and 5th-95th, real vs null.
    Conditional means  per date, the cross-sectional mean of a target within a
                       label (>= MIN_GROUP_NAMES names), then the paired per-date
                       difference; time mean with a Newey-West t, lag H (the
                       windows of consecutive dates overlap by H - 1 bars; one
                       lag more than the harness convention, conservative).
                       The same test on the null panel is shown as a sanity
                       check (not in BH).
    Rank association   ``metrics.daily_ic`` (>= ``metrics.MIN_NAMES`` names),
                       summarised with ``metrics.ic_summary`` at lag H. Also run on
                       the null panel as a sanity check (not in BH).
    Calibration        HIT_H: on trend/range bars (chaos excluded), the per-date
                       percentile rank of FWD_ER_H among those names scaled to
                       [-0.5, 0.5], times +1 for trend and -1 for range. Daily
                       IC of each confidence span vs HIT_H, mean HIT by
                       confidence quintile (``metrics.quantile_buckets``), and
                       churn: mean |dconfidence| per bar and crossings of 0.1
                       and 0.5 per 1000 bars.
    Consecutive bars for flips and churn are consecutive rows of the union
    trading calendar; a row where an asset has no bar breaks the pair.

Point-in-time universe
    The runner may set a feature cell to NaN where the asset was not an index
    member on that date (targets are never masked). Every statistic here
    counts only finite (member) feature cells, and a NaN row breaks a flip or
    churn pair, so an exit then re-entry is never counted as a label flip or a
    confidence crossing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from investbot.config.scoring_weights import REGIME_NAMES
from investbot.evaluation.evaluate import PROMISING_Q, PROMISING_SIGN_SHARE
from investbot.evaluation.labels import HORIZONS as HARNESS_HORIZONS
from investbot.evaluation.metrics import (
    MIN_NAMES,
    bucket_returns,
    daily_ic,
    ic_summary,
    quantile_buckets,
)
from investbot.evaluation.performance import BOOTSTRAP_MIN_PERIODS
from investbot.evaluation.stats import (
    ROBUST_BOOT_N,
    ROBUST_BOOT_SEED,
    ROBUST_MAX_LAG_FRAC,
    ROBUST_RHO_CLIP,
    acf,
    andrews_bartlett_lag,
    bh_adjust,
    block_bootstrap_mean_pvalue,
    newey_west_tstat,
    normal_pvalue,
)
from investbot.evaluation.walkforward import Split, oos_dates, walk_forward_splits
from investbot.scoring.technical import regime

REGIME_HORIZONS: tuple[int, ...] = (20, 60)
EX_POST_HURST_WINDOW: int = regime.REGIME_WINDOW
CONFIDENCE_SPANS: tuple[int, ...] = (1, 3, 5, 10, 20)
NULL_SEED: int = 20261018
MIN_GROUP_NAMES: int = 3
#: H6 (pre-registered): shuffled-null seeds; the first is NULL_SEED (the existing null panel).
NULL_EXCESS_SEEDS: tuple[int, ...] = (20261018, 20261019, 20261020, 20261021, 20261022)

# Pre-registered robust inference (task 350; see the module docstring). The bootstrap and
# Andrews constants are the task-340 ones from ``stats.py``, imported above, not redefined.
#: Columns ``robust_summary`` returns, appended after the existing columns of each test.
ROBUST_COLUMNS: tuple[str, ...] = ("acf1", "lag_auto", "t_auto", "p_auto", "boot_block", "p_boot")
#: Columns ``run_study(robust=True)`` appends to ``tests`` and ``h6`` after ``ROBUST_COLUMNS``.
ROBUST_VERDICT_COLUMNS: tuple[str, ...] = ("q_auto", "q_boot", "verdict_auto", "verdict_boot")
#: Size check of the three inference rules (run once, reported, not tuned against).
SIZE_CHECK_N: int = 690
SIZE_CHECK_HORIZON: int = 20
SIZE_CHECK_RUNS: int = 200
SIZE_CHECK_SEED: int = 20261020
SIZE_CHECK_AR_RHO: float = 0.99
SIZE_CHECK_WINDOW: int = 250
SIZE_CHECK_LEVEL: float = 0.05

#: Confidence levels whose crossings count as churn.
CROSSING_LEVELS: tuple[float, ...] = (0.1, 0.5)
#: Lag of the variance ratio (FWD_VR5).
VR_LAG: int = 5

RANGE, TREND, CHAOS = regime.REGIME_RANGE, regime.REGIME_TREND, regime.REGIME_CHAOS

CONDITIONING_COLUMNS: tuple[str, ...] = (
    "REGIME", "REGIME_CONFIDENCE", "REGIME_CONFIDENCE_RAW", "REGIME_SCORE_TREND",
    "REGIME_SCORE_CHAOS", f"HURST_{regime.REGIME_WINDOW}", f"LYAP_{regime.REGIME_WINDOW}",
    "VOL_RATIO", f"REALVOL_{regime.VOL_LONG}",
)
HURST_COL = f"HURST_{regime.REGIME_WINDOW}"
REALVOL_COL = f"REALVOL_{regime.VOL_LONG}"


def confidence_column(span: int) -> str:
    return f"REGIME_CONFIDENCE_S{span}"


def target_columns(horizons=REGIME_HORIZONS, hurst_window: int = EX_POST_HURST_WINDOW) -> list[str]:
    return ([f"FWD_{k}_{h}" for h in horizons for k in ("ER", "AC1", "VR5", "VOLX", "MDD")]
            + [f"FWD_HURST_{hurst_window}"])


# ──────────────────────────────────────────────────────────
# Features (truncation-invariant)
# ──────────────────────────────────────────────────────────

def regime_conditioning(bars: pd.DataFrame, spans=CONFIDENCE_SPANS) -> dict[str, np.ndarray]:
    """
    The regime columns the study conditions on, from one ``regime.compute_regime`` call.

    ``CONDITIONING_COLUMNS`` plus ``REGIME_CONFIDENCE_S<span>`` =
    ``regime.smooth_confidence(REGIME_CONFIDENCE_RAW, span)`` for every span
    (span 1 is the raw gap itself). Features: bar t uses bars <= t only.
    """
    reg = regime.compute_regime(bars)
    out = {c: reg[c] for c in CONDITIONING_COLUMNS}
    raw = reg["REGIME_CONFIDENCE_RAW"]
    for span in spans:
        out[confidence_column(span)] = regime.smooth_confidence(raw, span)
    return out


# ──────────────────────────────────────────────────────────
# Labels (use bars after t; never join into a feature frame)
# ──────────────────────────────────────────────────────────

def _windows(x: np.ndarray, length: int) -> np.ndarray:
    """Rows x[t .. t+length-1] for t = 0 .. n-length (empty if too short)."""
    if len(x) < length:
        return np.empty((0, length))
    return sliding_window_view(x, length)


def _efficiency_ratio(r: np.ndarray) -> np.ndarray:
    path = np.abs(r).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(path > 0, np.abs(r.sum(axis=1)) / path, np.nan)


def _lag1_autocorr(r: np.ndarray) -> np.ndarray:
    """Sample ACF at lag 1: sum (r_i - m)(r_{i+1} - m) / sum (r_i - m)^2 (as statsmodels ``acf``)."""
    d = r - r.mean(axis=1, keepdims=True)
    den = (d * d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (d[:, 1:] * d[:, :-1]).sum(axis=1) / den, np.nan)


def _variance_ratio(r: np.ndarray, q: int = VR_LAG) -> np.ndarray:
    """var(overlapping q-bar sums) / (q * var(r)), both sample variances (ddof 1); NaN if < 2 sums."""
    n_rows, h = r.shape
    if h - q + 1 < 2:
        return np.full(n_rows, np.nan)
    c = np.concatenate([np.zeros((n_rows, 1)), np.cumsum(r, axis=1)], axis=1)
    sums = c[:, q:] - c[:, :-q]
    v1 = r.var(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v1 > 0, sums.var(axis=1, ddof=1) / (q * v1), np.nan)


def _max_drawdown(c: np.ndarray) -> np.ndarray:
    """Worst close / running max - 1 along each row (<= 0)."""
    return (c / np.maximum.accumulate(c, axis=1) - 1.0).min(axis=1)


def forward_regime_targets(
    close: np.ndarray,
    horizons=REGIME_HORIZONS,
    hurst_window: int = EX_POST_HURST_WINDOW,
    realvol: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Ex-post regime targets. **Labels, not features**: the value at bar t uses
    closes t .. t+H, so these arrays must never be joined into a feature frame.

    With r = daily log returns of ``close``, for every H in ``horizons``:

        FWD_ER_<H>     |sum r[t+1..t+H]| / sum |r[t+1..t+H]|  (efficiency ratio, 0..1)
        FWD_AC1_<H>    lag-1 autocorrelation of r[t+1..t+H] (``_lag1_autocorr``)
        FWD_VR5_<H>    var(overlapping 5-bar sums) / (5 var(r)) inside r[t+1..t+H]
                       (NaN for H < 6: fewer than two 5-bar sums)
        FWD_VOLX_<H>   std(r[t+1..t+H], ddof 1) / realvol[t]; realvol is
                       REALVOL_250 (known at t, the only input before t). All NaN
                       when ``realvol`` is None.
        FWD_MDD_<H>    max drawdown of close over [t, t+H] (<= 0)

    and ``FWD_HURST_<W>`` = ``regime.rolling_hurst(log close, W)`` read at t+W,
    the DFA Hurst of exactly the returns (t, t+W].

    NaN where the window runs past the data or contains a NaN (or a
    non-positive close). Pure, vectorised, no dropna.
    """
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_c = np.where(c > 0, np.log(c), np.nan)
    rv = (np.full(n, np.nan) if realvol is None
          else np.asarray(realvol, dtype=np.float64))
    if len(rv) != n:
        raise ValueError("realvol must have the same length as close")

    out: dict[str, np.ndarray] = {}
    for h in horizons:
        if h < 2:
            raise ValueError("horizons must be >= 2")
        lw = _windows(log_c, h + 1)                     # rows: log close t .. t+h
        m = len(lw)
        ok = np.isfinite(lw).all(axis=1)
        r = np.diff(lw, axis=1)
        cols = {
            "ER": _efficiency_ratio(r),
            "AC1": _lag1_autocorr(r),
            "VR5": _variance_ratio(r),
            "VOLX": np.full(m, np.nan),
            "MDD": _max_drawdown(np.exp(lw)),
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            rv_t = rv[:m]
            cols["VOLX"] = np.where(rv_t > 0, r.std(axis=1, ddof=1) / rv_t, np.nan)
        for k, v in cols.items():
            a = np.full(n, np.nan)
            a[:m] = np.where(ok, v, np.nan)
            out[f"FWD_{k}_{h}"] = a

    hurst = regime.rolling_hurst(log_c, hurst_window)
    fh = np.full(n, np.nan)
    if n > hurst_window:
        fh[: n - hurst_window] = hurst[hurst_window:]
    out[f"FWD_HURST_{hurst_window}"] = fh
    return out


# ──────────────────────────────────────────────────────────
# Shuffled-returns null
# ──────────────────────────────────────────────────────────

def shuffled_null_bars(bars: pd.DataFrame, seed: int = NULL_SEED) -> pd.DataFrame:
    """
    One asset's bars with the bar order permuted (all serial dependence removed).

    Bar 0 is kept. For bars 1..n-1 the tuples (log return, high/close,
    low/close, open/close, volume) are permuted jointly with
    ``np.random.default_rng(seed)``, then OHLC is rebuilt from the first close:
    close = close[0] * exp(cumsum of the permuted returns), and each bar keeps
    its own shape relative to its close, so low <= open, close <= high still
    holds. Returns, bar shapes and volumes keep their multiset; autocorrelation
    and volatility clustering are gone. Same index as ``bars``.

    The same seed gives the same permutation for every asset of the same
    length, so assets with a common calendar are shuffled by a common date
    permutation (cross-sectional co-movement survives, serial dependence does
    not). A non-finite return makes that rebuilt bar NaN and is treated as 0
    for the bars after it.
    """
    o, h, l, c = (bars[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    n = len(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_c = np.where(c > 0, np.log(c), np.nan)
        ratios = np.stack([h / c, l / c, o / c], axis=1)
    r = np.r_[np.nan, np.diff(log_c)]
    vol = (bars["volume"].to_numpy(dtype=np.float64) if "volume" in bars.columns
           else np.full(n, np.nan))
    perm = np.r_[0, 1 + np.random.default_rng(seed).permutation(max(n - 1, 0))] if n else np.arange(0)
    rp, ratp, volp = r[perm], ratios[perm], vol[perm]
    steps = np.where(np.isfinite(rp), rp, 0.0)
    steps[0] = 0.0
    new_c = c[0] * np.exp(np.cumsum(steps)) if n else np.empty(0)
    bad = ~np.isfinite(rp)
    bad[0] = not np.isfinite(c[0]) if n else False
    new_c = np.where(bad, np.nan, new_c)
    data = {
        "open": new_c * ratp[:, 2],
        "high": new_c * ratp[:, 0],
        "low": new_c * ratp[:, 1],
        "close": new_c,
        "volume": volp,
    }
    return pd.DataFrame(data, index=bars.index)


# ──────────────────────────────────────────────────────────
# Per-asset study columns
# ──────────────────────────────────────────────────────────

STUDY_FEATURES: tuple[str, ...] = (
    "REGIME", "REGIME_SCORE_TREND", "REGIME_SCORE_CHAOS", HURST_COL, "VOL_RATIO",
    *(confidence_column(s) for s in CONFIDENCE_SPANS),
)


def asset_study_frames(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (features, targets) for one asset, only the float columns the study reads.

    Features are ``STUDY_FEATURES`` from ``regime_conditioning``; targets are
    ``forward_regime_targets`` normalised by that bar's REALVOL_250. Separate
    frames: targets are labels.
    """
    cond = regime_conditioning(bars)
    feats = pd.DataFrame({k: cond[k] for k in STUDY_FEATURES}, index=bars.index)
    tg = forward_regime_targets(bars["close"].to_numpy(dtype=np.float64),
                                realvol=cond[REALVOL_COL])
    return feats, pd.DataFrame(tg, index=bars.index)


def to_wide(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """{ticker: frame(date × column)} -> {column: frame(date × ticker)} on the union calendar."""
    if not frames:
        return {}
    tickers = sorted(frames)
    cols = list(next(iter(frames.values())).columns)
    return {c: pd.concat({t: frames[t][c] for t in tickers}, axis=1).sort_index() for c in cols}


# ──────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────

def study_calendar(regime_wide: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.DatetimeIndex:
    """Dates from the first one where REGIME is known for >= ``min_names`` names."""
    counts = regime_wide.notna().sum(axis=1).to_numpy()
    usable = counts >= min_names
    if not usable.any():
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(regime_wide.index[int(np.argmax(usable)):])


def study_splits(regime_wide: pd.DataFrame) -> list[Split]:
    """Walk-forward splits with the harness settings (embargo = its longest horizon)."""
    return walk_forward_splits(study_calendar(regime_wide), embargo=max(HARNESS_HORIZONS))


def _flip_counts(labels: pd.DataFrame, dates) -> tuple[np.ndarray, np.ndarray]:
    """Per column: (label changes, valid consecutive pairs) at rows in ``dates``."""
    prev = labels.shift(1)
    pair = labels.notna() & prev.notna()
    flip = pair & (labels != prev)
    rows = labels.index.isin(dates)
    return flip[rows].sum(axis=0).to_numpy(), pair[rows].sum(axis=0).to_numpy()


def label_base_rates(regime_wide: pd.DataFrame, dates) -> pd.DataFrame:
    """Per ticker and pooled ("ALL"): n bars, share of each label, flips per 1000 bars."""
    sub = regime_wide.loc[regime_wide.index.isin(dates)]
    flips, pairs = _flip_counts(regime_wide, dates)
    arr = sub.to_numpy(dtype=np.float64)
    rows = []
    for j, t in enumerate([*sub.columns, "ALL"]):
        a = arr[:, j] if t != "ALL" else arr.ravel()
        a = a[np.isfinite(a)]
        f = flips[j] if t != "ALL" else flips.sum()
        p = pairs[j] if t != "ALL" else pairs.sum()
        row = {"ticker": t, "n_bars": int(a.size)}
        for code, name in REGIME_NAMES.items():
            row[f"share_{name}"] = float((a == code).mean()) if a.size else math.nan
        row["flips_per_1000"] = 1000.0 * f / p if p else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def distribution(wide: pd.DataFrame, dates) -> dict[str, float]:
    a = wide.loc[wide.index.isin(dates)].to_numpy(dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0, "mean": math.nan, "p05": math.nan, "median": math.nan, "p95": math.nan}
    p05, med, p95 = np.percentile(a, [5, 50, 95])
    return {"n": int(a.size), "mean": float(a.mean()), "p05": float(p05),
            "median": float(med), "p95": float(p95)}


def group_mean(labels: pd.DataFrame, target: pd.DataFrame, codes,
               min_names: int = MIN_GROUP_NAMES) -> pd.Series:
    """Per date, the mean target over names whose label is in ``codes`` (NaN if < min_names)."""
    lab = labels.reindex_like(target).to_numpy(dtype=np.float64)
    y = target.to_numpy(dtype=np.float64)
    m = np.isin(lab, list(codes)) & np.isfinite(y)
    cnt = m.sum(axis=1)
    tot = np.where(m, y, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt >= min_names, tot / np.maximum(cnt, 1), np.nan)
    return pd.Series(mean, index=target.index)


def group_difference(labels: pd.DataFrame, target: pd.DataFrame, a, b,
                     min_names: int = MIN_GROUP_NAMES) -> pd.Series:
    """Paired per-date difference mean(target | label in a) - mean(target | label in b)."""
    return group_mean(labels, target, a, min_names) - group_mean(labels, target, b, min_names)


def _summary(x: pd.Series, lag: int) -> dict[str, float]:
    a = x.to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    t = newey_west_tstat(a, lag)
    return {"n_days": int(a.size), "mean": float(a.mean()) if a.size else math.nan,
            "t_nw": t, "p": normal_pvalue(t)}


def block_sign_share(x: pd.Series, splits: list[Split], expected_sign: int) -> float:
    """Share of test blocks (with data) whose mean of ``x`` has ``expected_sign``."""
    means = []
    for sp in splits:
        v = x.reindex(sp.test).to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            means.append(np.sign(v.mean()))
    return float(np.mean(np.array(means) == expected_sign)) if means else math.nan


@dataclass(frozen=True)
class Hypothesis:
    section: str          # "conditional" | "rank" | "calibration"
    group: str            # H1..H5
    name: str             # e.g. "trend − range" or a signal name
    target: str           # target column
    horizon: int          # bars ahead (the NW lag)
    expected_sign: int


#: Conditional-mean tests: (group, comparison, target stem, expected sign).
CONDITIONAL_TESTS: tuple[tuple[str, str, str, int], ...] = (
    ("H1", "trend − range", "ER", +1),
    ("H1", "trend − range", "AC1", +1),
    ("H1", "trend − range", "VR5", +1),
    ("H2", "chaos − rest", "VOLX", +1),
    ("H3", "chaos − rest", "MDD", -1),
)
COMPARISONS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "trend − range": ((TREND,), (RANGE,)),
    "chaos − rest": ((CHAOS,), (TREND, RANGE)),
}
#: Rank-association tests: (signal, target stem, expected sign).
RANK_TESTS: tuple[tuple[str, str, int], ...] = (
    ("REGIME_SCORE_TREND", "ER", +1), ("REGIME_SCORE_TREND", "HURST", +1),
    (HURST_COL, "ER", +1), (HURST_COL, "HURST", +1),
    ("REGIME_SCORE_CHAOS", "VOLX", +1), ("REGIME_SCORE_CHAOS", "MDD", -1),
    ("VOL_RATIO", "VOLX", +1), ("VOL_RATIO", "MDD", -1),
)
#: H6 null-excess tests (pre-registered): (section, comparison or signal, target stem), each
#: at every REGIME_HORIZONS horizon, expected sign +1.
NULL_EXCESS_TESTS: tuple[tuple[str, str, str], ...] = (
    ("conditional", "chaos − rest", "VOLX"),
    ("rank", "REGIME_SCORE_CHAOS", "VOLX"),
    ("rank", "VOL_RATIO", "VOLX"),
)
NULL_EXCESS_SIGN: int = +1
#: The only columns H6 reads (the runner keeps just these for the extra seeds).
NULL_EXCESS_FEATURES: tuple[str, ...] = ("REGIME", "REGIME_SCORE_CHAOS", "VOL_RATIO")
NULL_EXCESS_TARGETS: tuple[str, ...] = tuple(f"FWD_VOLX_{h}" for h in REGIME_HORIZONS)


def _target_keys(stem: str, horizons, hurst_window: int) -> list[tuple[str, int]]:
    if stem == "HURST":
        return [(f"FWD_HURST_{hurst_window}", hurst_window)]
    return [(f"FWD_{stem}_{h}", h) for h in horizons]


def hit_matrix(labels: pd.DataFrame, fwd_er: pd.DataFrame) -> pd.DataFrame:
    """
    HIT: per-date percentile rank of FWD_ER among trend/range names, in [-0.5, 0.5],
    times +1 for trend and -1 for range. NaN for chaos, missing labels or < 2 names.
    """
    lab = labels.reindex_like(fwd_er)
    tr = lab.isin([TREND, RANGE]) & fwd_er.notna()
    y = fwd_er.where(tr)
    n = tr.sum(axis=1).to_numpy(dtype=np.float64)[:, None]
    rank = y.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(n >= 2, (rank - 1.0) / (n - 1.0) - 0.5, np.nan)
    sign = np.where(lab.to_numpy(dtype=np.float64) == TREND, 1.0, -1.0)
    return pd.DataFrame(pct * sign, index=fwd_er.index, columns=fwd_er.columns)


def confidence_churn(conf: pd.DataFrame, dates, levels=CROSSING_LEVELS) -> dict[str, float]:
    """Mean |Δconfidence| per bar and crossings of each level per 1000 bars, pooled."""
    prev = conf.shift(1)
    rows = conf.index.isin(dates)
    c = conf.to_numpy(dtype=np.float64)[rows]
    p = prev.to_numpy(dtype=np.float64)[rows]
    pair = np.isfinite(c) & np.isfinite(p)
    n = int(pair.sum())
    out = {"n_pairs": n, "mean_abs_change": float(np.abs(c - p)[pair].mean()) if n else math.nan}
    for lv in levels:
        cross = pair & ((c >= lv) != (p >= lv))
        out[f"cross_{lv:g}_per_1000"] = 1000.0 * cross.sum() / n if n else math.nan
    return out


def verdict(row: dict | pd.Series, q_max: float = PROMISING_Q,
            share_min: float = PROMISING_SIGN_SHARE) -> str:
    """'supported' / 'wrong sign' / 'not supported' (see the module docstring)."""
    mean, q, share, sign = row["mean"], row["q"], row["block_share"], row["expected_sign"]
    if not (np.isfinite(mean) and np.isfinite(q)) or mean == 0:
        return "not supported"
    if np.sign(mean) == sign:
        if q < q_max and np.isfinite(share) and share >= share_min:
            return "supported"
        return "not supported"
    return "wrong sign" if q < q_max else "not supported"


@dataclass
class StudyResult:
    splits: list[Split]
    oos: pd.DatetimeIndex
    base_rates_real: pd.DataFrame
    base_rates_null: pd.DataFrame
    hurst_real: dict[str, float]
    hurst_null: dict[str, float]
    tests: pd.DataFrame                      # every BH-tested row (H1-H5)
    group_means: pd.DataFrame                # time-mean of each label's per-date mean
    null_conditional: pd.DataFrame           # the H1-H3 tests on the null panel
    null_rank: pd.DataFrame                  # the H4 tests on the null panel
    hit_quintiles: pd.DataFrame              # span, horizon, Q1..Q5 mean HIT
    churn: pd.DataFrame                      # span: churn stats
    notes: list[str] = field(default_factory=list)
    h6: pd.DataFrame = field(default_factory=pd.DataFrame)   # H6 null excess (own BH family)


def conditional_tests(features: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame],
                      dates, splits: list[Split], horizons=REGIME_HORIZONS,
                      hurst_window: int = EX_POST_HURST_WINDOW,
                      min_names: int = MIN_GROUP_NAMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tests, per-label means) of the pre-registered conditional-mean hypotheses H1-H3."""
    labels = features["REGIME"].loc[features["REGIME"].index.isin(dates)]
    tests = [(g, c, s, e) for g, c, s, e in CONDITIONAL_TESTS]
    tests.insert(3, ("H1", "trend − range", "HURST", +1))
    rows, means = [], []
    for group, comp, stem, sign in tests:
        a, b = COMPARISONS[comp]
        for col, h in _target_keys(stem, horizons, hurst_window):
            y = targets[col].reindex(index=labels.index, columns=labels.columns)
            d = group_difference(labels, y, a, b, min_names)
            rows.append({"section": "conditional", "group": group, "name": comp, "target": col,
                         "horizon": h, "expected_sign": sign, **_summary(d, h),
                         "block_share": block_sign_share(d, splits, sign)})
            m = {"target": col, "horizon": h}
            for code, name in REGIME_NAMES.items():
                g = group_mean(labels, y, (code,), min_names).to_numpy()
                g = g[np.isfinite(g)]
                m[name] = float(g.mean()) if g.size else math.nan
                m[f"{name}_days"] = int(g.size)
            means.append(m)
    return pd.DataFrame(rows), pd.DataFrame(means)


def rank_tests(features, targets, dates, splits, horizons=REGIME_HORIZONS,
               hurst_window: int = EX_POST_HURST_WINDOW) -> pd.DataFrame:
    """Daily Spearman IC (``metrics.daily_ic``, NW lag H) of the pre-registered H4 pairs."""
    rows = []
    for sig, stem, sign in RANK_TESTS:
        S = features[sig].loc[features[sig].index.isin(dates)]
        for col, h in _target_keys(stem, horizons, hurst_window):
            Y = targets[col].reindex(index=S.index, columns=S.columns)
            ic = daily_ic(S, Y, MIN_NAMES)
            s = ic_summary(ic, h + 1)                 # ic_summary uses lag horizon - 1
            rows.append({"section": "rank", "group": "H4", "name": sig, "target": col,
                         "horizon": h, "expected_sign": sign, "n_days": s["n_days"],
                         "mean": s["ic_mean"], "t_nw": s["ic_t_nw"], "p": s["ic_p"],
                         "block_share": block_sign_share(ic, splits, sign)})
    return pd.DataFrame(rows)


def calibration_tests(features, targets, dates, splits, horizons=REGIME_HORIZONS,
                      spans=CONFIDENCE_SPANS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(H5 IC tests, mean HIT by confidence quintile, churn) for every span."""
    labels = features["REGIME"].loc[features["REGIME"].index.isin(dates)]
    tr = labels.isin([TREND, RANGE])
    rows, quint, churn = [], [], []
    for h in horizons:
        er = targets[f"FWD_ER_{h}"].reindex(index=labels.index, columns=labels.columns)
        hit = hit_matrix(labels, er)
        for span in spans:
            conf = features[confidence_column(span)].loc[labels.index].where(tr)
            ic = daily_ic(conf, hit, MIN_NAMES)
            s = ic_summary(ic, h + 1)
            rows.append({"section": "calibration", "group": "H5",
                         "name": confidence_column(span), "target": f"HIT_{h}", "horizon": h,
                         "span": span, "expected_sign": +1, "n_days": s["n_days"],
                         "mean": s["ic_mean"], "t_nw": s["ic_t_nw"], "p": s["ic_p"],
                         "block_share": block_sign_share(ic, splits, +1)})
            b = quantile_buckets(conf, hit, MIN_NAMES)
            qm = bucket_returns(b, hit).mean(axis=0)
            quint.append({"span": span, "horizon": h,
                          **{f"Q{k}": float(qm[k]) for k in qm.index}})
    for span in spans:
        churn.append({"span": span,
                      **confidence_churn(features[confidence_column(span)], dates)})
    return pd.DataFrame(rows), pd.DataFrame(quint), pd.DataFrame(churn)


def _per_date_stat(section: str, name: str, col: str, features: dict[str, pd.DataFrame],
                   targets: dict[str, pd.DataFrame], dates) -> pd.Series:
    """The per-date statistic of one H6 test on one panel, exactly as in H2 / H4:
    ``group_difference`` (conditional) or ``metrics.daily_ic`` (rank), on ``dates``."""
    if section == "conditional":
        labels = features["REGIME"].loc[features["REGIME"].index.isin(dates)]
        y = targets[col].reindex(index=labels.index, columns=labels.columns)
        a, b = COMPARISONS[name]
        return group_difference(labels, y, a, b)
    S = features[name].loc[features[name].index.isin(dates)]
    Y = targets[col].reindex(index=S.index, columns=S.columns)
    return daily_ic(S, Y, MIN_NAMES)


def null_excess_series(features: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame],
                       null_panels: list[tuple[dict, dict]], dates,
                       horizons=REGIME_HORIZONS) -> list[tuple[dict, pd.Series, pd.Series]]:
    """
    Per H6 test: (keys, x, null mean) as per-date series on the real panel's ``dates``.

    null mean(d) = mean over the panels of ``null_panels`` whose statistic on d is
    finite (NaN if none); x(d) = real(d) - null mean(d), NaN where either is NaN,
    so such a date is skipped by every summary.
    """
    out = []
    for section, name, stem in NULL_EXCESS_TESTS:
        for col, h in _target_keys(stem, horizons, EX_POST_HURST_WINDOW):
            real = _per_date_stat(section, name, col, features, targets, dates)
            nulls = np.column_stack(
                [_per_date_stat(section, name, col, nf, nt, dates)
                 .reindex(real.index).to_numpy(dtype=np.float64)
                 for nf, nt in null_panels]) if null_panels else np.full((len(real), 1), np.nan)
            ok = np.isfinite(nulls)
            cnt = ok.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                nm = np.where(cnt > 0, np.where(ok, nulls, 0.0).sum(axis=1) / np.maximum(cnt, 1),
                              np.nan)
            null_mean = pd.Series(nm, index=real.index)
            keys = {"section": section, "group": "H6", "name": name, "target": col,
                    "horizon": h, "expected_sign": NULL_EXCESS_SIGN}
            out.append((keys, real - null_mean, null_mean))
    return out


def null_excess_tests(features: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame],
                      null_panels: list[tuple[dict, dict]], dates,
                      splits: list[Split]) -> pd.DataFrame:
    """
    The pre-registered H6 family: real minus the seed-averaged shuffled null, per date.

    ``null_panels`` = [(null features, null targets), ...], one per seed (the first
    is the NULL_SEED panel). Same columns as ``StudyResult.tests``, plus
    ``null_mean`` (time mean of the seed-averaged null statistic over the dates
    that enter ``mean``, so the real statistic on those dates is mean + null_mean)
    and ``n_seeds``. BH over these 6 rows only; ``verdict`` as H1-H5.
    """
    rows = []
    for keys, x, null_mean in null_excess_series(features, targets, null_panels, dates):
        used = np.isfinite(x.to_numpy(dtype=np.float64))
        nm = null_mean.to_numpy(dtype=np.float64)[used]
        rows.append({**keys, **_summary(x, keys["horizon"]),
                     "block_share": block_sign_share(x, splits, NULL_EXCESS_SIGN),
                     "span": math.nan,
                     "null_mean": float(nm.mean()) if nm.size else math.nan,
                     "n_seeds": len(null_panels)})
    df = pd.DataFrame(rows)
    df["q"] = bh_adjust(df["p"].to_numpy(dtype=np.float64))
    df["verdict"] = [verdict(r) for _, r in df.iterrows()]
    cols = ["section", "group", "name", "target", "horizon", "expected_sign", "n_days", "mean",
            "t_nw", "p", "block_share", "span", "q", "verdict", "null_mean", "n_seeds"]
    return df[cols]


def run_study(features: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame],
              null_features: dict[str, pd.DataFrame], null_targets: dict[str, pd.DataFrame],
              splits: list[Split] | None = None,
              extra_null_panels: list[tuple[dict, dict]] | None = None) -> StudyResult:
    """
    Every statistic of the study on the OOS dates of ``splits`` (harness splits if None).

    ``extra_null_panels`` (the shuffled nulls of ``NULL_EXCESS_SEEDS[1:]``, only the
    ``NULL_EXCESS_FEATURES`` / ``NULL_EXCESS_TARGETS`` columns needed) adds the H6
    family in ``h6``, with ``null_features``/``null_targets`` as the first seed.
    Without it ``h6`` is empty and everything else is unchanged.
    """
    if splits is None:
        splits = study_splits(features["REGIME"])
    oos = oos_dates(splits)
    cond, means = conditional_tests(features, targets, oos, splits)
    rank = rank_tests(features, targets, oos, splits)
    calib, quint, churn = calibration_tests(features, targets, oos, splits)
    tests = pd.concat([cond, rank, calib], ignore_index=True)
    tests["q"] = bh_adjust(tests["p"].to_numpy(dtype=np.float64))
    tests["verdict"] = [verdict(r) for _, r in tests.iterrows()]
    null_cond, _ = conditional_tests(null_features, null_targets, oos, splits)
    null_rank = rank_tests(null_features, null_targets, oos, splits)
    h6 = pd.DataFrame()
    if extra_null_panels is not None:
        h6 = null_excess_tests(features, targets,
                               [(null_features, null_targets), *extra_null_panels], oos, splits)
    return StudyResult(
        splits=splits, oos=oos,
        base_rates_real=label_base_rates(features["REGIME"], oos),
        base_rates_null=label_base_rates(null_features["REGIME"], oos),
        hurst_real=distribution(features[HURST_COL], oos),
        hurst_null=distribution(null_features[HURST_COL], oos),
        tests=tests, group_means=means, null_conditional=null_cond, null_rank=null_rank,
        hit_quintiles=quint, churn=churn, h6=h6,
    )
