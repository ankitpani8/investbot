"""
Market regime: rolling Hurst exponent, rolling largest Lyapunov exponent,
realised volatility, and a soft regime label with a confidence.

Pure compute module: takes an OHLC DataFrame and returns a dict of float64
arrays. No DB I/O. Not wired into ``indicators.add_all_indicators``;
``scorer.py`` composes the technical modules.

Output columns (all float64, NaN during warmup or when the window has a NaN)
    HURST_<W>        rolling Hurst exponent of log returns over W returns (DFA-1).
    LYAP_<W>         rolling largest Lyapunov exponent of log returns over W
                     returns (Rosenstein), in nats per bar.
    REALVOL_<N>      rolling standard deviation of log returns (per bar, not
                     annualised), N = ``VOL_SHORT`` and ``VOL_LONG``.
    VOL_RATIO        REALVOL_<VOL_SHORT> / REALVOL_<VOL_LONG>: today's
                     volatility relative to the asset's own trailing norm.
    REGIME_SCORE_RANGE, REGIME_SCORE_TREND, REGIME_SCORE_CHAOS
                     soft evidence in [0, 1] for each regime (see below).
    REGIME           0 = mean-reverting/range, 1 = trending,
                     2 = chaotic/high-volatility. Argmax of the three scores.
    REGIME_CONFIDENCE_RAW
                     top score minus runner-up score, in [0, 1].
    REGIME_CONFIDENCE
                     causal EMA (span ``CONFIDENCE_EMA_SPAN``) of
                     REGIME_CONFIDENCE_RAW, in [0, 1]. This is the value the
                     confluence weights and the allocator blend read.

Hurst: why DFA and not R/S
    Classical rescaled-range (R/S) analysis is biased upwards on short
    samples (a random walk reads ~0.55-0.6 at 100 points unless the
    Anis-Lloyd correction is applied, and that correction assumes Gaussian
    increments) and it is sensitive to short-range autocorrelation and to
    slow drifts in the mean. Detrended fluctuation analysis with linear
    detrending (DFA-1) removes a local linear trend from the cumulative
    profile in every box, so a constant drift in returns does not leak into
    the estimate, and its small-sample bias at our window sizes is smaller.
    The DFA scaling exponent alpha equals H for a stationary increment
    series, so HURST is alpha computed on log returns: 0.5 for uncorrelated
    returns, > 0.5 for persistent (trending) and < 0.5 for anti-persistent
    (mean-reverting) returns.

    The profile of the returns is their cumulative sum, which is the log
    close up to a constant. A constant is removed by the per-box linear fit,
    so the profile for the window ending at bar t is just
    log(close[t-W .. t]) (W returns, W+1 closes). Boxes of size s are laid
    end-aligned (the newest box ends at t), scales are ``DFA_N_SCALES``
    log-spaced integers in [``DFA_MIN_SCALE``, W // ``DFA_MAX_SCALE_DIV``],
    and HURST is the OLS slope of log F(s) against log s.

    "Trending" for Hurst means persistent returns (a move tends to be
    followed by a move the same way). A constant drift with independent
    returns still reads 0.5, because DFA-1 removes it; the directional side
    of a trend is what ADX_14 contributes to the label.

Lyapunov: Rosenstein (1993)
    Log returns are delay-embedded with dimension ``embed_dim`` and lag
    ``lag``: the point at bar p is (r[p], r[p-lag], ..., r[p-(m-1)lag]), so it
    only uses returns up to p. Inside the window ending at t, every point
    p <= t - horizon gets its nearest neighbour q (Euclidean, |p - q| >
    ``theiler``, non-zero distance), and the mean log distance
    y(k) = <log ||x[p+k] - x[q+k]||> is tracked for k = 0..horizon, staying
    inside the window. LYAP is the OLS slope of y(k) on k.

    Defaults: ``embed_dim = 3`` (daily returns have little low-dimensional
    structure, and every extra dimension thins out neighbours in a 100-bar
    window), ``lag = 1`` (the autocorrelation of daily returns is already
    near zero at lag 1, the usual first-zero-crossing rule), ``theiler = 10``
    bars (two trading weeks, so neighbours are not just adjacent bars),
    ``horizon = 5`` bars (divergence for a bounded series saturates after
    roughly log(W) / lambda steps, so the fit stays in the linear part).

    Interpretation: the estimate is scale invariant (multiplying returns by
    a constant shifts y(k), not its slope). Pure noise also reads positive,
    because neighbours that are close by chance separate at the first step.
    So LYAP is used relative to its typical level on our data
    (``LYAP_CHAOS``), not as a proof of deterministic chaos.

Regime label and confidence
    trend_z = HURST_WEIGHT * (H - HURST_NEUTRAL) / HURST_SCALE
            + ADX_WEIGHT * (ADX_14 - ADX_NEUTRAL) / ADX_SCALE
    chaos_z = VOL_WEIGHT * (VOL_RATIO - VOL_RATIO_CHAOS) / VOL_RATIO_SCALE
            + LYAP_WEIGHT * (LYAP - LYAP_CHAOS) / LYAP_SCALE
    REGIME_SCORE_TREND = sigmoid(trend_z), REGIME_SCORE_RANGE = 1 - trend,
    REGIME_SCORE_CHAOS = sigmoid(chaos_z).
    H and LYAP are taken at ``regime_window``. REGIME is the argmax (ties go
    to the lower code) and REGIME_CONFIDENCE_RAW is the gap between the top
    two scores. Every score is a smooth function of the inputs and the gap of
    the top two is continuous, so the raw confidence is 0 exactly where the
    label flips and grows smoothly away from it. It is not a step function.

    REGIME_CONFIDENCE smooths the raw gap over time with an EMA
    (``CONFIDENCE_EMA_SPAN = 5`` bars, alpha = 2 / (span + 1), recursive form
    y[t] = alpha * x[t] + (1 - alpha) * y[t-1], seeded with the first valid
    raw value). A label that flips back and forth every few bars keeps the
    raw gap jumping between ~0 and its peak; the EMA damps that churn before
    it reaches the Black-Litterman <-> risk-parity blend, where each swing
    would mean rebalancing (tax and fees). It is NaN wherever the raw value
    is NaN, and after a NaN in the middle of the series it restarts from the
    next valid raw value instead of carrying state from before the gap.
    The label itself (REGIME) is not smoothed.

    ADX_14 is computed here from high/low/close with TA-Lib, so the module
    has no dependency on the indicator layer having run. It is not emitted,
    to avoid a duplicate ADX_14 column next to the indicators.

No lookahead
    Every value at bar t uses closes (and, for ADX, high/low/close) up to t
    only. Hurst, Lyapunov and realised volatility are fixed-length trailing
    windows, so they are also invariant to dropping old history. ADX is
    Wilder-smoothed (recursive), so the regime scores depend weakly on where
    the series starts, like ADX itself; the REGIME_CONFIDENCE EMA is
    recursive in the same way. Tested in ``tests/test_regime.py``.

Warmup
    HURST_<W> and LYAP_<W> need W + 1 closes (first value at row W),
    REALVOL_<N> needs N + 1, VOL_RATIO needs VOL_LONG + 1 and ADX_14 needs 28.
    The regime columns are NaN until all of their inputs exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from numpy.lib.stride_tricks import sliding_window_view

PRICE_COLUMNS: tuple[str, ...] = ("high", "low", "close")

#: Default rolling windows, in returns (bars), for HURST_<W>.
HURST_WINDOWS: tuple[int, ...] = (100, 250)
#: Default rolling windows, in returns (bars), for LYAP_<W>.
LYAP_WINDOWS: tuple[int, ...] = (100, 250)
#: Window whose HURST and LYAP feed the regime label. 100 bars (about five
#: months of daily data) reacts to regime changes; 250 lags by a year.
REGIME_WINDOW: int = 100

# DFA scale grid. Tiny boxes bias DFA-1 upwards: on 20 random walks the median
# HURST_100 is 0.56 with a minimum box of 4 and 0.53 with 6 (HURST_250: 0.54
# vs 0.51); 8 removes more bias but leaves W=100 with only ~5 scales and a
# noisier estimate (sd 0.12 vs 0.10).
DFA_MIN_SCALE: int = 6
DFA_MAX_SCALE_DIV: int = 4
DFA_N_SCALES: int = 10

# Rosenstein defaults
EMBED_DIM: int = 3
LAG: int = 1
THEILER: int = 10
HORIZON: int = 5
#: Windows ending in one block share one distance matrix (memory vs speed).
_LYAP_BLOCK: int = 128

# Realised volatility
VOL_SHORT: int = 20
VOL_LONG: int = 250
ADX_PERIOD: int = 14

# Trend vs range evidence. ADX 25 is the classical "trending" line (median
# ADX_14 on NIFTY50 daily is ~22.5). H = 0.6 or ADX = 35 is two scale units.
HURST_NEUTRAL: float = 0.5
HURST_SCALE: float = 0.05
ADX_NEUTRAL: float = 25.0
ADX_SCALE: float = 5.0
HURST_WEIGHT: float = 0.5
ADX_WEIGHT: float = 0.5

# Chaos / high-volatility evidence. VOL_RATIO = 1.25 (20-bar vol 25% above its
# one-year norm, about the 85th percentile on NIFTY50 daily) is the midpoint;
# 1.65 is two scale units. Chaos must also beat the larger of trend/range
# (always >= 0.5), so about 3% of NIFTY50 daily bars are labelled chaotic.
VOL_RATIO_CHAOS: float = 1.25
VOL_RATIO_SCALE: float = 0.2
#: About the 95th percentile of LYAP_100 on NIFTY50 daily returns (median
#: 0.26, 5th-95th 0.23-0.30). LYAP grows with the window (LYAP_250 median
#: 0.32), so retune this if ``regime_window`` changes. See the module
#: docstring for why LYAP is used relative to its own level.
LYAP_CHAOS: float = 0.30
LYAP_SCALE: float = 0.02
VOL_WEIGHT: float = 0.75
LYAP_WEIGHT: float = 0.25

#: Span of the causal EMA that turns REGIME_CONFIDENCE_RAW into
#: REGIME_CONFIDENCE (decided 2026-10-18, see docs/DECISIONS_PENDING.md).
CONFIDENCE_EMA_SPAN: int = 5

REGIME_RANGE: int = 0
REGIME_TREND: int = 1
REGIME_CHAOS: int = 2


# ──────────────────────────────────────────────────────────
# Hurst (DFA-1)
# ──────────────────────────────────────────────────────────

def dfa_scales(window: int) -> np.ndarray:
    """Box sizes used by HURST_<window>."""
    hi = window // DFA_MAX_SCALE_DIV
    if hi < DFA_MIN_SCALE + 1:
        raise ValueError(f"Hurst window {window} too short for DFA")
    raw = np.logspace(np.log10(DFA_MIN_SCALE), np.log10(hi), DFA_N_SCALES)
    return np.unique(np.round(raw).astype(int))


def _box_rss(y: np.ndarray, s: int) -> np.ndarray:
    """Residual sum of squares of a linear fit on every box y[e-s+1 .. e], keyed by e."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < s:
        return out
    boxes = sliding_window_view(y, s)
    x = np.arange(s, dtype=np.float64) - (s - 1) / 2.0
    yc = boxes - boxes.mean(axis=1, keepdims=True)
    sxy = (yc * x).sum(axis=1)
    rss = (yc * yc).sum(axis=1) - sxy * sxy / (x @ x)
    out[s - 1:] = np.maximum(rss, 0.0)
    return out


def rolling_hurst(log_close: np.ndarray, window: int) -> np.ndarray:
    """Rolling DFA-1 Hurst exponent of log returns over ``window`` returns."""
    n = len(log_close)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    scales = dfa_scales(window)
    profile_len = window + 1
    log_f = np.empty((len(scales), n))
    for i, s in enumerate(scales):
        rss = _box_rss(log_close, s)
        nb = profile_len // s
        f2 = np.zeros(n)
        f2[: (nb - 1) * s] = np.nan
        for j in range(nb):
            f2[j * s:] += rss[: n - j * s]
        f2 /= nb * s
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f[i] = np.where(f2 > 0, 0.5 * np.log(f2), np.nan)
    ls = np.log(scales.astype(np.float64))
    w = (ls - ls.mean()) / ((ls - ls.mean()) ** 2).sum()
    h = w @ log_f

    # Every close in log_close[t-window .. t] must be finite.
    finite = np.isfinite(log_close).astype(np.float64)
    count = np.convolve(finite, np.ones(profile_len), mode="full")[:n]
    ok = count == profile_len
    out[ok] = h[ok]
    return out


# ──────────────────────────────────────────────────────────
# Largest Lyapunov exponent (Rosenstein)
# ──────────────────────────────────────────────────────────

def _embed(r: np.ndarray, embed_dim: int, lag: int) -> np.ndarray:
    """Point at bar p is (r[p], r[p-lag], ...); rows before the first full point are NaN."""
    n = len(r)
    emb = np.full((n, embed_dim), np.nan)
    for d in range(embed_dim):
        shift = d * lag
        emb[shift:, d] = r[: n - shift]
    return emb


def rolling_lyapunov(
    log_ret: np.ndarray,
    window: int,
    embed_dim: int = EMBED_DIM,
    lag: int = LAG,
    theiler: int = THEILER,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Rolling Rosenstein largest Lyapunov exponent (per bar) over ``window`` returns."""
    n = len(log_ret)
    out = np.full(n, np.nan)
    span = (embed_dim - 1) * lag
    n_points = window - span                 # embedded points per window
    n_ref = n_points - horizon               # points that can be tracked horizon steps
    if n_ref <= theiler + 1:
        raise ValueError(
            f"Lyapunov window {window} too short for embed_dim={embed_dim}, "
            f"lag={lag}, theiler={theiler}, horizon={horizon}"
        )
    emb = _embed(log_ret, embed_dim, lag)
    point_ok = np.isfinite(emb).all(axis=1)

    ks = np.arange(horizon + 1, dtype=np.float64)
    kw = (ks - ks.mean()) / ((ks - ks.mean()) ** 2).sum()
    k_off = np.arange(horizon + 1)
    ref = np.arange(n_ref)
    band = np.abs(ref[:, None] - ref[None, :]) <= theiler

    first_t = window  # needs log_ret[t-window+1 .. t], and log_ret[0] is NaN
    for t0 in range(first_t, n, _LYAP_BLOCK):
        t1 = min(t0 + _LYAP_BLOCK, n) - 1
        pa = t0 - n_points + 1               # first point of the first window
        pts = emb[pa: t1 + 1]
        ok_blk = point_ok[pa: t1 + 1]
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        # Running count of bad points, to skip windows that contain one.
        bad_cum = np.concatenate([[0], np.cumsum(~ok_blk)])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_dist = np.log(dist)
        log_dist[~np.isfinite(log_dist)] = np.nan
        for t in range(t0, t1 + 1):
            a = t - n_points + 1 - pa
            if bad_cum[a + n_points] - bad_cum[a]:
                continue
            sub = dist[a: a + n_ref, a: a + n_ref].copy()
            sub[band] = np.inf
            sub[sub <= 0] = np.inf
            nn = np.argmin(sub, axis=1)
            has_nn = np.isfinite(sub[ref, nn])
            if not has_nn.any():
                continue
            ii = a + ref[has_nn]
            jj = a + nn[has_nn]
            traj = log_dist[ii[:, None] + k_off, jj[:, None] + k_off]
            valid = np.isfinite(traj).sum(axis=0)
            if (valid == 0).any():
                continue
            y = np.nansum(traj, axis=0) / valid
            out[t] = kw @ y
    return out


# ──────────────────────────────────────────────────────────
# Regime
# ──────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def regime_scores(
    hurst: np.ndarray, adx: np.ndarray, vol_ratio: np.ndarray, lyap: np.ndarray
) -> dict[str, np.ndarray]:
    """Soft evidence, label and raw (unsmoothed) confidence from the four inputs (all same length)."""
    trend_z = (HURST_WEIGHT * (hurst - HURST_NEUTRAL) / HURST_SCALE
               + ADX_WEIGHT * (adx - ADX_NEUTRAL) / ADX_SCALE)
    chaos_z = (VOL_WEIGHT * (vol_ratio - VOL_RATIO_CHAOS) / VOL_RATIO_SCALE
               + LYAP_WEIGHT * (lyap - LYAP_CHAOS) / LYAP_SCALE)
    trend = _sigmoid(trend_z)
    rng = 1.0 - trend
    chaos = _sigmoid(chaos_z)

    scores = np.stack([rng, trend, chaos], axis=1)
    valid = np.isfinite(scores).all(axis=1)
    label = np.full(len(hurst), np.nan)
    conf = np.full(len(hurst), np.nan)
    if valid.any():
        s = scores[valid]
        label[valid] = np.argmax(s, axis=1)
        top2 = np.sort(s, axis=1)[:, -2:]
        conf[valid] = np.clip(top2[:, 1] - top2[:, 0], 0.0, 1.0)
    return {
        "REGIME_SCORE_RANGE": rng,
        "REGIME_SCORE_TREND": trend,
        "REGIME_SCORE_CHAOS": chaos,
        "REGIME": label,
        "REGIME_CONFIDENCE_RAW": conf,
    }


def smooth_confidence(raw: np.ndarray, span: int = CONFIDENCE_EMA_SPAN) -> np.ndarray:
    """
    Causal EMA of ``raw`` (pandas ``ewm(span, adjust=False)``), restarted after every NaN.

    NaN wherever ``raw`` is NaN. Each run of finite values starts again from
    its own first value, so nothing from before a gap leaks past it.
    """
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    x = np.asarray(raw, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            prev = np.nan
            continue
        prev = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def list_regime_columns(
    hurst_windows: tuple[int, ...] = HURST_WINDOWS,
    lyap_windows: tuple[int, ...] = LYAP_WINDOWS,
) -> list[str]:
    """All column names produced by compute_regime, in output order."""
    return (
        [f"HURST_{w}" for w in hurst_windows]
        + [f"LYAP_{w}" for w in lyap_windows]
        + [f"REALVOL_{VOL_SHORT}", f"REALVOL_{VOL_LONG}", "VOL_RATIO"]
        + ["REGIME_SCORE_RANGE", "REGIME_SCORE_TREND", "REGIME_SCORE_CHAOS",
           "REGIME", "REGIME_CONFIDENCE_RAW", "REGIME_CONFIDENCE"]
    )


def compute_regime(
    df: pd.DataFrame,
    hurst_windows: tuple[int, ...] = HURST_WINDOWS,
    lyap_windows: tuple[int, ...] = LYAP_WINDOWS,
    regime_window: int = REGIME_WINDOW,
    embed_dim: int = EMBED_DIM,
    lag: int = LAG,
    theiler: int = THEILER,
    horizon: int = HORIZON,
    confidence_ema_span: int = CONFIDENCE_EMA_SPAN,
) -> dict[str, np.ndarray]:
    """
    Rolling Hurst, Lyapunov, realised volatility and the regime label.

    Parameters
    ----------
    df : DataFrame
        Lowercase 'high', 'low', 'close' columns, oldest bar first. Not mutated.
    hurst_windows, lyap_windows : tuple of int
        Rolling windows in returns for HURST_<W> and LYAP_<W>.
    regime_window : int
        Window whose HURST and LYAP feed the label. Must be in both tuples.
    embed_dim, lag, theiler, horizon : int
        Rosenstein parameters, see the module docstring.
    confidence_ema_span : int
        EMA span that smooths REGIME_CONFIDENCE_RAW into REGIME_CONFIDENCE.

    Returns
    -------
    dict[str, np.ndarray]
        float64 arrays of len(df), keyed as in ``list_regime_columns()``.
    """
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if regime_window not in hurst_windows or regime_window not in lyap_windows:
        raise ValueError(f"regime_window {regime_window} must be in hurst_windows and lyap_windows")

    h, l, c = (df[col].to_numpy(dtype=np.float64, copy=True) for col in PRICE_COLUMNS)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.where(c > 0, np.log(c), np.nan)
    log_ret = np.r_[np.nan, np.diff(log_c)]

    out: dict[str, np.ndarray] = {}
    for w in hurst_windows:
        out[f"HURST_{w}"] = rolling_hurst(log_c, w)
    for w in lyap_windows:
        out[f"LYAP_{w}"] = rolling_lyapunov(log_ret, w, embed_dim, lag, theiler, horizon)

    ret = pd.Series(log_ret)
    vol_s = ret.rolling(VOL_SHORT, min_periods=VOL_SHORT).std().to_numpy()
    vol_l = ret.rolling(VOL_LONG, min_periods=VOL_LONG).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_l > 0, vol_s / vol_l, np.nan)
    out[f"REALVOL_{VOL_SHORT}"] = vol_s
    out[f"REALVOL_{VOL_LONG}"] = vol_l
    out["VOL_RATIO"] = vol_ratio

    adx = talib.ADX(h, l, c, timeperiod=ADX_PERIOD).astype(np.float64)
    out.update(regime_scores(
        out[f"HURST_{regime_window}"], adx, vol_ratio, out[f"LYAP_{regime_window}"]
    ))
    out["REGIME_CONFIDENCE"] = smooth_confidence(out["REGIME_CONFIDENCE_RAW"], confidence_ema_span)
    return out


def add_regime(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return a new DataFrame: ``df`` plus every regime column (one pd.concat)."""
    cols = compute_regime(df, **kwargs)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
