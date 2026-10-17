"""
Buffett income-statement sub-scores (master plan §5.2, income table).

Pure compute module: takes one asset's annual financials and returns a
DataFrame. No DB I/O, no yfinance. The fundamental scorer owns loading and
storage; the composite fundamental score stays a 100 pass-through for the MVP.

Input
    One asset's **annual** rows, indexed by period end date (or integer fiscal
    year), ascending, with column names from the ``financials`` table
    (``investbot/data/models.py``): revenue, cogs, gross_profit, sga_expense,
    rnd_expense, depreciation, amortization, interest_expense, ebit,
    pretax_income, tax_paid, net_income, eps. Missing columns are treated as
    all-NaN and give NaN outputs, never exceptions.

Output (same index, one row per fiscal year)
    For every metric a raw column ``FUND_IS_<NAME>`` and a 0-100 score column
    ``FUND_IS_<NAME>_SCORE``, plus ``FUND_IS_TAX_FLAG`` (bool, no score) and
    ``FUND_IS_SCORE`` (equal-weighted mean of the available sub-scores, NaN
    if fewer than ``MIN_SUBSCORES`` are present).

    GROSS_MARGIN          mean of gross_profit / revenue over the window.
                          gross_profit falls back to revenue - cogs.
    GROSS_MARGIN_STD      sample std (ddof=1) of the yearly gross margins.
    SGA_RATIO             mean of sga_expense / gross_profit over the window.
    SGA_RATIO_STD         sample std of the yearly SGA ratios.
    RND_BURDEN            rnd_expense / gross_profit, latest year.
    DEPRECIATION_BURDEN   depreciation / ebit, latest year.
    INTEREST_BURDEN       interest_expense / ebit, latest year.
    NET_MARGIN            net_income / revenue, latest year.
    NET_MARGIN_SLOPE      OLS slope of the yearly net margin over the window,
                          in percentage points per year (x = period end in
                          years, so a missing year doesn't compress the axis).
    EPS_CONSISTENCY       share of year-over-year EPS increases among the
                          consecutive pairs in the window where both EPS
                          values are known.
    TAX_RATE              tax_paid / pretax_income, latest year.

    Ratios are fractions (0.25 = 25%); only NET_MARGIN_SLOPE is in pp/year.

No lookahead
    Row N reads rows ≤ N only, through a trailing window of at most
    ``lookback_years`` rows (``LOOKBACK_YEARS = 10``). Every mean, std, slope
    and EPS count uses that window; latest-year ratios use row N alone.

Conventions
    * Denominators ≤ 0 (revenue, gross_profit, ebit, pretax_income) give NaN
      for that year's ratio and its score, so a loss year can never read as a
      low (good) burden. Window means/std/slope skip NaN years.
    * Expense numerators (cogs, sga, rnd, depreciation, interest) are taken as
      absolute values: the ingestion stores them positive, and a sign flip in
      another source must not turn an expense into a "negative burden".
      net_income and tax_paid keep their sign (a loss or a tax credit is real).
    * R&D: a missing column or a NaN value counts as 0, not NaN. Indian filers
      rarely break R&D out (yfinance has it for ~15% of NIFTY50 annual rows),
      so NaN would drop the sub-score for almost every asset. The side effect
      is that RND_BURDEN_SCORE is usually 100 whenever gross profit is known.
    * Score band edges: the middle band is closed, so a ratio sitting exactly
      on a threshold gets the middle score (gross margin 40% → 60, SGA 30% → 60).
    * FUND_IS_TAX_FLAG is False when the tax rate is unknown (NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Window ────────────────────────────────────────────────────────────────
LOOKBACK_YEARS = 10
MIN_STD_YEARS = 3            # yearly values needed for a std
MIN_TREND_YEARS = 3          # yearly values needed for the slope / EPS consistency

# ── Score levels ──────────────────────────────────────────────────────────
SCORE_TOP = 100.0
SCORE_MID = 60.0
SCORE_LOW = 20.0
SCORE_DEPRECIATION_MID = 70.0
SCORE_DEPRECIATION_LOW = 40.0
SCORE_INCONSISTENT = 40.0    # high std / volatile EPS

# ── Thresholds (fractions unless stated) ──────────────────────────────────
GROSS_MARGIN_HIGH = 0.40
GROSS_MARGIN_LOW = 0.20
GROSS_MARGIN_STD_MAX = 0.03          # ≤ 3 pp std of gross margin counts as "low"
SGA_RATIO_LOW = 0.30
SGA_RATIO_HIGH = 0.50
SGA_RATIO_STD_MAX = 0.05             # ≤ 5 pp std of the SGA ratio counts as "low"
RND_BURDEN_LOW = 0.10
RND_BURDEN_HIGH = 0.30
DEPRECIATION_BURDEN_LOW = 0.10
DEPRECIATION_BURDEN_HIGH = 0.15
INTEREST_BURDEN_LOW = 0.15
INTEREST_BURDEN_HIGH = 0.25
NET_MARGIN_HIGH = 0.20
NET_MARGIN_LOW = 0.10
NET_MARGIN_SLOPE_FLAT_BAND = 0.5     # pp/year; |slope| ≤ band reads as flat
EPS_CONSISTENCY_HIGH = 0.8
EPS_CONSISTENCY_MID = 0.5
EXPECTED_TAX_RATE = 0.2517           # section 115BAA: 22% + 10% surcharge + 4% cess
TAX_RATE_TOLERANCE = 0.10
MIN_SUBSCORES = 4

PCT_POINTS = 100.0
DAYS_PER_YEAR = 365.25

INPUT_COLUMNS = (
    "revenue", "cogs", "gross_profit", "sga_expense", "rnd_expense",
    "depreciation", "amortization", "interest_expense", "ebit",
    "pretax_income", "tax_paid", "net_income", "eps",
)

PREFIX = "FUND_IS_"
METRICS = (
    "GROSS_MARGIN", "GROSS_MARGIN_STD", "SGA_RATIO", "SGA_RATIO_STD",
    "RND_BURDEN", "DEPRECIATION_BURDEN", "INTEREST_BURDEN", "NET_MARGIN",
    "NET_MARGIN_SLOPE", "EPS_CONSISTENCY", "TAX_RATE",
)
SCORED_METRICS = tuple(m for m in METRICS if m != "TAX_RATE")
TAX_FLAG_COL = f"{PREFIX}TAX_FLAG"
SCORE_COL = f"{PREFIX}SCORE"


# ── Helpers ───────────────────────────────────────────────────────────────

def _col(fin: pd.DataFrame, name: str) -> np.ndarray:
    if name not in fin.columns:
        return np.full(len(fin), np.nan)
    return pd.to_numeric(fin[name], errors="coerce").to_numpy(dtype=np.float64)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, NaN where den ≤ 0 or either side is NaN."""
    out = np.full(len(num), np.nan)
    ok = (den > 0) & np.isfinite(num) & np.isfinite(den)
    out[ok] = num[ok] / den[ok]
    return out


def _higher_better(x: np.ndarray, high: float, low: float,
                   top: float, mid: float, bottom: float) -> np.ndarray:
    """x > high → top, low ≤ x ≤ high → mid, x < low → bottom, NaN → NaN."""
    out = np.where(x > high, top, np.where(x >= low, mid, bottom))
    return np.where(np.isnan(x), np.nan, out)


def _lower_better(x: np.ndarray, low: float, high: float,
                  top: float, mid: float, bottom: float) -> np.ndarray:
    """x < low → top, low ≤ x ≤ high → mid, x > high → bottom, NaN → NaN."""
    out = np.where(x < low, top, np.where(x <= high, mid, bottom))
    return np.where(np.isnan(x), np.nan, out)


def _windows(n: int, lookback: int):
    for i in range(n):
        yield i, slice(max(0, i - lookback + 1), i + 1)


def _rolling_mean(x: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i, w in _windows(len(x), lookback):
        v = x[w][~np.isnan(x[w])]
        if v.size:
            out[i] = v.mean()
    return out


def _rolling_std(x: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i, w in _windows(len(x), lookback):
        v = x[w][~np.isnan(x[w])]
        if v.size >= MIN_STD_YEARS:
            out[i] = v.std(ddof=1)
    return out


def _rolling_slope(y: np.ndarray, t: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for i, w in _windows(len(y), lookback):
        yw, tw = y[w], t[w]
        ok = ~np.isnan(yw)
        if ok.sum() >= MIN_TREND_YEARS:
            tc = tw[ok] - tw[ok].mean()
            if not np.any(tc):
                continue
            out[i] = float(np.sum(tc * (yw[ok] - yw[ok].mean())) / np.sum(tc * tc))
    return out


def _rolling_eps_consistency(eps: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(len(eps), np.nan)
    for i, w in _windows(len(eps), lookback):
        e = eps[w]
        prev, cur = e[:-1], e[1:]
        ok = ~np.isnan(prev) & ~np.isnan(cur)
        if ok.sum() >= MIN_TREND_YEARS - 1:
            out[i] = float(np.mean(cur[ok] > prev[ok]))
    return out


def _year_axis(index: pd.Index) -> np.ndarray:
    """x axis for the slope, in years.

    A datetime index gives fractional years since 1970; a numeric index is
    taken as fiscal years; anything else falls back to the row position.
    """
    if pd.api.types.is_datetime64_any_dtype(index):
        days = (pd.DatetimeIndex(index) - pd.Timestamp("1970-01-01")).days
        return np.asarray(days, dtype=np.float64) / DAYS_PER_YEAR
    if pd.api.types.is_numeric_dtype(index):
        return np.asarray(index, dtype=np.float64)
    return np.arange(len(index), dtype=np.float64)


# ── Public API ────────────────────────────────────────────────────────────

def compute_income_statement_scores(fin: pd.DataFrame, *,
                                    lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Income-statement sub-scores for one asset's annual financials.

    See the module docstring for inputs, outputs and conventions.
    """
    if lookback_years < 1:
        raise ValueError("lookback_years must be ≥ 1")
    if not fin.index.is_monotonic_increasing:
        raise ValueError("fin must be indexed by period end date, ascending")

    revenue = _col(fin, "revenue")
    cogs = np.abs(_col(fin, "cogs"))
    gross_profit = _col(fin, "gross_profit")
    gross_profit = np.where(np.isnan(gross_profit), revenue - cogs, gross_profit)
    sga = np.abs(_col(fin, "sga_expense"))
    rnd = np.nan_to_num(np.abs(_col(fin, "rnd_expense")), nan=0.0)
    depreciation = np.abs(_col(fin, "depreciation"))
    interest = np.abs(_col(fin, "interest_expense"))
    ebit = _col(fin, "ebit")
    pretax = _col(fin, "pretax_income")
    tax_paid = _col(fin, "tax_paid")
    net_income = _col(fin, "net_income")
    eps = _col(fin, "eps")

    gm_year = _ratio(gross_profit, revenue)
    sga_year = _ratio(sga, gross_profit)
    nm_year = _ratio(net_income, revenue)

    raw: dict[str, np.ndarray] = {
        "GROSS_MARGIN": _rolling_mean(gm_year, lookback_years),
        "GROSS_MARGIN_STD": _rolling_std(gm_year, lookback_years),
        "SGA_RATIO": _rolling_mean(sga_year, lookback_years),
        "SGA_RATIO_STD": _rolling_std(sga_year, lookback_years),
        "RND_BURDEN": _ratio(rnd, gross_profit),
        "DEPRECIATION_BURDEN": _ratio(depreciation, ebit),
        "INTEREST_BURDEN": _ratio(interest, ebit),
        "NET_MARGIN": nm_year,
        "NET_MARGIN_SLOPE": _rolling_slope(nm_year * PCT_POINTS, _year_axis(fin.index),
                                           lookback_years),
        "EPS_CONSISTENCY": _rolling_eps_consistency(eps, lookback_years),
        "TAX_RATE": _ratio(tax_paid, pretax),
    }

    scores: dict[str, np.ndarray] = {
        "GROSS_MARGIN": _higher_better(raw["GROSS_MARGIN"], GROSS_MARGIN_HIGH, GROSS_MARGIN_LOW,
                                       SCORE_TOP, SCORE_MID, SCORE_LOW),
        "GROSS_MARGIN_STD": _lower_better(raw["GROSS_MARGIN_STD"], GROSS_MARGIN_STD_MAX,
                                          GROSS_MARGIN_STD_MAX,
                                          SCORE_TOP, SCORE_TOP, SCORE_INCONSISTENT),
        "SGA_RATIO": _lower_better(raw["SGA_RATIO"], SGA_RATIO_LOW, SGA_RATIO_HIGH,
                                   SCORE_TOP, SCORE_MID, SCORE_LOW),
        "SGA_RATIO_STD": _lower_better(raw["SGA_RATIO_STD"], SGA_RATIO_STD_MAX, SGA_RATIO_STD_MAX,
                                       SCORE_TOP, SCORE_TOP, SCORE_INCONSISTENT),
        "RND_BURDEN": _lower_better(raw["RND_BURDEN"], RND_BURDEN_LOW, RND_BURDEN_HIGH,
                                    SCORE_TOP, SCORE_MID, SCORE_LOW),
        "DEPRECIATION_BURDEN": _lower_better(raw["DEPRECIATION_BURDEN"], DEPRECIATION_BURDEN_LOW,
                                             DEPRECIATION_BURDEN_HIGH, SCORE_TOP,
                                             SCORE_DEPRECIATION_MID, SCORE_DEPRECIATION_LOW),
        "INTEREST_BURDEN": _lower_better(raw["INTEREST_BURDEN"], INTEREST_BURDEN_LOW,
                                         INTEREST_BURDEN_HIGH, SCORE_TOP, SCORE_MID, SCORE_LOW),
        "NET_MARGIN": _higher_better(raw["NET_MARGIN"], NET_MARGIN_HIGH, NET_MARGIN_LOW,
                                     SCORE_TOP, SCORE_MID, SCORE_LOW),
        "NET_MARGIN_SLOPE": _higher_better(raw["NET_MARGIN_SLOPE"], NET_MARGIN_SLOPE_FLAT_BAND,
                                           -NET_MARGIN_SLOPE_FLAT_BAND,
                                           SCORE_TOP, SCORE_MID, SCORE_LOW),
        # The EPS bands are "≥ 0.8" and "≥ 0.5", so both edges belong to the upper band.
        "EPS_CONSISTENCY": np.where(
            np.isnan(raw["EPS_CONSISTENCY"]), np.nan,
            np.where(raw["EPS_CONSISTENCY"] >= EPS_CONSISTENCY_HIGH, SCORE_TOP,
                     np.where(raw["EPS_CONSISTENCY"] >= EPS_CONSISTENCY_MID,
                              SCORE_MID, SCORE_INCONSISTENT))),
    }

    cols: dict[str, np.ndarray] = {}
    for name in METRICS:
        cols[f"{PREFIX}{name}"] = raw[name]
        if name in scores:
            cols[f"{PREFIX}{name}_SCORE"] = scores[name]

    tax_rate = raw["TAX_RATE"]
    with np.errstate(invalid="ignore"):
        cols[TAX_FLAG_COL] = np.abs(tax_rate - EXPECTED_TAX_RATE) > TAX_RATE_TOLERANCE

    score_mat = np.column_stack([scores[m] for m in SCORED_METRICS])
    n_avail = np.sum(~np.isnan(score_mat), axis=1)
    total = np.full(len(fin), np.nan)
    enough = n_avail >= MIN_SUBSCORES
    total[enough] = np.nanmean(score_mat[enough], axis=1)
    cols[SCORE_COL] = total

    return pd.concat({k: pd.Series(v, index=fin.index) for k, v in cols.items()}, axis=1)
