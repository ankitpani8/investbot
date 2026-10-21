"""
Markdown report and CSV for one evaluation run (``python main.py evaluate``).

``run_evaluation`` builds the panel from the DB (one asset at a time),
evaluates it and writes

    docs/reports/eval_<YYYYMMDD>.md        the report (committed)
    tmp_output/eval_metrics_<YYYYMMDD>.csv every metrics row (git-ignored), with a
                                           ``universe`` column

With ``universe="index"`` the point-in-time index universe is the headline and
the "active" universe (today's active stocks, unmasked) is evaluated on the
same walk-forward splits for the "Universe effect" table. Features are
computed once: the active panel is the index panel restricted to active
tickers, without ``IN_UNIVERSE`` (identical to ``build_dataset(universe="active")``).

Every headline metric uses the total-return labels (``evaluate.PRIMARY_RETURN``);
the "Dividend effect" table sets them next to the price-only labels. The Setup
records dividend coverage, unadjusted splits (their label periods are NaN) and
what is needed to compare two runs: ``SCORER_VERSION``, the git commit and a
DB fingerprint (``loader.db_fingerprint``).

An existing report is never overwritten: a second run on the same day writes
``eval_<YYYYMMDD>_b.md`` (then ``_c``, ...).

``render_report`` is pure: it only formats a metrics table.
"""
from __future__ import annotations

import logging
import math
import resource
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from investbot.data.index_membership import COMPLETE_FROM, HISTORY_START
from investbot.evaluation.baselines import BASELINE_COLUMNS
from investbot.evaluation.evaluate import (
    BH_LABELS,
    COMPARISON_RETURN,
    INVERTED_SUBSET,
    PRIMARY_RETURN,
    PROMISING_Q,
    PROMISING_SIGN_SHARE,
    ROBUST_COLUMNS,
    ROBUST_TIMING_KEY,
    evaluate,
    headline,
    universe_mask,
)
from investbot.evaluation.labels import HORIZONS
from investbot.evaluation.metrics import (
    HELD_SPLIT_KEYS,
    MIN_NAMES,
    POOLED_MIN_NAMES,
    REFERENCE_TRADE_VALUE,
    RISK_FREE_RATE_ANNUAL,
    ROBUST_ACF_LAGS,
    ROBUST_BOOT_N,
    ROBUST_BOOT_SEED,
    ROBUST_MAX_LAG_FRAC,
    ROBUST_RHO_CLIP,
    ROUND_TRIP_COST,
    SLIPPAGE_PER_SIDE,
    SURVIVORSHIP_KEYS,
    statutory_round_trip_cost,
)
from investbot.evaluation.performance import (
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_MIN_PERIODS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
)
from investbot.evaluation.panel import (
    SCORE_COLUMNS,
    UNIVERSE_COLUMN,
    VARIANT_TIMING_KEY,
    SPLIT_COUNT_STEM,
    build_dataset,
    point_in_time_membership,
    signal_columns,
    universe_coverage,
)
from investbot.evaluation.variants import VARIANT_COLUMNS, list_variant_columns
from investbot.evaluation.walkforward import MIN_TRAIN_BARS, TEST_MONTHS, Split, oos_dates

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "docs" / "reports"
CSV_DIR = REPO_ROOT / "tmp_output"
TOP_N = 15

#: Caveat for the "active" universe (today's constituents back-filled); the
#: "index" universe replaces it with ``universe_note`` (coverage-based).
SURVIVORSHIP_CAVEAT = (
    "**Survivorship bias.** The universe is *today's* NIFTY50 constituents, back-filled over the "
    "whole history. Names that were dropped from the index (usually after underperforming) are "
    "missing and names that were added (usually after outperforming) are present before they "
    "joined. Absolute returns, and any signal correlated with past winners, are biased upward. "
    "Run with `--universe index` for the point-in-time universe."
)


@dataclass
class RunInfo:
    n_assets: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    horizons: tuple[int, ...]
    start: date | None
    elapsed_s: float = math.nan
    universe: str = "active"
    index_code: str = "NIFTY50"
    #: Index members at any point in the panel window (with or without prices).
    n_members: int = 0
    #: Per-OOS-date share of members with a finite TECH_SCORE (``panel.universe_coverage``).
    coverage_mean: float = math.nan
    coverage_min: float = math.nan
    coverage_min_date: pd.Timestamp | None = None
    #: OOS dates left out of the coverage numbers (fewer than MIN_NAMES covered members).
    coverage_skipped_dates: int = 0
    #: Member ticker -> OOS dates as a member, for members with no price rows / too few bars.
    missing_prices: dict[str, int] = field(default_factory=dict)
    short_history: dict[str, int] = field(default_factory=dict)
    #: Panel dates before COMPLETE_FROM (membership fell back to "active") and how many are OOS.
    fallback_dates: int = 0
    fallback_oos_dates: int = 0
    complete_from: date | None = None
    n_active: int = 0
    peak_rss_mb: float = math.nan
    #: Seconds in ``compute_variants`` (all assets) and in ``evaluate`` (all universes).
    variants_s: float = math.nan
    evaluate_s: float = math.nan
    #: Columns ``evaluate`` was given (``signal_columns``); VAR_* among them.
    n_signal_columns: int = 0
    n_variant_columns: int = 0
    #: Return of the headline labels (``evaluate.PRIMARY_RETURN``).
    primary_return: str = PRIMARY_RETURN
    #: Dividend coverage: names with >= 1 dividend row (ex-date in the panel window), rows.
    n_dividend_names: int = 0
    n_dividend_rows: int = 0
    #: Unadjusted splits: (ticker, ex_date, factor, label periods NaN'd on SPLIT_COUNT_STEM).
    split_issues: list[tuple[str, date, float, int]] = field(default_factory=list)
    #: Reproducibility: scorer version, ``git rev-parse --short HEAD``, DB fingerprint.
    scorer_version: str = ""
    git_commit: str = "unknown"
    db_max_date: date | None = None
    db_price_rows: int = 0
    db_action_rows: int = 0
    #: Seconds in the robust IC step (``evaluate.ROBUST_TIMING_KEY``, both universes) and the
    #: number of tests it ran on (BH rows with a finite ``ic_lag_auto``, both universes).
    robust_s: float = math.nan
    robust_n_tests: int = 0
    #: Median number of names with a finite TECH_SCORE (inside the universe) per OOS date,
    #: over the dates with >= MIN_NAMES of them (``oos_names_median``); NaN if unknown.
    names_median: float = math.nan


def history_caveat(splits: list[Split], info: RunInfo) -> str:
    """The caveat on how much out-of-sample history there is, built from the run's splits."""
    oos = oos_dates(splits)
    if len(oos) == 0:
        return "There are no out-of-sample dates in this run (too little history after warmup)."
    start, end = oos.min(), oos.max()
    years = (end - start).days / 365.25
    names = (f"; median {info.names_median:.0f} names per OOS date"
             if info.names_median == info.names_median else "")
    return (f"Out-of-sample window: {start:%Y-%m-%d} → {end:%Y-%m-%d}, {len(oos)} dates "
            f"({years:.1f} years), after warmup (EMA_200, regime window) and the first training "
            f"year{names}.")


def oos_names_median(features: pd.DataFrame, oos: pd.DatetimeIndex) -> float:
    """Median over the OOS dates of the names with a finite TECH_SCORE (inside the universe when
    ``IN_UNIVERSE`` is present), counting only the dates with >= MIN_NAMES of them: the dates
    the metrics use. NaN if there are none."""
    col = features["TECH_SCORE"]
    mask = universe_mask(features)
    if mask is not None:
        col = col.where(mask)
    counts = col.notna().groupby(level="date").sum().reindex(oos)
    counts = counts[counts >= MIN_NAMES]
    return float(counts.median()) if len(counts) else math.nan


def universe_note(info: RunInfo) -> str:
    """Header caveat: the survivorship caveat for "active", coverage for "index"."""
    if info.universe != "index":
        return SURVIVORSHIP_CAVEAT
    text = (f"**Universe: point-in-time {info.index_code}.** A stock is scored only on dates "
            f"when it was an index member (`index_membership`, changes effective on or before "
            f"the date); names that left the index count for the periods they were members. "
            f"OOS coverage (members with a TECH_SCORE / members, per date): mean "
            f"{_share(info.coverage_mean)}, minimum {_share(info.coverage_min)}"
            + (f" ({info.coverage_min_date:%Y-%m-%d})" if info.coverage_min_date is not None
               else "") + ".")
    if np.isfinite(info.coverage_mean) and info.coverage_min < 1.0:
        oos_missing = [t for t, n in {**info.missing_prices, **info.short_history}.items() if n]
        text += (" **Coverage is below 100%:** members without usable price history (or still in "
                 "TECH_SCORE warmup) are left out on those dates, so a residual survivorship bias "
                 "remains" + (f" (no usable prices during the OOS window: {', '.join(oos_missing)})"
                              if oos_missing else "") + ".")
    if info.fallback_oos_dates:
        text += (f" {info.fallback_oos_dates} OOS dates fall before {info.complete_from} and use "
                 "today's active stocks.")
    return text


def dividend_warning(info: RunInfo) -> str:
    """A prominent warning when no dividend reached the labels ("" otherwise)."""
    if info.primary_return != "tr" or info.n_dividend_names > 0:
        return ""
    why = ("`corporate_actions` is empty" if info.db_action_rows == 0
           else "`corporate_actions` has no dividend rows in the evaluation window")
    return (f"**Warning: total return equals price in this run.** {why}, so every "
            "total-return label equals its price-only label and the \"Dividend effect\" "
            "table shows no difference. Run `python main.py ingest` (it fills "
            "`corporate_actions` from the daily download) and re-run `python main.py evaluate`.")


def git_commit(cwd: Path | None = None) -> str:
    """``git rev-parse --short HEAD`` of the repo (or ``cwd``), ``"unknown"`` if that fails."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd or REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    head = out.stdout.strip()
    return head if out.returncode == 0 and head else "unknown"


def _label_name(info: RunInfo) -> str:
    return "total-return" if info.primary_return == "tr" else "price"


def _share(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.1%}"


# ──────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────

def _f(x, fmt="{:+.3f}") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return fmt.format(x)


def _int(x) -> str:
    return "–" if x is None or not np.isfinite(x) else str(int(x))


def _pct(x) -> str:
    return _f(x * 100 if x is not None and np.isfinite(x) else math.nan, "{:+.2f}%")


def _dd_days_text(days, recovered) -> str:
    """Drawdown duration in days; ``*`` marks a worst drawdown that had not recovered by the end."""
    if days is None or not np.isfinite(days):
        return "–"
    return f"{int(days)}" + ("" if recovered == 1.0 else "*")


def with_dd_days_text(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus the ``held_dd_days_text`` column used by the "Held portfolio" table."""
    if "held_max_dd_duration_days_net" not in df:
        return df.assign(held_dd_days_text="–")
    return df.assign(held_dd_days_text=[
        _dd_days_text(d, rec) for d, rec in zip(df["held_max_dd_duration_days_net"],
                                                df["held_max_dd_recovered_net"])])


def _interval_text(lo, hi, fmt) -> str:
    """``[lo, hi]`` with each end formatted by ``fmt``; "–" when either end is missing."""
    if any(v is None or not np.isfinite(v) for v in (lo, hi)):
        return "–"
    return f"[{fmt(lo)}, {fmt(hi)}]"


#: (text column, lo key, hi key, formatter) of the bootstrap-interval table.
_BOOT_INTERVALS = (
    ("held_ir_text", "held_ir_net_lo", "held_ir_net_hi", lambda x: _f(x, "{:+.2f}")),
    ("held_sharpe_text", "held_sharpe_net_lo", "held_sharpe_net_hi", lambda x: _f(x, "{:+.2f}")),
    ("held_cagr_text", "held_cagr_net_lo", "held_cagr_net_hi", _pct),
    ("held_excess_text", "held_excess_cagr_lo", "held_excess_cagr_hi", _pct),
    ("held_sortino_text", "held_sortino_net_lo", "held_sortino_net_hi", lambda x: _f(x, "{:+.2f}")),
    ("held_calmar_text", "held_calmar_net_lo", "held_calmar_net_hi", lambda x: _f(x, "{:+.2f}")),
)


def with_interval_text(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus the ``[lo, hi]`` text columns of the bootstrap-interval table."""
    cols = {}
    for name, lo, hi, fmt in _BOOT_INTERVALS:
        cols[name] = ([_interval_text(a, b, fmt) for a, b in zip(df[lo], df[hi])]
                      if lo in df and hi in df else "–")
    return df.assign(**cols)


def _table(df: pd.DataFrame, cols: list[tuple[str, str, callable]]) -> str:
    """Markdown table: cols = [(header, column, formatter)]."""
    if df.empty:
        return "_no rows_\n"
    head = "| " + " | ".join(h for h, _, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fn(r[c]) if c in r else "–" for _, c, fn in cols) + " |")
    return "\n".join(lines) + "\n"


_s = str
_IC_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    ("days", "n_days", _int), ("mean IC", "ic_mean", _f), ("IC sd", "ic_std", _f),
    ("ICIR", "icir", _f), ("NW t", "ic_t_nw", lambda x: _f(x, "{:+.2f}")),
    ("p", "ic_p", lambda x: _f(x, "{:.3f}")), ("BH q", "ic_q", lambda x: _f(x, "{:.3f}")),
    ("hit", "ic_hit", lambda x: _f(x, "{:.2f}")),
    ("block sign", "block_sign_share", lambda x: _f(x, "{:.2f}")),
    ("promising", "promising", _f),
]
_Q_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    *[(f"Q{k}", f"q{k}_mean", _pct) for k in range(1, 6)],
    ("top−bottom", "spread_mean", _pct), ("NW t", "spread_t_nw", lambda x: _f(x, "{:+.2f}")),
    ("monotonic", "monotonicity", lambda x: _f(x, "{:+.2f}")),
    ("top turnover", "top_turnover", lambda x: _f(x, "{:.2f}")),
]
_COST_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int), ("periods", "lo_n_periods", _int),
    ("turnover", "top_turnover", lambda x: _f(x, "{:.2f}")),
    ("gross /yr", "lo_gross_ann", _pct), ("net /yr", "lo_net_ann", _pct),
    ("universe /yr", "lo_universe_ann", _pct), ("net − universe /yr", "lo_excess_net_ann", _pct),
]
_HELD_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int), ("periods", "held_n_periods", _int),
    ("turnover", "top_turnover", lambda x: _f(x, "{:.2f}")),
    ("CAGR gross", "held_cagr_gross", _pct), ("CAGR net", "held_cagr_net", _pct),
    ("CAGR universe", "held_cagr_universe", _pct), ("vol /yr", "held_vol_ann_net", _pct),
    ("max DD", "held_max_drawdown_net", _pct),
    ("IR vs universe", "held_info_ratio_net", lambda x: _f(x, "{:+.2f}")),
    ("Sharpe net", "held_sharpe_net", lambda x: _f(x, "{:+.2f}")),
    ("Sharpe univ.", "held_sharpe_universe", lambda x: _f(x, "{:+.2f}")),
    ("Sortino", "held_sortino_net", lambda x: _f(x, "{:+.2f}")),
    ("Calmar", "held_calmar_net", lambda x: _f(x, "{:+.2f}")),
    ("max DD days", "held_dd_days_text", _s),
    ("CVaR95 /period", "held_cvar95_net", lambda x: _f(x * 100, "{:.2f}%")),
]
_BOOT_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int), ("periods", "held_n_periods", _int),
    ("mean block", "held_boot_mean_block", _int),
    ("IR [lo, hi]", "held_ir_text", _s),
    ("share IR > 0", "held_ir_net_share_pos", lambda x: _f(x, "{:.2f}")),
    ("Sharpe net [lo, hi]", "held_sharpe_text", _s),
    ("CAGR net [lo, hi]", "held_cagr_text", _s),
    ("net − universe CAGR [lo, hi]", "held_excess_text", _s),
    ("Sortino net [lo, hi]", "held_sortino_text", _s),
    ("Calmar net [lo, hi]", "held_calmar_text", _s),
]
_SPLIT_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int), ("subset", "subset", _s),
    ("days", "n_days", _int), ("mean IC", "ic_mean", _f), ("ICIR", "icir", _f),
    ("NW t", "ic_t_nw", lambda x: _f(x, "{:+.2f}")), ("top−bottom", "spread_mean", _pct),
]

_REGIME_COLS = [
    *_SPLIT_COLS,
    ("pooled IC", "pooled_ic", _f), ("pooled NW t", "pooled_t_nw", lambda x: _f(x, "{:+.2f}")),
    ("pooled dates", "pooled_n_dates", _int), ("pooled obs", "pooled_n_obs", _int),
]


def _order(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    pos = {n: i for i, n in enumerate(names)}
    return df[df["signal"].isin(pos)].assign(_o=lambda d: d["signal"].map(pos)).sort_values(
        ["_o", "horizon"]).drop(columns="_o")


# ──────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────

def verdict_lines(head: pd.DataFrame, horizons) -> list[str]:
    """Plain-language findings, computed from the headline rows only."""
    lines = []
    base = head[head["signal"].isin(BASELINE_COLUMNS)]
    for h in horizons:
        hh = head[head["horizon"] == h]
        if hh.empty:
            continue
        b = base[base["horizon"] == h]
        best_b = b.loc[b["icir"].idxmax()] if b["icir"].notna().any() else None
        ts = hh[hh["signal"] == "TECH_SCORE"]
        parts = [f"**h = {h}:**"]
        if not ts.empty:
            r = ts.iloc[0]
            parts.append(f"TECH_SCORE mean IC {_f(r['ic_mean'])} (ICIR {_f(r['icir'])}, "
                         f"NW t {_f(r['ic_t_nw'], '{:+.2f}')}, BH q {_f(r['ic_q'], '{:.2f}')})"
                         f"{', promising' if r['promising'] else ', not significant'}.")
        if best_b is not None:
            parts.append(f"Best baseline {best_b['signal']} ICIR {_f(best_b['icir'])}.")
        beat = hh[(~hh["signal"].isin(BASELINE_COLUMNS)) & hh["promising"]
                  & (hh["ic_mean"] > 0)
                  & (hh["icir"] > (best_b["icir"] if best_b is not None else -np.inf))]
        if beat.empty:
            parts.append("No technical signal beats the baselines as designed.")
        else:
            parts.append("Beats the baselines as designed: " + ", ".join(beat["signal"]) + ".")
        inv = hh[hh["promising"] & (hh["ic_mean"] < 0)]
        if not inv.empty:
            parts.append("Promising only with the sign inverted: " + ", ".join(inv["signal"]) + ".")
        lines.append(" ".join(parts))
    return lines


# ──────────────────────────────────────────────────────────
# Polarity variants (task 180)
# ──────────────────────────────────────────────────────────

_VARIANT_COLS = [
    ("signal", "signal", _s), ("vs", "_vs", _s), ("h", "horizon", _int),
    ("mean IC", "ic_mean", _f), ("ICIR", "icir", _f),
    ("NW t", "ic_t_nw", lambda x: _f(x, "{:+.2f}")), ("BH q", "ic_q", lambda x: _f(x, "{:.3f}")),
    ("block sign", "block_sign_share", lambda x: _f(x, "{:.2f}")),
    ("promising", "promising", _f),
    ("held CAGR net", "held_cagr_net", _pct),
    ("held IR", "held_info_ratio_net", lambda x: _f(x, "{:+.2f}")),
]


def best_baseline(head: pd.DataFrame, h: int) -> pd.Series | None:
    """The baseline with the highest ICIR at horizon h (as in the Verdict), or None."""
    b = head[head["signal"].isin(BASELINE_COLUMNS) & (head["horizon"] == h)]
    return b.loc[b["icir"].idxmax()] if b["icir"].notna().any() else None


def _beats(v: pd.Series, prod: pd.Series | None, base: pd.Series | None) -> bool:
    """Promising as designed (q, block sign, IC > 0) with an ICIR above both production
    and the best baseline: the same bar the Verdict uses for "beats the baselines"."""
    if not (bool(v["promising"]) and v["ic_mean"] > 0 and np.isfinite(v["icir"])):
        return False
    for other in (prod, base):
        if other is not None and np.isfinite(other["icir"]) and not v["icir"] > other["icir"]:
            return False
    return True


def variant_verdicts(head: pd.DataFrame, horizons) -> list[str]:
    """One sentence per variant column: does it beat production *and* the best baseline?"""
    rows = head.set_index(["signal", "horizon"])
    lines = []
    for var in list_variant_columns():
        prod_name = VARIANT_COLUMNS[var]
        won, deltas = [], []
        for h in horizons:
            if (var, h) not in rows.index:
                continue
            v = rows.loc[(var, h)]
            prod = rows.loc[(prod_name, h)] if (prod_name, h) in rows.index else None
            if _beats(v, prod, best_baseline(head, h)):
                won.append(str(h))
            if prod is not None:
                deltas.append(f"h = {h} ΔICIR {_f(v['icir'] - prod['icir'])}, Δ held CAGR net "
                              f"{_pct(v['held_cagr_net'] - prod['held_cagr_net'])}")
        if not deltas:
            continue
        verdict = (f"beats production and the best baseline at h = {', '.join(won)}"
                   if won else "beats production and the best baseline at no horizon")
        lines.append(f"**{var}** (vs {prod_name}): {verdict} ({'; '.join(deltas)}).")
    return lines


def variants_section(head: pd.DataFrame, horizons, info: RunInfo) -> str:
    """The "Polarity variants (pre-registered)" section, from the headline rows."""
    out: list[str] = []
    w = out.append
    w("## Polarity variants (pre-registered)\n")
    if not head["signal"].isin(list(VARIANT_COLUMNS)).any():
        w("_No VAR_* columns in this run._\n")
        return "\n".join(out) + "\n"
    universe = ("the point-in-time index universe" if info.universe == "index"
                else f"the {info.universe} universe")
    w("Alternatives to the production oscillator polarity (follow momentum, fade to 0 at the "
      "extremes), fixed by Ankit's 2026-10-18 decision before any results were seen and not "
      "tuned (`investbot/evaluation/variants.py`): **VAR_OSC_REVERT** full mean reversion "
      "(`osc_extreme = −1`: a pinned oscillator is a sell); **VAR_OSC_MOMENTUM** pure momentum "
      "(`osc_extreme = +1`, no fade); **VAR_REGIME_POLARITY** production signals with every "
      "oscillator-rule signal (RSI, STOCH, WILLR, CCI, ULTOSC, BB_POS, KC_POS, MFI) negated on "
      "range-regime bars (revert in range, follow momentum otherwise). Each is aggregated with "
      "the production weights, regime blend and multiplier; `_MOMENTUM` is its momentum-family "
      f"score, compared with TECH_SCORE_MOMENTUM. Same walk-forward blocks, labels and "
      f"Benjamini–Hochberg family as every other signal ({len(head)} tests), on {universe}. "
      "Held CAGR net / IR: the open-to-open top-quintile portfolio (see above). "
      "**Beats** = promising as designed (q < "
      f"{PROMISING_Q:.2f}, block sign ≥ {PROMISING_SIGN_SHARE:.0%}, IC > 0) with an ICIR above "
      "both production and the best baseline.\n")
    order = ["TECH_SCORE", *[v for v in list_variant_columns() if VARIANT_COLUMNS[v] == "TECH_SCORE"],
             "TECH_SCORE_MOMENTUM",
             *[v for v in list_variant_columns() if VARIANT_COLUMNS[v] == "TECH_SCORE_MOMENTUM"]]
    for h in horizons:
        hh = _order(head[head["horizon"] == h], order).assign(
            _vs=lambda d: d["signal"].map(lambda x: VARIANT_COLUMNS.get(x, "production")))
        w(f"### h = {h}\n")
        w(_table(hh, _VARIANT_COLS))
        b = best_baseline(head, h)
        if b is not None:
            w(f"Best baseline at h = {h}: {b['signal']} (mean IC {_f(b['ic_mean'])}, ICIR "
              f"{_f(b['icir'])}, held CAGR net {_pct(b['held_cagr_net'])}, held IR "
              f"{_f(b['held_info_ratio_net'], '{:+.2f}')}).\n")
    for line in variant_verdicts(head, horizons):
        w(f"- {line}")
    w("")
    w("Production polarity is unchanged. Whether to switch it to a variant is Ankit's decision "
      "(docs/DECISIONS_PENDING.md, \"Polarity variants: harness results\").\n")
    return "\n".join(out) + "\n"


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

def universe_effect(index_table: pd.DataFrame, active_table: pd.DataFrame,
                    names: list[str]) -> pd.DataFrame:
    """Headline mean IC, ICIR and held net excess CAGR (net − universe) under both universes."""
    cols = ["signal", "horizon", "n_days", "ic_mean", "icir", "held_cagr_net",
            "held_cagr_universe"]
    a = headline(index_table)[cols]
    b = headline(active_table)[cols]
    m = a.merge(b, on=["signal", "horizon"], how="outer", suffixes=("_index", "_active"))
    for u in ("index", "active"):
        m[f"held_excess_{u}"] = m[f"held_cagr_net_{u}"] - m[f"held_cagr_universe_{u}"]
    m["d_ic"] = m["ic_mean_index"] - m["ic_mean_active"]
    return _order(m, names)


_EFFECT_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    ("mean IC index", "ic_mean_index", _f), ("mean IC active", "ic_mean_active", _f),
    ("Δ IC", "d_ic", _f),
    ("ICIR index", "icir_index", _f), ("ICIR active", "icir_active", _f),
    ("held net − univ. index", "held_excess_index", _pct),
    ("held net − univ. active", "held_excess_active", _pct),
    ("CAGR univ. index", "held_cagr_universe_index", _pct),
    ("CAGR univ. active", "held_cagr_universe_active", _pct),
]


def dividend_effect(table: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Headline mean IC and held-portfolio CAGR (net, universe) under the primary return
    (``_tr``) and the price-only comparison rows (``_price``)."""
    cols = ["signal", "horizon", "ic_mean", "icir", "ic_q", "promising", "held_cagr_net",
            "held_cagr_universe"]
    a = headline(table)[cols]
    b = headline(table, COMPARISON_RETURN)[cols]
    m = a.merge(b, on=["signal", "horizon"], how="inner", suffixes=("_tr", "_price"))
    m["d_ic"] = m["ic_mean_tr"] - m["ic_mean_price"]
    m["d_held"] = m["held_cagr_net_tr"] - m["held_cagr_net_price"]
    return _order(m, names)


_DIVIDEND_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    ("mean IC price", "ic_mean_price", _f), ("mean IC TR", "ic_mean_tr", _f),
    ("Δ IC", "d_ic", _f),
    ("held CAGR net price", "held_cagr_net_price", _pct),
    ("held CAGR net TR", "held_cagr_net_tr", _pct), ("Δ held", "d_held", _pct),
    ("CAGR univ. price", "held_cagr_universe_price", _pct),
    ("CAGR univ. TR", "held_cagr_universe_tr", _pct),
]


def verdict_changes(table: pd.DataFrame) -> pd.DataFrame:
    """Tests whose BH significance (q < PROMISING_Q) or promising flag differs between the
    primary (TR) family and the price-only family."""
    m = dividend_effect(table, sorted(set(headline(table)["signal"])))
    sig_t = m["ic_q_tr"] < PROMISING_Q
    sig_p = m["ic_q_price"] < PROMISING_Q
    diff = (sig_t != sig_p) | (m["promising_tr"].astype(bool) != m["promising_price"].astype(bool))
    return m[diff].sort_values(["signal", "horizon"])


def dividend_section(table: pd.DataFrame, info: RunInfo, names: list[str]) -> str:
    out: list[str] = []
    w = out.append
    w("## Dividend effect: total return vs price only (OOS)\n")
    if not (table["label"] == COMPARISON_RETURN).any():
        w(f"_No price-only comparison rows (primary return is {info.primary_return!r})._\n")
        return "\n".join(out) + "\n"
    warn = dividend_warning(info)
    if warn:
        w(f"> {warn}\n")
    w("Same signals, walk-forward splits, buckets and costs; only the label changes. "
      "*TR* = `FWD_TR_RET_h` for the IC and `FWD_TR_OO_RET_h` for the held portfolio "
      "(the headline everywhere else in this report); *price* = `FWD_RET_h` / `FWD_OO_RET_h`, "
      "the labels of the reports before task 200. Held CAGR net: the open-to-open "
      "top-quintile portfolio net of costs; CAGR univ.: the equal-weight universe over the "
      "same periods. Dividends are not reinvested within a period and not taxed.\n")
    w(_table(dividend_effect(table, names), _DIVIDEND_COLS))
    flips = verdict_changes(table)
    if flips.empty:
        w("No test changes BH significance or promising flag between price and TR "
          f"(each its own Benjamini–Hochberg family of {len(headline(table))} tests).\n")
    else:
        w(f"Tests whose BH significance (q < {PROMISING_Q:.2f}) or promising flag differs "
          "between price and TR (each its own BH family):\n")
        w(_table(flips, [("signal", "signal", _s), ("h", "horizon", _int),
                         ("IC TR", "ic_mean_tr", _f), ("q TR", "ic_q_tr", lambda x: _f(x, "{:.3f}")),
                         ("promising TR", "promising_tr", _f),
                         ("IC price", "ic_mean_price", _f),
                         ("q price", "ic_q_price", lambda x: _f(x, "{:.3f}")),
                         ("promising price", "promising_price", _f)]))
    return "\n".join(out) + "\n"


def _setup_label_lines(info: RunInfo) -> list[str]:
    """Dividend coverage, split issues and the reproducibility line."""
    lines = []
    warn = dividend_warning(info)
    lines.append(f"- Dividend coverage: {info.n_dividend_names} of {info.n_assets} names have at "
                 f"least one dividend row with an ex-date in the panel window, "
                 f"{info.n_dividend_rows} dividend rows in total"
                 + (f". {warn}" if warn else "."))
    if info.split_issues:
        total = sum(n for *_, n in info.split_issues)
        lines.append(
            f"- Unadjusted splits (`quality.detect_unadjusted_splits`, history before the "
            f"ex-date on the old share scale): "
            + ", ".join(f"{t} {d} (×{f:g}, {n} periods)" for t, d, f, n in info.split_issues)
            + f". Every label whose period spans the ex-date boundary is NaN: {total} "
            f"`{SPLIT_COUNT_STEM}_h` periods over all h (the other label columns are NaN on "
            "the matching periods). Re-adjusting that history is a human decision "
            "(`python main.py dq`).")
    else:
        lines.append("- Unadjusted splits: none detected (no label period NaN'd).")
    fp_date = f"{info.db_max_date}" if info.db_max_date is not None else "–"
    lines.append(f"- Reproducibility: `SCORER_VERSION` {info.scorer_version or '–'}, git commit "
                 f"`{info.git_commit}`, DB fingerprint: max `price_daily.date` {fp_date}, "
                 f"{info.db_price_rows} `price_daily` rows, {info.db_action_rows} "
                 "`corporate_actions` rows.")
    return lines


def _setup_universe_lines(info: RunInfo) -> list[str]:
    span = (f"Bars {info.first_date:%Y-%m-%d} → {info.last_date:%Y-%m-%d}"
            + (f"; panel from {info.start}" if info.start else ""))
    if info.universe != "index":
        return [f"- Universe: {info.n_assets} active stocks (no point-in-time masking); {span}."]
    lines = [
        f"- Universe: point-in-time {info.index_code} from `index_membership`: "
        f"{info.n_members} names were members at some point since "
        f"{info.start or HISTORY_START}; "
        f"{info.n_assets} have enough daily bars to compute features (on their full stored "
        f"history; a signal counts only on dates with `IN_UNIVERSE`). {span}.",
        f"- Coverage on OOS dates (members with a finite TECH_SCORE / members): mean "
        f"{_share(info.coverage_mean)}, minimum {_share(info.coverage_min)}"
        + (f" on {info.coverage_min_date:%Y-%m-%d}" if info.coverage_min_date is not None else "")
        + f", over the OOS dates with ≥ {MIN_NAMES} covered members (the dates the metrics use; "
        f"{info.coverage_skipped_dates} OOS dates with fewer, e.g. purged sessions, are left out).",
    ]
    if info.missing_prices:
        lines.append("- Members with **no price rows** (need a human-run ingest; OOS member days "
                     "in brackets): " + ", ".join(f"{t} ({n})" for t, n in
                                                  sorted(info.missing_prices.items())) + ".")
    if info.short_history:
        lines.append(f"- Members with too few bars to score (OOS member days): "
                     + ", ".join(f"{t} ({n})" for t, n in sorted(info.short_history.items())) + ".")
    lines.append(f"- Membership before {info.complete_from} is not sourced and falls back to "
                 f"today's active stocks: {info.fallback_dates} panel dates "
                 f"({info.fallback_oos_dates} OOS).")
    lines.append(f"- Comparison universe \"active\": the {info.n_active} active stocks with "
                 "enough bars, no masking (the pre-170 behaviour), evaluated on the same "
                 "walk-forward splits (the calendar comes from the index universe).")
    return lines


# ──────────────────────────────────────────────────────────
# IC inference robustness (task 340)
# ──────────────────────────────────────────────────────────

def _yes_no(x) -> str:
    return "yes" if bool(x) else "no"


def _lag_text(h, lag) -> str:
    return "–" if lag is None or not np.isfinite(lag) else f"{int(h) - 1} → {int(lag)}"


_ROBUST_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int), ("IC mean", "ic_mean", _f),
    *[(f"ACF{lag}", f"ic_acf{lag}", lambda x: _f(x, "{:+.2f}")) for lag in ROBUST_ACF_LAGS],
    ("lag h−1 → auto", "_lag_text", _s),
    ("t NW", "ic_t_nw", lambda x: _f(x, "{:+.2f}")),
    ("t auto", "ic_t_auto", lambda x: _f(x, "{:+.2f}")),
    ("q", "ic_q", lambda x: _f(x, "{:.3f}")), ("q auto", "ic_q_auto", lambda x: _f(x, "{:.3f}")),
    ("q boot", "ic_q_boot", lambda x: _f(x, "{:.3f}")),
    ("promising", "promising", _yes_no), ("auto", "promising_auto", _yes_no),
    ("boot", "promising_boot", _yes_no),
]

_FAMILY_NAMES = {"open": "Headline family (next-open, primary return)",
                 COMPARISON_RETURN: "Price-only comparison family"}


def robust_changes(fam: pd.DataFrame) -> dict[str, int]:
    """Number of tests whose ``promising`` flag differs under the auto and boot q-values."""
    base = fam["promising"].astype(bool)
    return {kind: int((fam[f"promising_{kind}"].astype(bool) != base).sum())
            for kind in ("auto", "boot")}


def robust_rows(fam: pd.DataFrame) -> pd.DataFrame:
    """Tests promising under any of the three rules, plus TECH_SCORE at every horizon."""
    any_rule = (fam["promising"].astype(bool) | fam["promising_auto"].astype(bool)
                | fam["promising_boot"].astype(bool) | (fam["signal"] == "TECH_SCORE"))
    sel = fam[any_rule].sort_values(["signal", "horizon"])
    return sel.assign(_lag_text=[_lag_text(h, lag) for h, lag in
                                 zip(sel["horizon"], sel["ic_lag_auto"])])


def robust_section(table: pd.DataFrame, info: RunInfo | None = None) -> str:
    """"IC inference robustness (pre-registered)": the BH verdicts under the robust q-values."""
    out = ["## IC inference robustness (pre-registered)\n"]
    if "ic_p_auto" not in table.columns or not table["ic_p_auto"].notna().any():
        out.append("The robust IC step did not run for this table.\n")
        return "\n".join(out)
    out.append(
        "The verdict above uses a Newey-West t of the per-date IC series with a fixed lag h − 1, "
        "which covers the label overlap but not any further persistence of the IC series (at "
        "h = 1 it is a plain t-test). Two autocorrelation-robust p-values were pre-registered "
        "before this run (task 340) and are BH-adjusted within the same families: **auto** = "
        "Newey-West at the Andrews (1991) AR(1) plug-in lag for the Bartlett kernel, never below "
        f"h − 1 (lag-1 ACF clipped at ±{ROBUST_RHO_CLIP:g}, lag capped at "
        f"floor({ROBUST_MAX_LAG_FRAC:g} n)); **boot** = stationary block bootstrap of the "
        f"centred IC series ({ROBUST_BOOT_N} draws, seed {ROBUST_BOOT_SEED}, mean block "
        "max(h, auto lag + 1, round(n^(1/3))) dates), two-sided. ACF = sample autocorrelation "
        "of the per-date IC series. The promising rule is the same (q < "
        f"{PROMISING_Q:.2f} and the block sign share ≥ {PROMISING_SIGN_SHARE:.0%}), only the "
        "q-value differs.\n")
    fams = {label: headline(table, label) for label in BH_LABELS}
    fams = {k: v for k, v in fams.items() if not v.empty}
    for label, fam in fams.items():
        ch = robust_changes(fam)
        out.append(f"- {_FAMILY_NAMES.get(label, label)}: {len(fam)} tests; promising "
                   f"{int(fam['promising'].sum())} (fixed lag), "
                   f"{int(fam['promising_auto'].sum())} (auto), "
                   f"{int(fam['promising_boot'].sum())} (boot); verdict changes "
                   f"{ch['auto']} under auto, {ch['boot']} under boot; median ACF1 of the IC "
                   f"series {_f(float(fam['ic_acf1'].median()), '{:+.3f}')}.")
    out.append("")
    for label, fam in fams.items():
        if label != "open" and not any(robust_changes(fam).values()):
            continue
        out.append(f"{_FAMILY_NAMES.get(label, label)}: every test promising under any rule, "
                   "plus TECH_SCORE at every horizon.\n")
        out.append(_table(robust_rows(fam), _ROBUST_COLS))
    m = max(len(f) for f in fams.values())
    floor = 1.0 / (1.0 + ROBUST_BOOT_N)
    out.append(f"Resolution of the bootstrap p-value: with {ROBUST_BOOT_N} draws p boot ≥ "
               f"1 / {ROBUST_BOOT_N + 1} = {floor:.5f}, so in a family of {m} tests the "
               f"smallest attainable q boot is {m} / {ROBUST_BOOT_N + 1} = {m * floor:.3f}, and "
               f"q boot < {PROMISING_Q:.2f} for the top-ranked test needs p boot < "
               f"{PROMISING_Q / m:.5f}, i.e. **no** draw as extreme as the observed mean "
               f"(one such draw gives q {2 * m * floor:.3f}). For the strongest test the boot "
               "verdict is therefore a 0-versus-1 count out of "
               f"{ROBUST_BOOT_N} draws; read it as \"at the resolution limit\", not as evidence "
               "against the test. The draw count is pre-registered and was not changed.\n")
    out.append("**The verdict rule is unchanged**: `promising` (and every section above and "
               "below) still uses `ic_q` from the fixed lag h − 1. The auto and boot columns are a "
               "pre-registered robustness check, not a new rule; switching the rule is a separate "
               "decision.\n")
    if info is not None and info.robust_s == info.robust_s:
        out.append(f"Runtime of the robust step: {info.robust_s:.1f} s for "
                   f"{info.robust_n_tests} tests (both universes, per-date IC series only).\n")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────
# Survivorship split (task 420)
# ──────────────────────────────────────────────────────────

def _same_sign(pre, post) -> str:
    if not (np.isfinite(pre) and np.isfinite(post)):
        return "–"
    return "yes" if np.sign(pre) == np.sign(post) else "no"


def survivorship_rows(head: pd.DataFrame) -> pd.DataFrame:
    """Promising headline rows (by q) followed by TECH_SCORE at every horizon, with the
    "same sign" column."""
    prom = head[head["promising"].astype(bool)].sort_values(["ic_q", "signal", "horizon"])
    tech = head[(head["signal"] == "TECH_SCORE") & ~head.index.isin(prom.index)]
    sel = pd.concat([prom, tech.sort_values("horizon")])
    return sel.assign(_same_sign=[_same_sign(a, b) for a, b in
                                  zip(sel["ic_mean_pre_complete"], sel["ic_mean_post_complete"])])


_SURVIVORSHIP_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    ("mean IC before", "ic_mean_pre_complete", _f), ("dates before", "n_dates_pre_complete", _int),
    ("mean IC from", "ic_mean_post_complete", _f), ("dates from", "n_dates_post_complete", _int),
    ("same sign", "_same_sign", _s),
]


def held_split_rows(head: pd.DataFrame) -> pd.DataFrame:
    """``survivorship_rows`` followed by every other headline row whose full-window held CAGR
    net is above the universe's (de-duplicated, in that order)."""
    base = survivorship_rows(head)
    beats = head[head["held_cagr_net"] > head["held_cagr_universe"]]
    beats = beats[~beats.index.isin(base.index)].sort_values(["signal", "horizon"])
    return pd.concat([base, beats])


_HELD_SPLIT_COLS = [
    ("signal", "signal", _s), ("h", "horizon", _int),
    ("periods before", "n_periods_pre_complete", _int),
    ("CAGR net before", "cagr_net_pre_complete", _pct),
    ("universe before", "cagr_universe_pre_complete", _pct),
    ("excess before", "excess_cagr_pre_complete", _pct),
    ("IR before", "info_ratio_net_pre_complete", lambda x: _f(x, "{:+.2f}")),
    ("periods from", "n_periods_post_complete", _int),
    ("CAGR net from", "cagr_net_post_complete", _pct),
    ("universe from", "cagr_universe_post_complete", _pct),
    ("excess from", "excess_cagr_post_complete", _pct),
    ("IR from", "info_ratio_net_post_complete", lambda x: _f(x, "{:+.2f}")),
]


def survivorship_section(table: pd.DataFrame, info: RunInfo) -> str:
    """"Survivorship split (descriptive)": mean IC before / from ``COMPLETE_FROM``."""
    cut = info.complete_from or COMPLETE_FROM
    out = ["## Survivorship split (descriptive)\n"]
    head = headline(table)
    if not all(k in head.columns for k in SURVIVORSHIP_KEYS) or head.empty:
        out.append("The survivorship split was not computed for this table.\n")
        return "\n".join(out)
    out.append(
        f"Mean IC of the promising tests and of TECH_SCORE, on the OOS dates before {cut} "
        f"(\"before\") and on or after it (\"from\"), with the number of dates on each side. "
        f"Dates before {cut} use today's constituent list (survivorship bias), so a mean IC "
        "there can be inflated; no test is run and no verdict uses these columns.\n")
    out.append(_table(survivorship_rows(head), _SURVIVORSHIP_COLS))
    if all(k in head.columns for k in HELD_SPLIT_KEYS):
        out.append(f"### Held portfolio before / from {cut} (descriptive)\n")
        out.append(
            "The same rows plus every headline row whose held CAGR net over the whole OOS window "
            "is above the equal-weight universe's. Net CAGR, universe CAGR, their difference and "
            "the net information ratio vs the universe, on the held-portfolio periods with a "
            f"rebalance date before {cut} (\"before\") and on or after it (\"from\"); periods "
            "are assigned by rebalance date, so one that straddles the cut counts as \"before\". "
            "The \"before\" universe is survivorship-inflated; no test is run and no verdict "
            "uses these columns.\n")
        out.append(_table(held_split_rows(head), _HELD_SPLIT_COLS))
    return "\n".join(out)


def inverted_section(table: pd.DataFrame) -> str:
    """Descriptive held-portfolio view of promising-inverted signals (negated signal)."""
    inv = table[(table["label"] == "open") & (table["subset"] == INVERTED_SUBSET)]
    out = ["## Promising signals, inverted direction (descriptive)\n"]
    if inv.empty:
        out.append("No promising signal has an inverted IC in this run, so there are no "
                   "inverted-direction rows.\n")
        return "\n".join(out)
    inv = inv.sort_values(["signal", "horizon"])
    out.append("Each row is a headline test that is promising with a **negative** IC, re-run on "
               "the negated signal (same labels, universe masking, splits, costs and bootstrap): "
               "its top bucket is the original lowest non-empty bucket.\n")
    out.append("Cost check (open→close):\n")
    out.append(_table(inv, _COST_COLS))
    out.append("Held portfolio (open-to-open, compounded):\n")
    out.append(_table(with_dd_days_text(inv), _HELD_COLS))
    out.append("Bootstrap intervals:\n")
    out.append(_table(with_interval_text(inv), _BOOT_COLS))
    out.append("**Caveat.** These rows are chosen *because* their IC was significant with the "
               "wrong sign, so their portfolio numbers are conditional on that selection: they "
               "are not a test and carry no q-value, and the bootstrap intervals do not account "
               "for the selection. The production polarity is unchanged.\n")
    return "\n".join(out)


def render_report(table: pd.DataFrame, splits: list[Split], info: RunInfo,
                  report_date: date, compare: pd.DataFrame | None = None) -> str:
    """``compare``: the "active"-universe table on the same splits (adds "Universe effect")."""
    head = headline(table)
    horizons = info.horizons
    families = [c for c in SCORE_COLUMNS if c != "TECH_SCORE"]
    out: list[str] = []
    w = out.append

    w(f"# Evaluation report {report_date:%Y-%m-%d}\n")
    w(f"> {universe_note(info)}\n")
    if dividend_warning(info):
        w(f"> {dividend_warning(info)}\n")
    w("Generated by `python main.py evaluate` (`investbot/evaluation/`). Nothing is fitted: "
      "every signal is evaluated with its designed polarity on walk-forward test blocks only.\n")

    w("## Setup\n")
    oos = [s.test for s in splits]
    for line in _setup_universe_lines(info):
        w(line)
    if oos:
        w(f"- Out-of-sample (OOS): {len(splits)} test blocks of {TEST_MONTHS} months, "
          f"{oos[0][0]:%Y-%m-%d} → {oos[-1][-1]:%Y-%m-%d}, {sum(len(t) for t in oos)} dates. "
          f"Expanding train window (≥ {MIN_TRAIN_BARS} bars from the first date with "
          f"TECH_SCORE on ≥ {MIN_NAMES} names), embargo of {max(horizons)} bars before each block.")
    if info.primary_return == "tr":
        w(f"- Labels (total return, dividends credited on their ex-date bars, not reinvested "
          f"within a period, untaxed): `FWD_TR_RET_h = (close[t+h] + D[t+2..t+h]) / open[t+1] − 1` "
          f"(signal known after t's close, entry at the next open, so a dividend whose ex-date "
          f"is the entry bar is not earned), h ∈ {{{', '.join(map(str, horizons))}}}. "
          "Close-to-close `FWD_TR_CC_RET_h = (close[t+h] + D[t+1..t+h]) / close[t] − 1` for "
          "comparison only. The held portfolio uses `FWD_TR_OO_RET_h = (open[t+1+h] + "
          "D[t+2..t+1+h]) / open[t+1] − 1` (open-to-open, so periods tile the timeline; an exit "
          "at an ex-date's open earns that dividend). D[a..b] = dividends per share on stored "
          "bars a..b. The price-only labels `FWD_RET_h = close[t+h] / open[t+1] − 1` and "
          "`FWD_OO_RET_h` are in the \"Dividend effect\" table only.")
    else:
        w(f"- Labels: `FWD_RET_h = close[t+h] / open[t+1] − 1` (signal known after t's close, "
          f"entry at the next open), h ∈ {{{', '.join(map(str, horizons))}}}. "
          "Close-to-close shown for comparison only. The held portfolio uses "
          "`FWD_OO_RET_h = open[t+1+h] / open[t+1] − 1` (open-to-open, so periods tile the "
          "timeline).")
    for line in _setup_label_lines(info):
        w(line)
    w("- IC: daily cross-sectional Spearman; ICIR = mean / sd of daily IC (not annualised); "
      "t-stats use Newey-West standard errors with lag h − 1.")
    w(f"- Multiple testing: Benjamini–Hochberg over all {len(head)} signal × horizon tests. "
      f"**Promising** = BH q < {PROMISING_Q:.2f} and the IC sign holds in ≥ "
      f"{PROMISING_SIGN_SHARE:.0%} of test blocks.")
    w(f"- Risk-free rate for Sharpe and Sortino: {RISK_FREE_RATE_ANNUAL:.2%} a year, a constant "
      "placeholder for the 91-day T-bill (master plan §9.1), not a dated series. VaR/CVaR are "
      "per holding period of h bars.")
    w(f"- Bootstrap intervals (held portfolio): stationary block bootstrap, {BOOTSTRAP_N} "
      f"resamples, {BOOTSTRAP_LEVEL:.0%} percentile intervals, seed {BOOTSTRAP_SEED} (the same "
      "for every row), mean block length max(1, round(n^(1/3))) periods, pre-registered (no "
      f"Politis–White selection), none below {BOOTSTRAP_MIN_PERIODS} periods.")
    w(f"- Costs: round trip {ROUND_TRIP_COST:.3%} of traded value = statutory charges and "
      f"brokerage from `config/tax_regimes.py` on a ₹{REFERENCE_TRADE_VALUE / 1e5:g} lakh order "
      f"({statutory_round_trip_cost():.3%}) + slippage {SLIPPAGE_PER_SIDE:.3%} per side.")
    if info.elapsed_s == info.elapsed_s:
        w(f"- Runtime: {info.elapsed_s:.0f} s"
          + (f"; peak RSS {info.peak_rss_mb:.0f} MB" if info.peak_rss_mb == info.peak_rss_mb
             else "") + " (features computed one asset at a time, once for both universes).")
    if info.variants_s == info.variants_s:
        share = (info.n_variant_columns / info.n_signal_columns if info.n_signal_columns
                 else math.nan)
        w(f"- Added runtime of the {info.n_variant_columns} polarity-variant columns: "
          f"{info.variants_s:.1f} s to compute them (signal and aggregation steps only, no "
          f"indicator recompute), plus about {info.evaluate_s * share:.0f} s of evaluation "
          f"(their {share:.0%} share of the {info.n_signal_columns} evaluated columns × "
          f"{info.evaluate_s:.0f} s in `evaluate`, both universes; an estimate, not timed per "
          "column).")
    w("")

    w("## Verdict\n")
    for line in verdict_lines(head, horizons):
        w(f"- {line}")
    w("")

    w(robust_section(table, info))

    w(survivorship_section(table, info))

    w(f"## Top {TOP_N} signals by |ICIR| (OOS, next-open {_label_name(info)} labels)\n")
    for h in horizons:
        hh = head[head["horizon"] == h]
        top = hh.reindex(hh["icir"].abs().sort_values(ascending=False).index).head(TOP_N)
        w(f"### h = {h}\n")
        w(_table(top, _IC_COLS))

    w("## Baselines (OOS)\n")
    w(_table(_order(head, list(BASELINE_COLUMNS)), _IC_COLS))

    w("## TECH_SCORE and family scores (OOS)\n")
    w(_table(_order(head, ["TECH_SCORE", *families]), _IC_COLS))
    w("TECH_SCORE_MTF is absent: daily-only frames have no multi-timeframe family.\n")

    w("## Quintile portfolios (OOS)\n")
    w("Top/bottom are the highest/lowest non-empty quintile of each date "
      "(Q5/Q1 for continuous signals; discrete signals fill fewer buckets).\n")
    w(_table(_order(head, ["TECH_SCORE", *families, *BASELINE_COLUMNS]), _Q_COLS))

    w("## Cost reality check: top-quintile long-only, open→close per period (OOS)\n")
    w(f"Equal weight, rebalanced every h bars; net = gross − turnover × {ROUND_TRIP_COST:.3%}. "
      "Universe = equal weight over all names in the same periods. Simple annualisation "
      "(mean period return × 252 / h); not compounded, no taxes. Each period runs from open[t+1] "
      "to close[t+h], so the overnight gap before the next entry is not held: absolute "
      "returns at short horizons are mostly the (negative) open-to-close drift. Compare "
      "against the universe column, not against zero.\n")
    promising = head[head["promising"]]["signal"].unique().tolist()
    portfolio_names = ["TECH_SCORE", *families, *BASELINE_COLUMNS,
                       *[p for p in promising if p not in SCORE_COLUMNS
                         and p not in BASELINE_COLUMNS]]
    w(_table(_order(head, portfolio_names), _COST_COLS))

    w("## Held portfolio (open-to-open, compounded)\n")
    w("The same top-quintile equal-weight portfolio, rebalanced every h bars, but held "
      "continuously: each period runs open[t+1] → open[t+1+h], so consecutive periods tile the "
      f"OOS timeline and the overnight gap is included. Net = gross − turnover × "
      f"{ROUND_TRIP_COST:.3%} per period, compounded. CAGR over the invested periods "
      "(n × h / 252 years); vol = sd of net period returns × √(252 / h); max DD = worst "
      "peak-to-trough of the compounded net curve; IR = mean / sd of (net − universe) period "
      "returns × √(252 / h). CAGR, vol and IR use no risk-free rate: the benchmark is the "
      "equal-weight universe over the same periods. The OOS blocks are contiguous, so the curve spans the whole OOS window. "
      "No taxes.\n")
    w(_table(with_dd_days_text(_order(head, portfolio_names)), _HELD_COLS))
    w("Risk-adjusted columns use the periods above: Sharpe and Sortino are annualised "
      "(× √(252 / h)) against the risk-free rate in the Setup (Sortino's downside target is that "
      "rate), Calmar = CAGR net / |max DD|, max DD days = bars from the peak before the worst "
      "drawdown until the curve is back at it (`*` = not recovered by the end of the window, "
      "counted to the end), CVaR95 = mean loss of the worst 5% of holding periods (a loss over "
      "h bars, not a daily one). These are cost-only signal portfolios without taxes: the §9.2 "
      "go/no-go thresholds apply to the full after-tax strategy, so no verdict is given here.\n")

    w(f"## Held portfolio: bootstrap intervals ({BOOTSTRAP_LEVEL:.0%}, stationary block "
      "bootstrap)\n")
    w(_table(with_interval_text(_order(head, portfolio_names)), _BOOT_COLS))
    w(f"Each row resamples its own held-portfolio periods (net and universe jointly, the "
      f"same indices for both) with a stationary block bootstrap (Politis & Romano 1994; "
      f"{BOOTSTRAP_N} resamples, mean block = round(n^(1/3)) periods, circular), seed "
      f"{BOOTSTRAP_SEED} for every row; percentile intervals, \"–\" with fewer than "
      f"{BOOTSTRAP_MIN_PERIODS} periods. The intervals are **per row and not BH-adjusted**: "
      f"with this many rows, about {1 - BOOTSTRAP_LEVEL:.0%} of the intervals exclude 0 by "
      "chance even when nothing works, so the Benjamini–Hochberg verdict above stays the "
      "headline. Calmar intervals at h = 20 rest on about 34 periods and are descriptive.\n")

    w(inverted_section(table))

    w(dividend_section(table, info, ["TECH_SCORE", *families, *BASELINE_COLUMNS]))

    w(variants_section(head, horizons, info))

    if compare is not None and not compare.empty:
        w("## Universe effect: point-in-time index vs today's active stocks (OOS)\n")
        w("Same walk-forward splits and labels. *index* masks every signal and the universe "
          "benchmark to the members on each date; *active* scores today's active stocks on every "
          "date. Held net − univ. = held-portfolio CAGR net of costs minus the equal-weight "
          "universe CAGR over the same periods.\n")
        prom = sorted(set(head.loc[head["promising"], "signal"])
                      | set(headline(compare).loc[lambda d: d["promising"], "signal"]))
        w(_table(universe_effect(table, compare,
                                 ["TECH_SCORE", *families, *BASELINE_COLUMNS,
                                  *[p for p in prom if p not in SCORE_COLUMNS
                                    and p not in BASELINE_COLUMNS]]), _EFFECT_COLS))
        flips = _significance_changes(table, compare)
        if not flips.empty:
            w("Tests whose BH significance (q < "
              f"{PROMISING_Q:.2f}) or promising flag differs between the universes:\n")
            w(_table(flips, [("signal", "signal", _s), ("h", "horizon", _int),
                             ("IC index", "ic_mean_index", _f), ("q index", "ic_q_index",
                                                                 lambda x: _f(x, "{:.3f}")),
                             ("promising index", "promising_index", _f),
                             ("IC active", "ic_mean_active", _f), ("q active", "ic_q_active",
                                                                   lambda x: _f(x, "{:.3f}")),
                             ("promising active", "promising_active", _f)]))
        else:
            w("No test changes BH significance or promising flag between the universes.\n")

    w("## By regime (OOS)\n")
    w(f"Cross-section restricted to names in that regime on that date. Daily IC and quintiles "
      f"need ≥ {MIN_NAMES} names on a date, so chaotic is mostly empty. The **pooled IC** uses "
      f"every date with ≥ {POOLED_MIN_NAMES} names: per-date ranks scaled to [−0.5, 0.5], "
      "pooled over all (date, name) rows (larger cross-sections weigh more); its t-stat is "
      "Newey-West (lag h − 1) on the per-date sums.\n")
    reg = table[(table["label"] == "open") & table["subset"].str.startswith("regime:")]
    w(_table(_order(reg, ["TECH_SCORE", *families, *BASELINE_COLUMNS]), _REGIME_COLS))

    w("## By year (OOS)\n")
    yr = table[(table["label"] == "open") & table["subset"].str.startswith("year:")]
    w(_table(_order(yr, ["TECH_SCORE", *BASELINE_COLUMNS]), _SPLIT_COLS))

    w("## Walk-forward blocks: mean IC per test block\n")
    blk = table[(table["label"] == "open") & table["subset"].str.startswith("block:")]
    names = ["TECH_SCORE", *families, *BASELINE_COLUMNS]
    if not blk.empty:
        piv = blk[blk["signal"].isin(names)].pivot_table(
            index=["signal", "horizon"], columns="subset", values="ic_mean", sort=False)
        cols = [c for c in piv.columns]
        lines = ["| signal | h | " + " | ".join(c.removeprefix("block:") for c in cols) + " |",
                 "|" + "---|" * (len(cols) + 2)]
        for n in names:
            for h in horizons:
                if (n, h) in piv.index:
                    lines.append(f"| {n} | {h} | " + " | ".join(_f(v) for v in piv.loc[(n, h)]) + " |")
        w("\n".join(lines) + "\n")

    w("## Next-open vs close-to-close labels (OOS mean IC)\n")
    w("The close-to-close label assumes a fill at the signal bar's close, which is not tradable. "
      "A large gap between the two means the edge sits in the overnight move.\n")
    cc = table[(table["label"] == "close") & (table["subset"] == "oos")]
    both = head.merge(cc[["signal", "horizon", "ic_mean", "icir"]], on=["signal", "horizon"],
                      suffixes=("", "_cc"))
    w(_table(_order(both, ["TECH_SCORE", *families, *BASELINE_COLUMNS]), [
        ("signal", "signal", _s), ("h", "horizon", _int),
        ("IC next-open", "ic_mean", _f), ("IC close-close", "ic_mean_cc", _f),
        ("ICIR next-open", "icir", _f), ("ICIR close-close", "icir_cc", _f)]))

    w(f"## Benjamini–Hochberg: tests with q < {PROMISING_Q:.2f}\n")
    sig = head[head["ic_q"] < PROMISING_Q].sort_values("ic_q")
    if sig.empty:
        best = head.sort_values("ic_p").head(10)
        w(f"None of the {len(head)} tests survives. The 10 smallest raw p-values:\n")
        w(_table(best, _IC_COLS))
    else:
        w(_table(sig, _IC_COLS))

    w("## Caveats\n")
    w(f"- {universe_note(info)}")
    w(f"- {history_caveat(splits, info)}")
    w("- A missing session (purged bar) is skipped, not filled: t+1 is the next *stored* bar.")
    w("- Total return credits gross cash dividends; dividend tax (slab rate) is not modelled, "
      "and neither is STCG/LTCG on the trades.")
    w("- A signal significant only with its sign inverted is a finding about polarity, "
      "not something to switch on: polarity decisions go to Ankit with this evidence.")
    return "\n".join(out) + "\n"


def _significance_changes(index_table: pd.DataFrame, active_table: pd.DataFrame) -> pd.DataFrame:
    cols = ["signal", "horizon", "ic_mean", "ic_q", "promising"]
    m = headline(index_table)[cols].merge(headline(active_table)[cols], on=["signal", "horizon"],
                                          suffixes=("_index", "_active"))
    sig_i = m["ic_q_index"] < PROMISING_Q
    sig_a = m["ic_q_active"] < PROMISING_Q
    diff = (sig_i != sig_a) | (m["promising_index"].astype(bool) != m["promising_active"].astype(bool))
    return m[diff].sort_values(["signal", "horizon"])


def report_paths(report_dir: Path, csv_dir: Path, report_date: date) -> tuple[Path, Path]:
    """``eval_<YYYYMMDD>.md`` and its CSV; ``_b``, ``_c``, ... if that report already exists."""
    stem = f"{report_date:%Y%m%d}"
    for suffix in ["", *(f"_{chr(c)}" for c in range(ord("b"), ord("z") + 1))]:
        report = report_dir / f"eval_{stem}{suffix}.md"
        if not report.exists():
            return report, csv_dir / f"eval_metrics_{stem}{suffix}.csv"
    raise FileExistsError(f"too many reports for {stem} in {report_dir}")


def active_subset(features: pd.DataFrame, labels: pd.DataFrame,
                  active: list[str] | set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The "active" dataset from an "index" one: active tickers only, no ``IN_UNIVERSE``.
    Features are full-history in both modes, so this equals ``build_dataset(universe="active")``
    for every active ticker present in the index panel."""
    keep = features.index.get_level_values("ticker").isin(list(active))
    return features.loc[keep].drop(columns=[UNIVERSE_COLUMN], errors="ignore"), labels.loc[keep]


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far (Linux reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _coverage_info(session, index_code: str, start, end, features: pd.DataFrame,
                   oos: pd.DatetimeIndex, active: list[str], complete_from: date) -> dict:
    from investbot.evaluation.loader import membership_for, tickers_with_prices, universe_tickers
    members_all = universe_tickers(session, index_code, start, end)
    member = point_in_time_membership(membership_for(session, index_code, members_all, oos),
                                      active, complete_from)
    score = features["TECH_SCORE"].unstack("ticker")
    cov = universe_coverage(member, score.reindex(index=oos))
    in_panel = set(features.index.get_level_values("ticker"))
    priced = tickers_with_prices(session, members_all)
    oos_days = member.sum(axis=0)
    dates = features.index.get_level_values("date")
    # dates the metrics skip anyway (< MIN_NAMES names with a value, e.g. a purged session
    # where only a few assets have a bar) would read as a coverage collapse; leave them out
    evaluated = cov["covered"] >= MIN_NAMES
    c = cov.loc[evaluated, "coverage"]
    return {
        "n_members": len(members_all),
        "coverage_mean": float(c.mean()) if c.notna().any() else math.nan,
        "coverage_min": float(c.min()) if c.notna().any() else math.nan,
        "coverage_min_date": c.idxmin() if c.notna().any() else None,
        "missing_prices": {t: int(oos_days[t]) for t in members_all if t not in priced},
        "short_history": {t: int(oos_days[t]) for t in members_all
                          if t in priced and t not in in_panel},
        "fallback_dates": int(pd.DatetimeIndex(dates.unique()).to_series()
                              .lt(pd.Timestamp(complete_from)).sum()),
        "fallback_oos_dates": int((oos < pd.Timestamp(complete_from)).sum()),
        "complete_from": complete_from,
        "coverage_skipped_dates": int((~evaluated).sum()),
    }


def label_summary(label_info: dict[str, dict], tickers: set[str]) -> dict:
    """RunInfo fields from ``build_dataset``'s ``label_info``, over the headline ``tickers``."""
    rows = {t: v for t, v in label_info.items() if t in tickers}
    return {
        "n_dividend_names": sum(v["dividend_rows"] > 0 for v in rows.values()),
        "n_dividend_rows": sum(v["dividend_rows"] for v in rows.values()),
        "split_issues": sorted((t, i.ex_date, i.factor, n) for t, v in rows.items()
                               for i, n in v["splits"]),
    }


def run_evaluation(
    horizons=HORIZONS,
    start: date | None = None,
    end: date | None = None,
    *,
    universe: str = "index",
    index_code: str = "NIFTY50",
    session=None,
    report_dir: Path = REPORT_DIR,
    csv_dir: Path = CSV_DIR,
    report_date: date | None = None,
) -> tuple[Path, Path, pd.DataFrame, RunInfo]:
    """Build the panel, evaluate, write the report and CSV. Returns (report, csv, table, info).

    ``universe="index"`` also evaluates the "active" universe on the same splits
    (the "Universe effect" table); ``table`` is the headline universe's metrics.
    """
    from investbot.evaluation.loader import active_tickers

    from investbot.evaluation.loader import db_fingerprint
    from investbot.scoring.technical.scorer import SCORER_VERSION

    t0 = time.monotonic()
    timings: dict[str, float] = {}
    label_info: dict[str, dict] = {}
    horizons = tuple(sorted(int(h) for h in horizons))
    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    try:
        features, labels = build_dataset(start=start, end=end, horizons=horizons, session=session,
                                         universe=universe, index_code=index_code,
                                         variants=True, timings=timings,
                                         label_info=label_info)
        if features.empty:
            raise RuntimeError("no assets with enough daily bars to evaluate")
        t_eval = time.monotonic()
        table, splits = evaluate(features, labels, horizons, timings=timings)
        eval_s = time.monotonic() - t_eval
        if table.empty:
            raise RuntimeError("evaluation produced no metrics (too little history?)")
        active = active_tickers(session)
        extra: dict = {}
        compare = None
        if universe == "index":
            missing = [t for t in active
                       if t not in set(features.index.get_level_values("ticker"))]
            fa, la = active_subset(features, labels, active)
            if missing:      # active stocks outside the index universe (none for NIFTY50)
                fm, lm = build_dataset(missing, start, end, horizons=horizons, session=session,
                                       universe="active", variants=True, timings=timings,
                                       label_info=label_info)
                fa, la = pd.concat([fa, fm]).sort_index(), pd.concat([la, lm]).sort_index()
            t_eval = time.monotonic()
            compare, _ = evaluate(fa, la, horizons, splits=splits, timings=timings)
            eval_s += time.monotonic() - t_eval
            extra = _coverage_info(session, index_code, start, end, features,
                                   oos_dates(splits), active, COMPLETE_FROM)
            extra["n_active"] = fa.index.get_level_values("ticker").nunique()
            del fa, la
        extra.update(db_fingerprint(session))
    finally:
        if own:
            session.close()
    dates = features.index.get_level_values("date")
    sig_cols = signal_columns(features)
    extra.update(variants_s=timings.get(VARIANT_TIMING_KEY, 0.0), evaluate_s=eval_s,
                 n_signal_columns=len(sig_cols),
                 n_variant_columns=sum(c in VARIANT_COLUMNS for c in sig_cols),
                 **label_summary(label_info, set(features.index.get_level_values("ticker"))),
                 scorer_version=SCORER_VERSION, git_commit=git_commit(),
                 robust_s=timings.get(ROBUST_TIMING_KEY, math.nan),
                 names_median=oos_names_median(features, oos_dates(splits)),
                 robust_n_tests=sum(int(t["ic_lag_auto"].notna().sum())
                                    for t in (table, compare) if t is not None))
    info = RunInfo(n_assets=features.index.get_level_values("ticker").nunique(),
                   first_date=dates.min(), last_date=dates.max(), horizons=horizons,
                   start=start, elapsed_s=time.monotonic() - t0, universe=universe,
                   index_code=index_code, peak_rss_mb=peak_rss_mb(), **extra)
    report_date = report_date or date.today()
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    report, csv = report_paths(report_dir, csv_dir, report_date)
    out = table.assign(universe=universe)
    if compare is not None:
        out = pd.concat([out, compare.assign(universe="active")], ignore_index=True)
    # the robust (task 340), survivorship (task 420) and held-split (task 470) columns go last,
    # so every earlier column keeps its position in the CSV
    tail = [*ROBUST_COLUMNS, *SURVIVORSHIP_KEYS, *HELD_SPLIT_KEYS]
    out = out[[c for c in out.columns if c not in tail] + [c for c in tail if c in out.columns]]
    out.to_csv(csv, index=False)
    report.write_text(render_report(table, splits, info, report_date, compare), encoding="utf-8")
    return report, csv, table, info
