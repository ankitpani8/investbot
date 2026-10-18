"""
Markdown report and CSV for one evaluation run (``python main.py evaluate``).

``run_evaluation`` builds the panel from the DB (one asset at a time),
evaluates it and writes

    docs/reports/eval_<YYYYMMDD>.md        the report (committed)
    tmp_output/eval_metrics_<YYYYMMDD>.csv every metrics row (git-ignored)

An existing report is never overwritten: a second run on the same day writes
``eval_<YYYYMMDD>_b.md`` (then ``_c``, ...).

``render_report`` is pure: it only formats a metrics table.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from investbot.evaluation.baselines import BASELINE_COLUMNS
from investbot.evaluation.evaluate import (
    PROMISING_Q,
    PROMISING_SIGN_SHARE,
    evaluate,
    headline,
)
from investbot.evaluation.labels import HORIZONS
from investbot.evaluation.metrics import (
    MIN_NAMES,
    POOLED_MIN_NAMES,
    REFERENCE_TRADE_VALUE,
    ROUND_TRIP_COST,
    SLIPPAGE_PER_SIDE,
    statutory_round_trip_cost,
)
from investbot.evaluation.panel import SCORE_COLUMNS, build_dataset
from investbot.evaluation.walkforward import MIN_TRAIN_BARS, TEST_MONTHS, Split

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "docs" / "reports"
CSV_DIR = REPO_ROOT / "tmp_output"
TOP_N = 15

SURVIVORSHIP_CAVEAT = (
    "**Survivorship bias.** The universe is *today's* NIFTY50 constituents, back-filled over the "
    "whole history. Names that were dropped from the index (usually after underperforming) are "
    "missing and names that were added (usually after outperforming) are present before they "
    "joined. Absolute returns, and any signal correlated with past winners, are biased upward. "
    "Historical constituents are an open data task in docs/BACKLOG.md."
)


@dataclass
class RunInfo:
    n_assets: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    horizons: tuple[int, ...]
    start: date | None
    elapsed_s: float = math.nan


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
# Report
# ──────────────────────────────────────────────────────────

def render_report(table: pd.DataFrame, splits: list[Split], info: RunInfo,
                  report_date: date) -> str:
    head = headline(table)
    horizons = info.horizons
    families = [c for c in SCORE_COLUMNS if c != "TECH_SCORE"]
    out: list[str] = []
    w = out.append

    w(f"# Evaluation report {report_date:%Y-%m-%d}\n")
    w(f"> {SURVIVORSHIP_CAVEAT}\n")
    w("Generated by `python main.py evaluate` (`investbot/evaluation/`). Nothing is fitted: "
      "every signal is evaluated with its designed polarity on walk-forward test blocks only.\n")

    w("## Setup\n")
    oos = [s.test for s in splits]
    w(f"- Universe: {info.n_assets} active stocks; bars {info.first_date:%Y-%m-%d} → "
      f"{info.last_date:%Y-%m-%d}" + (f"; panel from {info.start}" if info.start else "") + ".")
    if oos:
        w(f"- Out-of-sample (OOS): {len(splits)} test blocks of {TEST_MONTHS} months, "
          f"{oos[0][0]:%Y-%m-%d} → {oos[-1][-1]:%Y-%m-%d}, {sum(len(t) for t in oos)} dates. "
          f"Expanding train window (≥ {MIN_TRAIN_BARS} bars from the first date with "
          f"TECH_SCORE on ≥ {MIN_NAMES} names), embargo of {max(horizons)} bars before each block.")
    w(f"- Labels: `FWD_RET_h = close[t+h] / open[t+1] − 1` (signal known after t's close, "
      f"entry at the next open), h ∈ {{{', '.join(map(str, horizons))}}}. "
      "Close-to-close shown for comparison only. The held portfolio uses "
      "`FWD_OO_RET_h = open[t+1+h] / open[t+1] − 1` (open-to-open, so periods tile the timeline).")
    w("- IC: daily cross-sectional Spearman; ICIR = mean / sd of daily IC (not annualised); "
      "t-stats use Newey-West standard errors with lag h − 1.")
    w(f"- Multiple testing: Benjamini–Hochberg over all {len(head)} signal × horizon tests. "
      f"**Promising** = BH q < {PROMISING_Q:.2f} and the IC sign holds in ≥ "
      f"{PROMISING_SIGN_SHARE:.0%} of test blocks.")
    w(f"- Costs: round trip {ROUND_TRIP_COST:.3%} of traded value = statutory charges and "
      f"brokerage from `config/tax_regimes.py` on a ₹{REFERENCE_TRADE_VALUE / 1e5:g} lakh order "
      f"({statutory_round_trip_cost():.3%}) + slippage {SLIPPAGE_PER_SIDE:.3%} per side.")
    if info.elapsed_s == info.elapsed_s:
        w(f"- Runtime: {info.elapsed_s:.0f} s.")
    w("")

    w("## Verdict\n")
    for line in verdict_lines(head, horizons):
        w(f"- {line}")
    w("")

    w(f"## Top {TOP_N} signals by |ICIR| (OOS, next-open labels)\n")
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
      "returns × √(252 / h). No risk-free rate: the benchmark is the equal-weight universe over "
      "the same periods. The OOS blocks are contiguous, so the curve spans the whole OOS window. "
      "No taxes.\n")
    w(_table(_order(head, portfolio_names), _HELD_COLS))

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
    w(f"- {SURVIVORSHIP_CAVEAT}")
    w("- Five years of daily history leave about three years out of sample after warmup "
      "(EMA_200, regime window) and the first training year; ~50 names per date.")
    w("- A missing session (purged bar) is skipped, not filled: t+1 is the next *stored* bar.")
    w("- A signal significant only with its sign inverted is a finding about polarity, "
      "not something to switch on: polarity decisions go to Ankit with this evidence.")
    return "\n".join(out) + "\n"


def report_paths(report_dir: Path, csv_dir: Path, report_date: date) -> tuple[Path, Path]:
    """``eval_<YYYYMMDD>.md`` and its CSV; ``_b``, ``_c``, ... if that report already exists."""
    stem = f"{report_date:%Y%m%d}"
    for suffix in ["", *(f"_{chr(c)}" for c in range(ord("b"), ord("z") + 1))]:
        report = report_dir / f"eval_{stem}{suffix}.md"
        if not report.exists():
            return report, csv_dir / f"eval_metrics_{stem}{suffix}.csv"
    raise FileExistsError(f"too many reports for {stem} in {report_dir}")


def run_evaluation(
    horizons=HORIZONS,
    start: date | None = None,
    end: date | None = None,
    *,
    session=None,
    report_dir: Path = REPORT_DIR,
    csv_dir: Path = CSV_DIR,
    report_date: date | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build the panel, evaluate, write the report and CSV. Returns (report, csv, table)."""
    t0 = time.monotonic()
    horizons = tuple(sorted(int(h) for h in horizons))
    features, labels = build_dataset(start=start, end=end, horizons=horizons, session=session)
    if features.empty:
        raise RuntimeError("no assets with enough daily bars to evaluate")
    table, splits = evaluate(features, labels, horizons)
    if table.empty:
        raise RuntimeError("evaluation produced no metrics (too little history?)")
    dates = features.index.get_level_values("date")
    info = RunInfo(n_assets=features.index.get_level_values("ticker").nunique(),
                   first_date=dates.min(), last_date=dates.max(), horizons=horizons,
                   start=start, elapsed_s=time.monotonic() - t0)
    report_date = report_date or date.today()
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    report, csv = report_paths(report_dir, csv_dir, report_date)
    table.to_csv(csv, index=False)
    report.write_text(render_report(table, splits, info, report_date), encoding="utf-8")
    return report, csv, table
