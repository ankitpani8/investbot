"""
Regime study runner: does the real-time regime label predict the ex-post regime?

    python -m scripts.regime_study [--end YYYY-MM-DD] [--universe index|active] [--no-h6]
                                   [--no-robust]

Loads the stocks one at a time (``loader.load_daily_history``), computes the
regime conditioning and the ex-post targets on the real bars and on the
shuffled-returns null (``regime_study.shuffled_null_bars``), keeps only the
float columns the study reads, runs ``regime_study.run_study`` on the
harness's out-of-sample dates and writes

    docs/reports/regime_study_<YYYYMMDD>.md          the report (committed)
    tmp_output/regime_study_tests_<YYYYMMDD>.csv     every tested row (git-ignored)
    tmp_output/regime_study_base_rates_<YYYYMMDD>.csv

Universe (``--universe``, default "index" when ``index_membership`` has rows
for the index, else "active" with a warning, as ``main.py evaluate``)
    "index"   every past or present member of the index up to ``end``
              (``loader.universe_tickers``), the rule of ``panel.build_dataset``:
              features are computed on each asset's full stored history, then
              every conditioning feature (real and null) is set to NaN on the
              dates when the asset was not a member (``point_in_time_membership``,
              changes effective <= t; before ``COMPLETE_FROM`` "active today").
              Ex-post targets are never masked: a target window may run past an
              exit, as the harness labels do. The "active" universe is studied on
              the same splits for the report's "Universe effect" table.
    "active"  today's active stocks, unmasked (the pre-260 study, with the
              survivorship caveat).

H6 (task 330, on unless ``--no-h6``): the shuffled nulls of
``regime_study.NULL_EXCESS_SEEDS[1:]`` are computed in the same one-asset-at-a-time
pass, only the H6 columns kept, features masked like the other null features;
the report gains the section "H6: real minus shuffled null (pre-registered)" and
``tmp_output/regime_study_h6_<YYYYMMDD>.csv``. Every other table is unchanged.

Robust inference (task 350, on unless ``--no-robust``): ``run_study(robust=True)``
appends ``rs.ROBUST_COLUMNS`` and ``rs.ROBUST_VERDICT_COLUMNS`` to the tests and H6
tables (and, last, to their CSVs); the report gains the section "Robust inference
(pre-registered)" after H6, with the pre-registered size check (``rs.size_check``,
run once per report). Every other table and column is unchanged.

An existing report is never overwritten (``_b``, ``_c``, ... suffixes).
No DB writes, no network.
"""
from __future__ import annotations

import argparse
import logging
import math
import resource
import subprocess
import sys
import time
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import REGIME_NAMES
from investbot.data.index_membership import COMPLETE_FROM
from investbot.evaluation import regime_study as rs
from investbot.evaluation.evaluate import PROMISING_Q, PROMISING_SIGN_SHARE
from investbot.evaluation.labels import HORIZONS as HARNESS_HORIZONS
from investbot.evaluation.metrics import MIN_NAMES
from investbot.evaluation.panel import UNIVERSES, point_in_time_membership, universe_coverage
from investbot.evaluation.report import SURVIVORSHIP_CAVEAT
from investbot.evaluation.walkforward import MIN_TRAIN_BARS, TEST_MONTHS
from investbot.scoring.technical import regime as R
from investbot.scoring.technical.scorer import SCORER_VERSION

logger = logging.getLogger("investbot.regime_study")

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "docs" / "reports"
CSV_DIR = REPO_ROOT / "tmp_output"
#: Fewer daily bars than this: no regime label can exist (VOL_LONG warmup) and the asset is skipped.
MIN_STUDY_BARS: int = R.VOL_LONG + 1
INDEX_CODE: str = "NIFTY50"


def report_paths(report_dir: Path, csv_dir: Path, report_date: date) -> tuple[Path, Path, Path]:
    """Report, tests CSV and base-rates CSV; ``_b``, ``_c``, ... if the report exists."""
    stem = f"{report_date:%Y%m%d}"
    for suffix in ["", *(f"_{chr(c)}" for c in range(ord("b"), ord("z") + 1))]:
        report = report_dir / f"regime_study_{stem}{suffix}.md"
        if not report.exists():
            return (report, csv_dir / f"regime_study_tests_{stem}{suffix}.csv",
                    csv_dir / f"regime_study_base_rates_{stem}{suffix}.csv")
    raise FileExistsError(f"too many regime-study reports for {stem} in {report_dir}")


def git_short_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # KiB on Linux


# ──────────────────────────────────────────────────────────
# Panel
# ──────────────────────────────────────────────────────────

def mask_non_members(frame: pd.DataFrame, member: np.ndarray) -> pd.DataFrame:
    """A copy of ``frame`` with every column NaN on the rows where ``member`` is False
    (boolean indexing; rows are kept, nothing is dropped)."""
    out = frame.copy()
    out.loc[~np.asarray(member, dtype=bool)] = np.nan
    return out


def _wide4(real: dict, null: dict) -> tuple[dict, dict, dict, dict]:
    return (rs.to_wide({t: f for t, (f, _) in real.items()}),
            rs.to_wide({t: y for t, (_, y) in real.items()}),
            rs.to_wide({t: f for t, (f, _) in null.items()}),
            rs.to_wide({t: y for t, (_, y) in null.items()}))


def extra_null_frames(bars: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, targets) of one asset's shuffled null with ``seed``, only the
    columns H6 reads (``rs.NULL_EXCESS_FEATURES`` / ``rs.NULL_EXCESS_TARGETS``)."""
    f, y = rs.asset_study_frames(rs.shuffled_null_bars(bars, seed=seed))
    return f[list(rs.NULL_EXCESS_FEATURES)], y[list(rs.NULL_EXCESS_TARGETS)]


def _wide_pairs(by_seed: dict[int, dict]) -> list[tuple[dict, dict]]:
    """{seed: {ticker: (f, y)}} -> [(features wide, targets wide)] in seed order."""
    return [(rs.to_wide({t: f for t, (f, _) in frames.items()}),
             rs.to_wide({t: y for t, (_, y) in frames.items()})) for frames in by_seed.values()]


def build_study_panels(session, end: date | None = None, *, universe: str = "index",
                       index_code: str = INDEX_CODE, complete_from: date = COMPLETE_FROM,
                       compare_active: bool = False, null_seeds: tuple[int, ...] = ()):
    """
    (features, targets, null features, null targets, info) as {column: date × ticker}.

    ``universe="active"``: today's active stocks, unmasked (bit for bit the
    pre-260 panels). ``universe="index"``: every member of ``index_code`` up to
    ``end``; features and null features NaN on non-member dates (see the module
    docstring), targets untouched. ``info["membership"]`` is then the
    point-in-time membership (union calendar × every member, with or without
    prices) and, with ``compare_active``, ``info["active_panels"]`` holds the
    four unmasked panels of today's active stocks (computed in the same pass).
    Assets with fewer than ``MIN_STUDY_BARS`` bars (or none) are skipped and
    listed; a member with no history is never filled.

    ``null_seeds`` (H6: ``NULL_EXCESS_SEEDS[1:]``): for each seed, the shuffled null
    of every asset in the same pass, only ``extra_null_frames`` columns, features
    masked like the null features (targets never). Then ``info["null_excess_panels"]``
    is [(features, targets) per seed] and, with ``compare_active``,
    ``info["active_null_excess_panels"]`` the unmasked ones of the active stocks;
    ``info["h6_s"]`` is the time spent on them. Without seeds ``info`` is unchanged.
    """
    from investbot.evaluation.loader import (
        active_tickers,
        load_daily_history,
        membership_for,
        tickers_with_prices,
        universe_tickers,
    )

    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}, not {universe!r}")
    active = active_tickers(session)
    active_set = set(active)
    tickers = active if universe == "active" else universe_tickers(session, index_code, None, end)

    real, null = {}, {}
    comp_real, comp_null = {}, {}
    extra = {s: {} for s in null_seeds}                  # H6: seed -> {ticker: (f, y)}
    comp_extra = {s: {} for s in null_seeds}
    h6_s = 0.0
    skipped, n_rows, member_rows = [], 0, 0
    first, last = None, None
    for ticker in tickers:
        bars = load_daily_history(session, ticker, end)
        if len(bars) < MIN_STUDY_BARS:
            skipped.append(f"{ticker} ({len(bars)} bars)")
            continue
        n_rows += len(bars)
        first = bars.index[0] if first is None else min(first, bars.index[0])
        last = bars.index[-1] if last is None else max(last, bars.index[-1])
        f, y = rs.asset_study_frames(bars)
        nf, ny = rs.asset_study_frames(rs.shuffled_null_bars(bars))
        t_h6 = time.monotonic()
        ex = {s: extra_null_frames(bars, s) for s in null_seeds}
        h6_s += time.monotonic() - t_h6
        if universe == "index":
            if compare_active and ticker in active_set:
                comp_real[ticker], comp_null[ticker] = (f, y), (nf, ny)
                for s, fy in ex.items():
                    comp_extra[s][ticker] = fy
            member = point_in_time_membership(
                membership_for(session, index_code, [ticker], bars.index), active_set,
                complete_from)[ticker].to_numpy(dtype=bool)
            member_rows += int(member.sum())
            f, nf = mask_non_members(f, member), mask_non_members(nf, member)
            ex = {s: (mask_non_members(ef, member), ey) for s, (ef, ey) in ex.items()}
        real[ticker], null[ticker] = (f, y), (nf, ny)
        for s, fy in ex.items():
            extra[s][ticker] = fy
        logger.info("regime study: %s done (%d bars)", ticker, len(bars))
    info = {"universe": universe, "n_assets": len(real), "skipped": skipped, "n_rows": n_rows,
            "first": first, "last": last}
    panels = _wide4(real, null)
    if null_seeds:
        info["null_excess_panels"] = _wide_pairs(extra)
    if universe == "index":
        calendar = (panels[0]["REGIME"].index if panels[0] else pd.DatetimeIndex([]))
        info.update(
            index_code=index_code, complete_from=complete_from, n_members=len(tickers),
            member_rows=member_rows,
            priced=tickers_with_prices(session, tickers),
            membership=point_in_time_membership(
                membership_for(session, index_code, tickers, calendar), active_set,
                complete_from),
        )
        if compare_active:
            for ticker in (t for t in active if t not in set(tickers)):   # none for NIFTY50
                bars = load_daily_history(session, ticker, end)
                if len(bars) < MIN_STUDY_BARS:
                    continue
                comp_real[ticker] = rs.asset_study_frames(bars)
                comp_null[ticker] = rs.asset_study_frames(rs.shuffled_null_bars(bars))
                t_h6 = time.monotonic()
                for s in null_seeds:
                    comp_extra[s][ticker] = extra_null_frames(bars, s)
                h6_s += time.monotonic() - t_h6
            info["active_panels"] = _wide4(comp_real, comp_null)
            info["n_active"] = len(comp_real)
            if null_seeds:
                info["active_null_excess_panels"] = _wide_pairs(comp_extra)
    if null_seeds:
        info["h6_s"] = h6_s
    return (*panels, info)


def coverage_summary(membership: pd.DataFrame, regime_wide: pd.DataFrame, oos,
                     priced: set[str], complete_from: date = COMPLETE_FROM) -> dict:
    """
    Coverage of the point-in-time universe on the OOS dates, as ``report._coverage_info``:
    per date, members with a finite REGIME / members (``panel.universe_coverage``,
    members without prices count as uncovered), mean and minimum over the OOS
    dates with >= ``MIN_NAMES`` covered members (the dates the statistics can
    use); OOS member days of the members with no price rows and of the priced
    members not studied (too few bars); calendar and OOS dates before
    ``complete_from`` (the "active today" fallback).
    """
    oos = pd.DatetimeIndex(oos)
    member = membership.reindex(index=oos, fill_value=False).astype(bool)
    cov = universe_coverage(member, regime_wide.reindex(index=oos))
    evaluated = cov["covered"] >= MIN_NAMES
    c = cov.loc[evaluated, "coverage"]
    days = member.sum(axis=0)
    studied = set(regime_wide.columns)
    cf = pd.Timestamp(complete_from)
    return {
        "coverage_mean": float(c.mean()) if c.notna().any() else math.nan,
        "coverage_min": float(c.min()) if c.notna().any() else math.nan,
        "coverage_min_date": c.idxmin() if c.notna().any() else None,
        "coverage_skipped_dates": int((~evaluated).sum()),
        "missing_prices": {t: int(days[t]) for t in membership.columns if t not in priced},
        "short_history": {t: int(days[t]) for t in membership.columns
                          if t in priced and t not in studied},
        "fallback_dates": int((membership.index < cf).sum()),
        "fallback_oos_dates": int((oos < cf).sum()),
    }


# ──────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────

def _f(x, fmt="{:+.3f}") -> str:
    if x is None:
        return "–"
    try:
        if not np.isfinite(x):
            return "–"
    except TypeError:
        return str(x)
    return fmt.format(x)


def _pct(x) -> str:
    return _f(x * 100 if x is not None and np.isfinite(x) else math.nan, "{:.1f}%")


def _t(x) -> str:
    return _f(x, "{:+.2f}")


def _q(x) -> str:
    return _f(x, "{:.3f}")


def _i(x) -> str:
    return "–" if x is None or not np.isfinite(x) else str(int(x))


def _table(df: pd.DataFrame, cols) -> str:
    if df.empty:
        return "_no rows_\n"
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fn(r[c]) if c in r else "–" for _, c, fn in cols) + " |")
    return "\n".join(lines) + "\n"


_SIGN = {1: ">", -1: "<"}


def _expect(r) -> str:
    if r["section"] == "conditional":
        return f"{_SIGN[r['expected_sign']]} 0"
    return f"IC {_SIGN[r['expected_sign']]} 0"


_TEST_COLS = [
    ("target", "target", str), ("H", "horizon", _i), ("expected", "expect", str),
    ("days", "n_days", _i), ("mean", "mean", lambda x: _f(x, "{:+.4f}")), ("NW t", "t_nw", _t),
    ("BH q", "q", _q), ("blocks with expected sign", "block_share", lambda x: _f(x, "{:.2f}")),
    ("verdict", "verdict", str),
]


def _verdict_paragraph(tests: pd.DataFrame, title: str, what: str) -> str:
    parts = []
    for _, r in tests.iterrows():
        parts.append(f"{r['name']} → {r['target']}: **{r['verdict']}** "
                     f"(mean {_f(r['mean'], '{:+.4f}')}, t {_t(r['t_nw'])}, q {_q(r['q'])}, "
                     f"expected sign in {_f(r['block_share'], '{:.0%}')} of blocks)")
    counts = tests["verdict"].value_counts()
    head = ", ".join(f"{counts.get(k, 0)} {k}" for k in ("supported", "not supported", "wrong sign"))
    return f"**{title}.** {what} Result: {head}. " + "; ".join(parts) + ".\n"


def _share(x) -> str:
    return _f(x * 100 if x is not None and np.isfinite(x) else math.nan, "{:.1f}%")


def universe_note(info: dict) -> str:
    """Header note: the survivorship caveat for "active", the coverage note for "index"."""
    if info.get("universe", "active") != "index":
        return SURVIVORSHIP_CAVEAT
    text = (f"**Universe: point-in-time {info['index_code']}.** A stock's regime features count "
            f"only on dates when it was an index member (`index_membership`, changes effective "
            f"on or before the date); names that left the index count for the periods they were "
            f"members. OOS coverage (members with a finite REGIME / members, per date): mean "
            f"{_share(info.get('coverage_mean'))}, minimum {_share(info.get('coverage_min'))}"
            + (f" ({info['coverage_min_date']:%Y-%m-%d})"
               if info.get("coverage_min_date") is not None else "") + ".")
    cmin = info.get("coverage_min", math.nan)
    if np.isfinite(cmin) and cmin < 1.0:
        gaps = [t for t, n in {**info.get("missing_prices", {}),
                               **info.get("short_history", {})}.items() if n]
        text += (" **Coverage is below 100%:** members without usable price history (or still "
                 "in REGIME warmup) are left out on those dates, so a residual survivorship bias "
                 "remains" + (f" (no usable prices during the OOS window: {', '.join(gaps)})"
                              if gaps else "") + ".")
    if info.get("fallback_oos_dates"):
        text += (f" {info['fallback_oos_dates']} OOS dates fall before {info['complete_from']} "
                 "and use today's active stocks.")
    return text


def _setup_universe_lines(info: dict, end: date | None) -> list[str]:
    span = (f"{info['n_rows']} daily bars, {info['first']:%Y-%m-%d} → {info['last']:%Y-%m-%d}"
            + (f" (`--end {end}`)" if end else ""))
    if info.get("universe", "active") != "index":
        lines = [f"- Universe: {info['n_assets']} active stocks (today's NIFTY50 constituents, "
                 f"see the survivorship caveat above); {span}."]
        if info["skipped"]:
            lines.append(f"- Skipped (< {MIN_STUDY_BARS} bars): {', '.join(info['skipped'])}.")
        return lines
    lines = [
        f"- Universe: point-in-time {info['index_code']} from `index_membership`: "
        f"{info['n_members']} names were members at some point up to the last bar, "
        f"{len(info['priced'])} of them have price rows and {info['n_assets']} have ≥ "
        f"{MIN_STUDY_BARS} bars and are studied; {span}. The conditioning features (REGIME, "
        "the confidence spans, REGIME_SCORE_TREND/CHAOS, HURST_100, VOL_RATIO) are computed on "
        "each asset's full stored history and set to NaN on the dates when it was not a member, "
        "on the real and the shuffled-null panel alike "
        f"({info['member_rows']} member bars of {info['n_rows']}). The ex-post targets are never "
        "masked: a target window may run past an exit, as the harness labels do.",
        f"- Coverage on OOS dates (members with a finite REGIME / members): mean "
        f"{_share(info.get('coverage_mean'))}, minimum {_share(info.get('coverage_min'))}"
        + (f" on {info['coverage_min_date']:%Y-%m-%d}"
           if info.get("coverage_min_date") is not None else "")
        + f", over the OOS dates with ≥ {MIN_NAMES} covered members "
        f"({info.get('coverage_skipped_dates', 0)} OOS dates with fewer are left out).",
    ]
    if info.get("missing_prices"):
        lines.append("- Members with **no price rows** (never filled; OOS member days in "
                     "brackets): " + ", ".join(f"{t} ({n})" for t, n in
                                               sorted(info["missing_prices"].items())) + ".")
    if info.get("short_history"):
        lines.append(f"- Members with price rows but < {MIN_STUDY_BARS} bars, skipped (OOS "
                     "member days): " + ", ".join(f"{t} ({n})" for t, n in
                                                  sorted(info["short_history"].items())) + ".")
    lines.append(f"- Membership before {info['complete_from']} is not sourced and falls back to "
                 f"today's active stocks: {info.get('fallback_dates', 0)} calendar dates "
                 f"({info.get('fallback_oos_dates', 0)} OOS).")
    if "n_active" in info:
        lines.append(f"- Comparison universe \"active\": the {info['n_active']} active stocks "
                     f"with ≥ {MIN_STUDY_BARS} bars, unmasked (the pre-260 study), on the same "
                     "walk-forward splits; see \"Universe effect\".")
    return lines


_KEYS = ["section", "group", "name", "target", "horizon"]


def universe_effect(index_tests: pd.DataFrame, active_tests: pd.DataFrame) -> pd.DataFrame:
    """Every pre-registered test: statistic, NW t, BH q and verdict in both universes."""
    cols = [*_KEYS, "mean", "t_nw", "q", "verdict"]
    m = index_tests[cols].merge(active_tests[cols], on=_KEYS, how="left",
                                suffixes=("_index", "_active"))       # the index table's order
    m["changed"] = m["verdict_index"] != m["verdict_active"]
    return m


def _test_label(r) -> str:
    return f"{r['group']} {r['name']} → {r['target']}"


def _universe_effect_section(res: rs.StudyResult, compare: rs.StudyResult) -> list[str]:
    out = ["## Universe effect\n",
           "Every pre-registered test (H1–H5) in the point-in-time *index* universe (the headline "
           "above) and in today's *active* stocks, unmasked (the universe of the pre-260 "
           "study), on the same walk-forward splits and OOS dates, each with its own BH "
           f"adjustment over its {len(res.tests)} tests. Statistic: mean paired difference "
           "(H1–H3) or mean daily IC (H4, H5).\n"]
    ue = universe_effect(res.tests, compare.tests)
    ue["test"] = [_test_label(r) for _, r in ue.iterrows()]
    out.append(_table(ue, [
        ("test", "test", str), ("H", "horizon", _i),
        ("stat index", "mean_index", lambda x: _f(x, "{:+.4f}")), ("t index", "t_nw_index", _t),
        ("q index", "q_index", _q), ("verdict index", "verdict_index", str),
        ("stat active", "mean_active", lambda x: _f(x, "{:+.4f}")),
        ("t active", "t_nw_active", _t), ("q active", "q_active", _q),
        ("verdict active", "verdict_active", str),
    ]))
    changed = ue[ue["changed"]]
    if changed.empty:
        out.append("**No verdict changes between the universes.**\n")
    else:
        out.append("**Verdicts that change between the universes:** "
                   + "; ".join(f"{r['test']} (H {r['horizon']}): {r['verdict_active']} (active) "
                               f"→ {r['verdict_index']} (index)" for _, r in changed.iterrows())
                   + ".\n")
    counts = {u: ue[f"verdict_{u}"].value_counts() for u in ("index", "active")}
    out.append("Verdict counts: " + "; ".join(
        f"{u}: " + ", ".join(f"{counts[u].get(k, 0)} {k}"
                            for k in ("supported", "not supported", "wrong sign"))
        for u in ("index", "active")) + ".\n")
    return out


def h6_universe_effect(index_h6: pd.DataFrame, active_h6: pd.DataFrame) -> pd.DataFrame:
    """The H6 rows of both universes side by side (``universe_effect`` plus null mean)."""
    cols = [*_KEYS, "mean", "null_mean", "t_nw", "q", "block_share", "verdict"]
    m = index_h6[cols].merge(active_h6[cols], on=_KEYS, how="left",
                             suffixes=("_index", "_active"))
    m["changed"] = m["verdict_index"] != m["verdict_active"]
    return m


_H6_COLS = [
    ("test", "test", str), ("H", "horizon", _i), ("days", "n_days", _i),
    ("real", "real", lambda x: _f(x, "{:+.4f}")),
    ("null mean", "null_mean", lambda x: _f(x, "{:+.4f}")),
    ("mean excess", "mean", lambda x: _f(x, "{:+.4f}")), ("NW t", "t_nw", _t),
    ("BH q (H6)", "q", _q), ("blocks with expected sign", "block_share", lambda x: _f(x, "{:.2f}")),
    ("verdict", "verdict", str),
]

H6_OPTIONS_PARAGRAPH = (
    "**What H6 would mean for the pending regime decision** (DECISIONS_PENDING \"Regime "
    "study: results\", options (a)–(d)); this report does not choose. If the H6 rank tests "
    "are *supported*, REGIME_SCORE_CHAOS / VOL_RATIO rank future volatility beyond the shared-"
    "denominator artefact (real volatility clustering), so the chaos side of the regime layer "
    "carries a real, if small, signal: that is evidence for keeping it, as in (a), or for a "
    "re-fit, as in (b), having a volatility target worth fitting, and it weakens (d)'s case "
    "for a fixed blend as far as the chaos/volatility part is concerned. It says nothing about "
    "trend vs range (H1) or confidence calibration (H5), which stay indistinguishable from the "
    "null. If H6 is *not supported*, every supported result of this study is mechanical, no "
    "regime input has passed it, and (d) (a fixed Black-Litterman ↔ risk-parity blend until a "
    "regime signal passes) loses its only counter-evidence; (b) would then be fitting "
    "thresholds to inputs with no demonstrated signal. A *wrong sign* would mean the real "
    "association is weaker than the mechanical one. (c) (`CONFIDENCE_EMA_SPAN`) rests on the "
    "calibration and churn tables, not on H6, either way.")


def _h6_section(res: rs.StudyResult, compare: rs.StudyResult | None) -> list[str]:
    h6 = res.h6.copy()
    n_seeds = int(h6["n_seeds"].iloc[0]) if len(h6) else 0
    out = ["## H6: real minus shuffled null (pre-registered)\n",
           "Pre-registered separately (module docstring, task 330) because the H2/H4 volatility "
           "tests have a large mechanical part: VOL_RATIO (and so REGIME_SCORE_CHAOS) and "
           "FWD_VOLX share the REALVOL_250 denominator, and REALVOL_20 sits inside REALVOL_250's "
           "window. Per OOS date, the real statistic (the H2 paired chaos − rest difference, or "
           "the H4 daily IC) minus the mean of the same statistic over the shuffled-returns "
           f"nulls of {n_seeds} seeds ({', '.join(map(str, rs.NULL_EXCESS_SEEDS[:n_seeds]))}; "
           "the first is the null of the tables above) whose value on that date is finite; a "
           "date with no finite null or real value is skipped. *real* and *null mean* are the "
           "time means of the two statistics on the dates used (real = null mean + mean "
           "excess). Newey-West t with lag H; expected sign > 0 for all six; Benjamini–Hochberg "
           "over these 6 tests only (the H1–H5 q-values above are unchanged); same verdict "
           "rule. Caveat: REALVOL_250 keeps these per-date statistics persistent for up to "
           "~250 bars, longer than the Newey-West lag, so the t-stats are optimistic on ~700 "
           "dates (on synthetic iid panels of ~480 OOS dates the lag-H t reached |t| ≈ 4 with a "
           "true mean of 0; on ~2,500 dates it stayed below 2).\n"]
    frames = [("index" if compare is not None else None, res)]
    if compare is not None and len(compare.h6):
        frames.append(("active", compare))
    for label, r in frames:
        t = r.h6.copy()
        t["test"] = [f"{x['name']} → {x['target']}" for _, x in t.iterrows()]
        t["real"] = t["mean"] + t["null_mean"]
        if label:
            out.append(f"**{label}** universe:\n")
        out.append(_table(t, _H6_COLS))
    counts = h6["verdict"].value_counts()
    out.append("H6 verdicts" + (" (index)" if compare is not None else "") + ": "
               + ", ".join(f"{counts.get(k, 0)} {k}"
                           for k in ("supported", "not supported", "wrong sign")) + ".\n")
    if compare is not None and len(compare.h6):
        ue = h6_universe_effect(res.h6, compare.h6)
        out.append("Universe effect (H6):\n")
        for _, x in ue.iterrows():
            out.append(f"- H6 {x['name']} → {x['target']}: index **{x['verdict_index']}** "
                       f"(excess {_f(x['mean_index'], '{:+.4f}')}, t {_t(x['t_nw_index'])}, "
                       f"q {_q(x['q_index'])}), active **{x['verdict_active']}** "
                       f"(excess {_f(x['mean_active'], '{:+.4f}')}, t {_t(x['t_nw_active'])}, "
                       f"q {_q(x['q_active'])})"
                       + ("; **verdict changes**." if x["changed"] else "; unchanged."))
        out.append("")
    out.append(H6_OPTIONS_PARAGRAPH + "\n")
    return out


#: The H4 tests that are "supported" under the fixed lag H in the earlier reports
#: (2026-10-18 to 2026-10-20, both universes); the robust section always lists them.
H4_SUPPORTED_TODAY: tuple[tuple[str, str], ...] = (
    ("REGIME_SCORE_CHAOS", "FWD_VOLX_20"), ("REGIME_SCORE_CHAOS", "FWD_VOLX_60"),
    ("VOL_RATIO", "FWD_VOLX_20"), ("VOL_RATIO", "FWD_VOLX_60"),
)
_VERDICTS = ("supported", "not supported", "wrong sign")
_RULES = (("fixed lag H", "verdict"), ("auto", "verdict_auto"), ("boot", "verdict_boot"))
#: A size-check rejection rate above this (twice the 5% level) is called over-rejection.
SIZE_OVER_REJECT: float = 2 * rs.SIZE_CHECK_LEVEL

_ROBUST_TEST_COLS = [
    ("section", "section", str), ("test", "test", str), ("days", "n_days", _i), ("mean", "mean", lambda x: _f(x, "{:+.4f}")),
    ("t (lag H)", "t_nw", _t), ("q", "q", _q), ("lag auto", "lag_auto", _i),
    ("t auto", "t_auto", _t), ("q auto", "q_auto", _q), ("boot block", "boot_block", _i),
    ("p boot", "p_boot", lambda x: _f(x, "{:.4f}")), ("q boot", "q_boot", _q),
    ("verdict (lag H)", "verdict", str), ("verdict auto", "verdict_auto", str),
    ("verdict boot", "verdict_boot", str),
]

def _h6_labels(h6: pd.DataFrame, verdict: str) -> list[str]:
    return [f"{x['name']} → {x['target']}" for _, x in h6[h6["verdict"] == verdict].iterrows()]


def h6_robust_clauses(h6: pd.DataFrame) -> tuple[str, str]:
    """The two H6-dependent clauses of the robust-options paragraph, chosen by the H6 verdicts
    (``StudyResult.h6``, the lag-H ``verdict``): none supported, all supported, k of m supported
    (the supported tests named), or H6 skipped (empty frame, ``--no-h6``)."""
    m = len(h6)
    if m == 0:
        return ("and H6 was not run (`--no-h6`), so how much of their excess over the shuffled "
                "null is shared-denominator mechanics is not measured here.",
                "but without H6 it cannot be told apart from the shared-denominator mechanics, "
                "so it does not by itself argue for keeping the chaos side as is (a) or for a "
                "re-fit (b).")
    supported = _h6_labels(h6, "supported")
    k = len(supported)
    if k == 0:
        return ("and H6 already showed that their excess over the shuffled null is not "
                f"significant (0 of {m} H6 tests supported).",
                "but it is still mostly the shared-denominator mechanics H6 measures, so it does "
                "not by itself argue for keeping the chaos side as is (a) or for a re-fit (b).")
    if k == m:
        return (f"and H6 found their excess over the shuffled null significant ({m} of {m} H6 "
                "tests supported).",
                "and, as H6 found an excess over the shuffled null in every test, it is not "
                "only the shared-denominator mechanics H6 measures: that is evidence for keeping "
                "the chaos side as is (a) or for a re-fit (b), subject to the persistence caveat "
                "in the H6 section, and it concerns volatility only.")
    return (f"and H6 found their excess over the shuffled null significant in {k} of {m} tests "
            f"(supported: {', '.join(supported)}).",
            "but H6 supports only part of it, so the rest is still the shared-denominator "
            "mechanics H6 measures: the supported H6 tests are evidence for keeping the chaos "
            "side as is (a) or for a re-fit (b) only as far as they go, subject to the "
            "persistence caveat in the H6 section.")


def robust_options_paragraph(res: rs.StudyResult) -> str:
    """"What the robust re-test would mean ..." with the H6 clauses built from ``res.h6``."""
    shown, mechanics = h6_robust_clauses(res.h6)
    return (
        "**What the robust re-test would mean for the pending regime decision** "
        "(DECISIONS_PENDING \"Regime study: results\", options (a)–(d)); this report does not "
        "choose. The four H4 volatility tests (REGIME_SCORE_CHAOS / VOL_RATIO vs FWD_VOLX) are "
        "the only *supported* results of H1–H5, " + shown + " If they stay *supported* under "
        "the automatic lag and the bootstrap, the raw association is not an inference artefact, "
        + mechanics + " If they become *not supported*, no pre-registered test of the regime "
        "layer survives autocorrelation-robust inference: that removes the last evidence that "
        "the regime inputs carry a signal on daily NIFTY50, which strengthens (d) (a fixed "
        "Black-Litterman ↔ risk-parity blend until a regime signal passes) and weakens (b) "
        "(fitting thresholds to inputs with no demonstrated signal). A robust rule can turn a "
        "test *supported* only where it is less conservative than the lag-H t; the size check "
        "above says how far either rule can be trusted on series as persistent as these, and a "
        "rule that still over-rejects there makes a robust *supported* weak evidence while a "
        "robust *not supported* stands. (c) (`CONFIDENCE_EMA_SPAN`) rests on the calibration "
        "and churn tables: H5 is not supported under any rule unless the change table above "
        "lists an H5 test.")


def _family_frames(r: rs.StudyResult) -> list[tuple[str, pd.DataFrame]]:
    out = [("H1–H5", r.tests)]
    if len(r.h6):
        out.append(("H6", r.h6))
    return out


def _robust_counts(r: rs.StudyResult) -> pd.DataFrame:
    rows = []
    for fam, df in _family_frames(r):
        for rule, col in _RULES:
            c = df[col].value_counts()
            rows.append({"family": fam, "tests": len(df), "rule": rule,
                         **{k: int(c.get(k, 0)) for k in _VERDICTS}})
    return pd.DataFrame(rows)


def _robust_groups(r: rs.StudyResult) -> pd.DataFrame:
    """Per group (H1 ... H6): test count, horizons and medians of the robust inputs."""
    both = pd.concat([df for _, df in _family_frames(r)], ignore_index=True)
    rows = []
    for g, df in both.groupby("group", sort=True):
        rows.append({"group": g, "tests": len(df),
                     "H": ", ".join(map(str, sorted(set(df["horizon"])))),
                     "median_days": float(df["n_days"].median()),
                     "median_acf1": float(df["acf1"].median()),
                     "median_lag_auto": float(df["lag_auto"].median()),
                     "median_boot_block": float(df["boot_block"].median())})
    return pd.DataFrame(rows)


_GROUP_ROWS = (("tests", "tests", _i), ("horizons H", "H", str),
               ("median days", "median_days", lambda x: _f(x, "{:.0f}")),
               ("median acf1", "median_acf1", lambda x: _f(x, "{:.2f}")),
               ("median lag auto", "median_lag_auto", lambda x: _f(x, "{:.0f}")),
               ("median boot block", "median_boot_block", lambda x: _f(x, "{:.0f}")))


def _groups_table(g: pd.DataFrame) -> str:
    """``_robust_groups`` transposed: one column per group, one row per statistic."""
    lines = ["| statistic | " + " | ".join(g["group"]) + " |", "|" + "---|" * (len(g) + 1)]
    for label, col, fn in _GROUP_ROWS:
        lines.append(f"| {label} | " + " | ".join(fn(v) for v in g[col]) + " |")
    return "\n".join(lines) + "\n"


def robust_changes(r: rs.StudyResult) -> pd.DataFrame:
    """Every test (both families) whose verdict under auto or boot differs from the lag-H one."""
    both = pd.concat([df for _, df in _family_frames(r)], ignore_index=True)
    changed = (both["verdict_auto"] != both["verdict"]) | (both["verdict_boot"] != both["verdict"])
    return both[changed].reset_index(drop=True)


def _with_test_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["test"] = [f"{_test_label(x)} (H {x['horizon']})" for _, x in df.iterrows()]
    return df


def _size_check_lines(size: pd.DataFrame | None) -> list[str]:
    if size is None or not len(size):
        return []
    r0 = size.iloc[0]
    out = [f"**Size check** (pre-registered, run once, not tuned against): {int(r0['runs'])} "
           f"seeded synthetic per-date series per DGP, n = {int(r0['n'])}, true mean 0, H = "
           f"{int(r0['horizon'])}; share of runs rejecting at the two-sided "
           f"{rs.SIZE_CHECK_LEVEL:.0%} level (nominal {rs.SIZE_CHECK_LEVEL:.0%}).\n",
           _table(size, [("DGP", "name", str), ("reject, lag H", "reject_fixed", _pct),
                         ("reject, auto", "reject_auto", _pct),
                         ("reject, boot", "reject_boot", _pct),
                         ("median lag auto", "median_lag_auto", lambda x: _f(x, "{:.0f}")),
                         ("median acf1", "median_acf1", lambda x: _f(x, "{:.2f}"))])]
    bad = [(x["name"], rule, x[f"reject_{k}"]) for _, x in size.iterrows()
           for rule, k in (("auto", "auto"), ("boot", "boot"))
           if x[f"reject_{k}"] > SIZE_OVER_REJECT]
    if bad:
        out.append("**The robust rules still over-reject** (rate above "
                   f"{SIZE_OVER_REJECT:.0%}, twice the nominal level): "
                   + "; ".join(f"{rule} on {name}: {_pct(v)}" for name, rule, v in bad)
                   + ". On series this persistent a robust *supported* is not conclusive; no "
                   "further method is added in this task (see DECISIONS_PENDING).\n")
    else:
        out.append(f"Neither robust rule rejects more than {SIZE_OVER_REJECT:.0%} of the time "
                   "on either DGP.\n")
    return out


def resolution_note(res: rs.StudyResult, compare: rs.StudyResult | None = None) -> str:
    """
    One line on the bootstrap's resolution. p boot >= 1/(N + 1); with k tests of a family of
    m tied at that floor, BH gives them q = m / (k (N + 1)), so the single-test floor
    m / (N + 1) of the pre-registration holds only for k = 1. The k of each family is listed.
    """
    n1 = rs.ROBUST_BOOT_N + 1
    floor = 1.0 / n1
    parts = []
    for label, r in [("index" if compare is not None else None, res),
                     *([("active", compare)] if compare is not None else [])]:
        if r is None or "p_boot" not in r.tests.columns:
            continue
        for fam, df in _family_frames(r):
            m = int(df["p_boot"].notna().sum())
            k = int((df["p_boot"] <= floor * (1 + 1e-9)).sum())
            q = df["q_boot"].min()
            parts.append(f"{label + ' ' if label else ''}{fam}: {k} of {m} at the floor"
                         + (f", min q boot {_q(q)}" if np.isfinite(q) else ""))
    return (f"Bootstrap resolution: with {rs.ROBUST_BOOT_N} draws p boot ≥ 1/{n1} ≈ {floor:.4f} "
            "(no draw as extreme as the mean). For a single top test at that floor the smallest "
            f"q boot is m/{n1} (35/{n1} ≈ {35 / n1:.3f} for H1–H5, 6/{n1} ≈ {6 / n1:.3f} for H6); "
            f"k tests tied at the floor all get m/(k·{n1}), so the minimum q boot can be lower: "
            + "; ".join(parts) + ".\n")


def _robust_section(res: rs.StudyResult, compare: rs.StudyResult | None,
                    size: pd.DataFrame | None) -> list[str]:
    out = ["## Robust inference (pre-registered)\n",
           "Pre-registered in the module docstring (task 350) before any robust run on the DB. "
           "Every test above is re-tested on the **same per-date series and dates** with the two "
           "robust checks of task 340: *auto* = Newey-West t at lag max(H, Andrews AR(1) "
           f"plug-in lag), capped at floor({rs.ROBUST_MAX_LAG_FRAC:g} n) but never below H, rho "
           f"clipped at ±{rs.ROBUST_RHO_CLIP:g}; *boot* = stationary block bootstrap p of the "
           f"centred series, {rs.ROBUST_BOOT_N} draws, seed {rs.ROBUST_BOOT_SEED}, mean block "
           "max(H, lag auto + 1, round(n^(1/3))). q-values are Benjamini–Hochberg within the "
           "existing families (H1–H5: one family; H6: its own), and the verdict rule is "
           "unchanged except that q is replaced (same sign and block-share conditions). The "
           "lag-H verdicts above stay the pre-registered result; this section only reports "
           "where a robust rule disagrees.\n"]
    frames = [("index" if compare is not None else None, res)]
    if compare is not None:
        frames.append(("active", compare))
    for label, r in frames:
        if "verdict_auto" not in r.tests.columns:
            continue
        if label:
            out.append(f"### {label} universe\n")
        out.append("Verdict counts per family and rule:\n")
        out.append(_table(_robust_counts(r), [
            ("rule", "rule", str), ("family", "family", str), ("tests", "tests", _i),
            *[(k, k, _i) for k in _VERDICTS]]))
        out.append("Persistence of the per-date series per group (medians over the group's "
                   "tests):\n")
        out.append(_groups_table(_robust_groups(r)))
        ch = robust_changes(r)
        if ch.empty:
            out.append("**No verdict changes under auto or boot.**\n")
        else:
            out.append(f"**Tests whose verdict changes under auto or boot ({len(ch)}):**\n")
            out.append(_table(_with_test_label(ch), _ROBUST_TEST_COLS))
        keys = set(H4_SUPPORTED_TODAY)
        listed = np.array([(n, t) in keys for n, t in zip(r.tests["name"], r.tests["target"])],
                          dtype=bool)
        h4 = r.tests[(r.tests["group"] == "H4").to_numpy() & listed]
        out.append("The four H4 tests that are *supported* under the lag-H rule in the earlier "
                   "reports, always listed:\n")
        out.append(_table(_with_test_label(h4), _ROBUST_TEST_COLS))
    out.append(resolution_note(res, compare))
    out.extend(_size_check_lines(size))
    out.append(robust_options_paragraph(res) + "\n")
    return out


def _repro_line(info: dict, git_hash: str, elapsed_s: float, rss_mb: float) -> str:
    db = info.get("db") or {}
    db_txt = (f"DB: max price date {db.get('db_max_date')}, {db.get('db_price_rows')} "
              f"price rows, {db.get('db_action_rows')} corporate-action rows"
              if db else "DB: not recorded")
    rt = f"runtime {elapsed_s:.0f} s"
    if "active_s" in info:
        own = elapsed_s - info["active_s"]
        rt += (f" ({info['active_s']:.0f} s of it for the active-universe comparison"
               + ("; running both universes more than doubles the runtime of the index study "
                  "alone" if info["active_s"] > own else "") + ")")
    if "h6_s" in info:
        rt += f"; H6's {len(rs.NULL_EXCESS_SEEDS) - 1} extra null seeds {info['h6_s']:.0f} s"
    if "robust_s" in info:
        rt += (f"; robust inference {info['robust_s']:.0f} s (size check "
               f"{info.get('size_check_s', 0.0):.0f} s of it)")
    return f"- SCORER_VERSION {SCORER_VERSION}; git `{git_hash}`; {db_txt}; {rt}; " \
           f"peak RSS {rss_mb:.0f} MB."


def render_report(res: rs.StudyResult, info: dict, report_date: date, *,
                  elapsed_s: float, rss_mb: float, git_hash: str, end: date | None,
                  compare: rs.StudyResult | None = None,
                  size_check: pd.DataFrame | None = None) -> str:
    """``info["universe"]`` "index" adds the coverage note and, with ``compare`` (the
    active universe on the same splits), the "Universe effect" table; "active" (or
    missing) gives the pre-260 layout with the survivorship caveat."""
    index_mode = info.get("universe", "active") == "index"
    out: list[str] = []
    w = out.append
    tests = res.tests.copy()
    tests["expect"] = [_expect(r) for _, r in tests.iterrows()]
    cond = tests[tests["section"] == "conditional"]
    rank = tests[tests["section"] == "rank"]
    cal = tests[tests["section"] == "calibration"]
    span_now = R.CONFIDENCE_EMA_SPAN

    w(f"# Regime study {report_date:%Y-%m-%d}: real-time regime label vs the ex-post regime\n")
    w(f"> {universe_note(info)}\n")
    w("Generated by `python -m scripts.regime_study` (`investbot/evaluation/regime_study.py`). "
      "A study only: no threshold, span or weight is changed, and nothing here recommends new "
      "constants. Hypotheses, expected signs, horizons, spans and the null seed were "
      "pre-registered in the module docstring before the first run.\n")

    w("## Setup\n")
    for line in _setup_universe_lines(info, end):
        w(line)
    oos = res.oos
    if len(oos):
        w(f"- Out-of-sample (OOS): {len(res.splits)} walk-forward test blocks of {TEST_MONTHS} "
          f"months, {oos[0]:%Y-%m-%d} → {oos[-1]:%Y-%m-%d}, {len(oos)} dates; same settings as "
          f"the harness (≥ {MIN_TRAIN_BARS} bars after the first date with REGIME on ≥ "
          f"{MIN_NAMES} names, embargo {max(HARNESS_HORIZONS)} bars). Blocks: "
          + ", ".join(f"`{s.label}`" for s in res.splits) + ". Every statistic below uses "
          "OOS dates only. Targets near the end of the data are NaN (their window runs past "
          "the last bar), so the last block is shorter for H = 60 and FWD_HURST_100.")
    w(f"- Conditioning: `regime.compute_regime` with the production constants "
      f"(REGIME_WINDOW {R.REGIME_WINDOW}, CONFIDENCE_EMA_SPAN {span_now}); confidence spans "
      f"{', '.join(map(str, rs.CONFIDENCE_SPANS))} via `regime.smooth_confidence` "
      "(span 1 = REGIME_CONFIDENCE_RAW).")
    w(f"- Targets (labels, closes t … t+H): FWD_ER, FWD_AC1, FWD_VR5, FWD_VOLX (std of forward "
      f"returns / REALVOL_250[t]), FWD_MDD at H ∈ {{{', '.join(map(str, rs.REGIME_HORIZONS))}}}; "
      f"FWD_HURST_{rs.EX_POST_HURST_WINDOW} = DFA Hurst of the next "
      f"{rs.EX_POST_HURST_WINDOW} returns (same function and window as the real-time HURST_100).")
    w(f"- Null: every asset's bars shuffled in time (`shuffled_null_bars`, seed {rs.NULL_SEED}): "
      "returns, bar shapes and volumes kept, all serial dependence removed.")
    w(f"- Tests: {len(tests)} pre-registered tests (conditional means, rank association, "
      f"calibration), Newey-West t with lag H, two-sided normal p, Benjamini–Hochberg over all "
      f"of them. **Supported** = expected sign, BH q < {PROMISING_Q:.2f} and the expected sign "
      f"in ≥ {PROMISING_SIGN_SHARE:.0%} of the test blocks; **wrong sign** = opposite sign with "
      f"q < {PROMISING_Q:.2f}.")
    w(_repro_line(info, git_hash, elapsed_s, rss_mb))
    w("")

    # ── Base rates
    w("## Label base rates (real vs null)\n")
    w("Share of OOS bars per label and label flips per 1000 bars (a flip = the label differs "
      "from the previous bar's). The null has no serial dependence at all, so any label "
      "structure that survives there comes from the estimators and thresholds, not the market.\n")
    pooled = pd.DataFrame([
        {"panel": "real", **res.base_rates_real.iloc[-1].to_dict()},
        {"panel": "null", **res.base_rates_null.iloc[-1].to_dict()},
    ])
    share_cols = [(f"{n}", f"share_{n}", _pct) for n in REGIME_NAMES.values()]
    w(_table(pooled, [("panel", "panel", str), ("bars", "n_bars", _i), *share_cols,
                      ("flips / 1000 bars", "flips_per_1000", lambda x: _f(x, "{:.1f}"))]))
    w(f"HURST_{R.REGIME_WINDOW} on OOS bars (a random walk reads ≈ 0.53 at this window, the "
      "known DFA small-sample bias):\n")
    hd = pd.DataFrame([{"panel": "real", **res.hurst_real}, {"panel": "null", **res.hurst_null}])
    w(_table(hd, [("panel", "panel", str), ("bars", "n", _i), ("mean", "mean", _q),
                  ("5th pct", "p05", _q), ("median", "median", _q), ("95th pct", "p95", _q)]))
    real_a = res.base_rates_real.iloc[:-1].set_index("ticker")
    null_a = res.base_rates_null.iloc[:-1].set_index("ticker")
    per = real_a.join(null_a, rsuffix="_null").reset_index()
    w("<details><summary>Per asset</summary>\n")
    w(_table(per, [("ticker", "ticker", str), ("bars", "n_bars", _i),
                   *[(f"{n}", f"share_{n}", _pct) for n in REGIME_NAMES.values()],
                   ("flips/1000", "flips_per_1000", lambda x: _f(x, "{:.1f}")),
                   *[(f"null {n}", f"share_{n}_null", _pct) for n in REGIME_NAMES.values()],
                   ("null flips/1000", "flips_per_1000_null", lambda x: _f(x, "{:.1f}"))]))
    w("</details>\n")

    # ── Conditional means
    w("## Conditional means\n")
    w(f"Per date, the cross-sectional mean of the target within each label (dates with ≥ "
      f"{rs.MIN_GROUP_NAMES} names in the label), then the time mean over OOS dates "
      "(\"days\" = dates where the label has enough names):\n")
    gm = res.group_means.drop_duplicates("target")
    w(_table(gm, [("target", "target", str), ("H", "horizon", _i),
                  *[(n, n, lambda x: _f(x, "{:+.4f}")) for n in REGIME_NAMES.values()],
                  *[(f"{n} days", f"{n}_days", _i) for n in REGIME_NAMES.values()]]))
    w("Tests: paired per-date differences (trend − range; chaos − rest, rest = trend ∪ range). "
      "The last two columns run the same test on the shuffled null panel (not in BH; it "
      "should be ≈ 0).\n")
    nc = res.null_conditional.set_index(["name", "target"])
    c2 = cond.copy()
    c2["null_mean"] = [nc["mean"].get((n, t), math.nan) for n, t in zip(c2["name"], c2["target"])]
    c2["null_t"] = [nc["t_nw"].get((n, t), math.nan) for n, t in zip(c2["name"], c2["target"])]
    w(_table(c2, [("test", "name", str), *_TEST_COLS,
                  ("null mean", "null_mean", lambda x: _f(x, "{:+.4f}")), ("null t", "null_t", _t)]))

    # ── Rank association
    w("## Rank association\n")
    w(f"Daily cross-sectional Spearman IC (dates with ≥ {MIN_NAMES} names), Newey-West t with "
      "lag H.\n")
    w("The last two columns run the same IC on the shuffled null panel (not in BH), which has "
      "no volatility clustering. What is left there is mechanical: VOL_RATIO (and so "
      "REGIME_SCORE_CHAOS) and FWD_VOLX share the REALVOL_250 denominator, so sampling noise "
      "in it correlates them across names; a burst of recent volatility also inflates "
      "REALVOL_250, which pulls FWD_VOLX down after it. Compare each IC with its null IC.\n")
    nr = res.null_rank.set_index(["name", "target"])
    r2 = rank.copy()
    r2["null_mean"] = [nr["mean"].get((n, t), math.nan) for n, t in zip(r2["name"], r2["target"])]
    r2["null_t"] = [nr["t_nw"].get((n, t), math.nan) for n, t in zip(r2["name"], r2["target"])]
    w(_table(r2, [("signal", "name", str), *_TEST_COLS,
                  ("null IC", "null_mean", lambda x: _f(x, "{:+.4f}")), ("null t", "null_t", _t)]))

    # ── Calibration
    w("## Confidence calibration by span\n")
    w("Trend and range bars only (chaos excluded). HIT_H = the bar's per-date percentile rank of "
      "FWD_ER_H among those names, in [−0.5, 0.5], × +1 for trend and −1 for range: positive "
      "when the label was right relative to the cross-section. The IC of confidence vs HIT is "
      f"positive when higher confidence means the label is more right. Span {span_now} is "
      "production.\n")
    w(_table(cal, [("confidence", "name", str), *_TEST_COLS]))
    w("Mean HIT by confidence quintile (per-date quintiles of confidence, time mean of each "
      "quintile's mean HIT; Q5 = most confident):\n")
    w(_table(res.hit_quintiles, [("span", "span", _i), ("H", "horizon", _i),
                                 *[(f"Q{k}", f"Q{k}", lambda x: _f(x, "{:+.4f}"))
                                   for k in range(1, 6)]]))
    w("Churn on OOS bars (all labels): mean |Δconfidence| per bar and crossings of each level "
      "per 1000 bars. Each crossing of a blend threshold is a potential rebalance.\n")
    w(_table(res.churn, [("span", "span", _i), ("bar pairs", "n_pairs", _i),
                         ("mean |Δ|", "mean_abs_change", lambda x: _f(x, "{:.4f}")),
                         *[(f"crossings of {lv:g} / 1000", f"cross_{lv:g}_per_1000",
                            lambda x: _f(x, "{:.1f}")) for lv in rs.CROSSING_LEVELS]]))

    # ── Verdict
    w("## Verdict per pre-registered hypothesis\n")
    w(_verdict_paragraph(cond[cond["group"] == "H1"], "H1 persistence",
                         "Bars labelled trend are followed by more persistent paths than bars "
                         "labelled range (trend − range > 0)."))
    w(_verdict_paragraph(cond[cond["group"] == "H2"], "H2 volatility expansion",
                         "Chaos is followed by higher volatility relative to the asset's "
                         "one-year norm than the other labels (chaos − rest > 0)."))
    w(_verdict_paragraph(cond[cond["group"] == "H3"], "H3 drawdown",
                         "Chaos is followed by deeper drawdowns (chaos − rest < 0)."))
    w(_verdict_paragraph(rank, "H4 rank association",
                         "The continuous inputs rank the cross-section by its ex-post regime."))
    w(_verdict_paragraph(cal, "H5 confidence calibration",
                         "Higher REGIME_CONFIDENCE (each span) means the trend/range label is "
                         "more often right."))

    if index_mode and compare is not None:
        for line in _universe_effect_section(res, compare):
            w(line)

    if len(res.h6):
        for line in _h6_section(res, compare if index_mode else None):
            w(line)

    if "verdict_auto" in res.tests.columns:
        for line in _robust_section(res, compare if index_mode else None, size_check):
            w(line)

    w("## Options for Ankit\n")
    w("This report does not recommend new constants. The choices it informs:\n")
    w("1. **Keep the thresholds and span as they are** (`regime.py` unchanged; current state).")
    w("2. **A re-fit task** for the regime thresholds/weights (HURST_NEUTRAL/SCALE, ADX, "
      "VOL_RATIO_CHAOS, LYAP_CHAOS) against these ex-post targets, with its own walk-forward "
      "split and pre-registration (BACKLOG \"re-fit regime thresholds\").")
    w(f"3. **Change `CONFIDENCE_EMA_SPAN`** (now {span_now}) using the calibration-IC and churn "
      "tables above (BACKLOG \"test `CONFIDENCE_EMA_SPAN`\").")
    w("4. **Revisit the `REGIME_CONFIDENCE` definition** (top score minus runner-up), e.g. if "
      "confidence does not rank how often the label is right.")
    w("5. **HURST_100 bias correction** (BACKLOG \"HURST_100 ~0.53 on a random walk\"): compare "
      "the real and null Hurst distributions above.")
    w("")

    w("## Caveats\n")
    if index_mode:
        w("- Residual survivorship: members without price rows (or with too few bars) are left "
          "out on their member dates (see Setup); the coverage note at the top gives the share "
          "of members studied. Before "
          f"{info['complete_from']} membership falls back to today's active stocks.")
    else:
        w(f"- {SURVIVORSHIP_CAVEAT}")
    w("- The targets overlap heavily (consecutive dates share H − 1 bars, 99 for "
      "FWD_HURST_100), so the effective sample is far smaller than the number of days; the "
      "Newey-West t accounts for this only approximately, especially for H = 60 and 100 bars "
      "against ~690 OOS dates.")
    w("- The null shares one date permutation across assets with the same calendar, so its "
      "cross-section still co-moves; only serial dependence is removed.")
    w("- FWD_VOLX is normalised by REALVOL_250 and VOL_RATIO = REALVOL_20 / REALVOL_250. The "
      "shared denominator alone makes VOL_RATIO vs FWD_VOLX positive across names (see the null "
      "IC), and volatility clustering adds to it; REALVOL_20 is also inside REALVOL_250's "
      "window, so a high VOL_RATIO inflates the normaliser and lowers FWD_VOLX (see the null "
      "chaos − rest difference).")
    w("- FWD_HURST_100 has the same DFA small-sample bias as the real-time HURST_100; the "
      "trend − range difference is unaffected by a common bias.")
    return "\n".join(out) + "\n"


@lru_cache(maxsize=1)
def _size_check_table() -> pd.DataFrame:
    return rs.size_check()


def cached_size_check() -> pd.DataFrame:
    """``rs.size_check()`` with the pre-registered settings, computed once per process
    (it is seeded synthetic data, independent of the DB and the universe)."""
    return _size_check_table().copy()


_ROBUST_ALL: tuple[str, ...] = (*rs.ROBUST_COLUMNS, *rs.ROBUST_VERDICT_COLUMNS)


def robust_last(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the robust columns (if any) moved after every other column, including
    ``universe``, so the CSVs keep their pre-350 columns first and in order."""
    rob = [c for c in _ROBUST_ALL if c in df.columns]
    return df[[c for c in df.columns if c not in rob] + rob]


def run(end: date | None = None, *, universe: str = "index", index_code: str = INDEX_CODE,
        session=None, report_dir: Path = REPORT_DIR, csv_dir: Path = CSV_DIR,
        report_date: date | None = None,
        complete_from: date = COMPLETE_FROM, h6: bool = True,
        robust: bool = True) -> tuple[Path, rs.StudyResult]:
    """Build the panels, run the study, write the report and CSVs. With
    ``universe="index"`` the active universe is also studied on the same splits.
    ``h6`` adds the pre-registered H6 family (``NULL_EXCESS_SEEDS[1:]`` as extra
    shuffled nulls, section "H6" and ``regime_study_h6_<date>.csv``). ``robust`` adds the
    pre-registered robust columns, the size check and the section
    "Robust inference (pre-registered)".
    """
    from investbot.evaluation.loader import db_fingerprint

    t0 = time.monotonic()
    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    try:
        feats, tg, nfeats, ntg, info = build_study_panels(
            session, end, universe=universe, index_code=index_code,
            complete_from=complete_from, compare_active=(universe == "index"),
            null_seeds=rs.NULL_EXCESS_SEEDS[1:] if h6 else ())
        info["db"] = db_fingerprint(session)
    finally:
        if own:
            session.close()
    if not feats:
        raise RuntimeError("no assets with enough daily bars for the regime study")
    t_study = time.monotonic()
    timings: dict[str, float] = {}
    res = rs.run_study(feats, tg, nfeats, ntg,
                       extra_null_panels=info.pop("null_excess_panels", None),
                       robust=robust, timings=timings)
    logger.info("regime study: statistics %.1f s", time.monotonic() - t_study)
    if h6:
        logger.info("regime study: H6 extra null seeds %s: %.1f s of panel building",
                    rs.NULL_EXCESS_SEEDS[1:], info["h6_s"])
    compare = None
    if universe == "index":
        info.update(coverage_summary(info.pop("membership"), feats["REGIME"], res.oos,
                                     info["priced"], complete_from))
        t_cmp = time.monotonic()
        afeats, atg, anfeats, antg = info.pop("active_panels")
        if afeats:
            compare = rs.run_study(afeats, atg, anfeats, antg, splits=res.splits,
                                   extra_null_panels=info.pop("active_null_excess_panels", None),
                                   robust=robust, timings=timings)
        info.pop("active_null_excess_panels", None)
        del afeats, atg, anfeats, antg
        # the active frames come out of the same per-asset pass, so this is the extra cost
        info["active_s"] = time.monotonic() - t_cmp
    size = None
    if robust:
        t_size = time.monotonic()
        size = cached_size_check()
        info["size_check_s"] = time.monotonic() - t_size
        info["robust_s"] = timings.get("robust", 0.0) + info["size_check_s"]
        logger.info("regime study: robust inference %.1f s (size check %.1f s)",
                    info["robust_s"], info["size_check_s"])
    report_date = report_date or date.today()
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    report, tests_csv, rates_csv = report_paths(report_dir, csv_dir, report_date)
    tests_out = res.tests if compare is None else pd.concat(
        [res.tests.assign(universe="index"), compare.tests.assign(universe="active")],
        ignore_index=True)
    robust_last(tests_out).to_csv(tests_csv, index=False)
    if len(res.h6):
        h6_out = res.h6 if compare is None or not len(compare.h6) else pd.concat(
            [res.h6.assign(universe="index"), compare.h6.assign(universe="active")],
            ignore_index=True)
        robust_last(h6_out).to_csv(tests_csv.with_name(tests_csv.name.replace("_tests_", "_h6_")),
                                   index=False)
    pd.concat([res.base_rates_real.assign(panel="real"),
               res.base_rates_null.assign(panel="null")]).to_csv(rates_csv, index=False)
    text = render_report(res, info, report_date, elapsed_s=time.monotonic() - t0,
                         rss_mb=peak_rss_mb(), git_hash=git_short_hash(), end=end,
                         compare=compare, size_check=size)
    report.write_text(text, encoding="utf-8")
    return report, res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None,
                    help="last bar date to load (default: all)")
    ap.add_argument("--universe", choices=UNIVERSES, default=None,
                    help="index: point-in-time members (default when index_membership has "
                         "rows); active: today's active stocks, unmasked")
    ap.add_argument("--no-h6", action="store_true",
                    help="skip the pre-registered H6 family (the extra shuffled-null seeds)")
    ap.add_argument("--no-robust", action="store_true",
                    help="skip the pre-registered robust inference (auto lag, block bootstrap)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logging.getLogger("investbot.scoring").setLevel(logging.WARNING)

    from investbot.data.db import SessionLocal
    from investbot.evaluation.loader import has_index_membership

    session = SessionLocal()
    try:
        has_members = has_index_membership(session, INDEX_CODE)
        universe = args.universe
        if universe is None:
            universe = "index" if has_members else "active"
            if not has_members:
                print(f"\n  WARNING: index_membership has no {INDEX_CODE} rows (run "
                      "`python main.py setup`); studying today's active stocks instead, "
                      "with survivorship bias.\n")
        elif universe == "index" and not has_members:
            print(f"index_membership has no {INDEX_CODE} rows; run `python main.py setup` first")
            sys.exit(1)
        report, res = run(args.end, universe=universe, session=session, h6=not args.no_h6,
                          robust=not args.no_robust)
    finally:
        session.close()
    counts = res.tests["verdict"].value_counts().to_dict()
    print(f"\n  Report: {report}\n  Universe: {universe}\n  Verdicts: {counts}")
    if len(res.h6):
        print(f"  H6 verdicts: {res.h6['verdict'].value_counts().to_dict()}")
    if "verdict_auto" in res.tests.columns:
        for rule in ("auto", "boot"):
            print(f"  Verdicts ({rule}): {res.tests[f'verdict_{rule}'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
