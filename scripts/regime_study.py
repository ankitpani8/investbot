"""
Regime study runner: does the real-time regime label predict the ex-post regime?

    python -m scripts.regime_study [--end YYYY-MM-DD] [--universe index|active]

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


def build_study_panels(session, end: date | None = None, *, universe: str = "index",
                       index_code: str = INDEX_CODE, complete_from: date = COMPLETE_FROM,
                       compare_active: bool = False):
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
        if universe == "index":
            if compare_active and ticker in active_set:
                comp_real[ticker], comp_null[ticker] = (f, y), (nf, ny)
            member = point_in_time_membership(
                membership_for(session, index_code, [ticker], bars.index), active_set,
                complete_from)[ticker].to_numpy(dtype=bool)
            member_rows += int(member.sum())
            f, nf = mask_non_members(f, member), mask_non_members(nf, member)
        real[ticker], null[ticker] = (f, y), (nf, ny)
        logger.info("regime study: %s done (%d bars)", ticker, len(bars))
    info = {"universe": universe, "n_assets": len(real), "skipped": skipped, "n_rows": n_rows,
            "first": first, "last": last}
    panels = _wide4(real, null)
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
            info["active_panels"] = _wide4(comp_real, comp_null)
            info["n_active"] = len(comp_real)
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
    return f"- SCORER_VERSION {SCORER_VERSION}; git `{git_hash}`; {db_txt}; {rt}; " \
           f"peak RSS {rss_mb:.0f} MB."


def render_report(res: rs.StudyResult, info: dict, report_date: date, *,
                  elapsed_s: float, rss_mb: float, git_hash: str, end: date | None,
                  compare: rs.StudyResult | None = None) -> str:
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


def run(end: date | None = None, *, universe: str = "index", index_code: str = INDEX_CODE,
        session=None, report_dir: Path = REPORT_DIR, csv_dir: Path = CSV_DIR,
        report_date: date | None = None,
        complete_from: date = COMPLETE_FROM) -> tuple[Path, rs.StudyResult]:
    """Build the panels, run the study, write the report and CSVs. With
    ``universe="index"`` the active universe is also studied on the same splits."""
    from investbot.evaluation.loader import db_fingerprint

    t0 = time.monotonic()
    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    try:
        feats, tg, nfeats, ntg, info = build_study_panels(
            session, end, universe=universe, index_code=index_code,
            complete_from=complete_from, compare_active=(universe == "index"))
        info["db"] = db_fingerprint(session)
    finally:
        if own:
            session.close()
    if not feats:
        raise RuntimeError("no assets with enough daily bars for the regime study")
    res = rs.run_study(feats, tg, nfeats, ntg)
    compare = None
    if universe == "index":
        info.update(coverage_summary(info.pop("membership"), feats["REGIME"], res.oos,
                                     info["priced"], complete_from))
        t_cmp = time.monotonic()
        afeats, atg, anfeats, antg = info.pop("active_panels")
        if afeats:
            compare = rs.run_study(afeats, atg, anfeats, antg, splits=res.splits)
        del afeats, atg, anfeats, antg
        # the active frames come out of the same per-asset pass, so this is the extra cost
        info["active_s"] = time.monotonic() - t_cmp
    report_date = report_date or date.today()
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    report, tests_csv, rates_csv = report_paths(report_dir, csv_dir, report_date)
    tests_out = res.tests if compare is None else pd.concat(
        [res.tests.assign(universe="index"), compare.tests.assign(universe="active")],
        ignore_index=True)
    tests_out.to_csv(tests_csv, index=False)
    pd.concat([res.base_rates_real.assign(panel="real"),
               res.base_rates_null.assign(panel="null")]).to_csv(rates_csv, index=False)
    text = render_report(res, info, report_date, elapsed_s=time.monotonic() - t0,
                         rss_mb=peak_rss_mb(), git_hash=git_short_hash(), end=end,
                         compare=compare)
    report.write_text(text, encoding="utf-8")
    return report, res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None,
                    help="last bar date to load (default: all)")
    ap.add_argument("--universe", choices=UNIVERSES, default=None,
                    help="index: point-in-time members (default when index_membership has "
                         "rows); active: today's active stocks, unmasked")
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
        report, res = run(args.end, universe=universe, session=session)
    finally:
        session.close()
    counts = res.tests["verdict"].value_counts().to_dict()
    print(f"\n  Report: {report}\n  Universe: {universe}\n  Verdicts: {counts}")


if __name__ == "__main__":
    main()
