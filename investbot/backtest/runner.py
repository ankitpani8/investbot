"""
Backtest driver: one harness signal at one rebalance frequency through the after-tax
simulator, next to the equal-weight universe benchmark and the same strategy with
``tax=False``.

Wires ``inputs.load_backtest_inputs`` (wide frames from the DB), ``strategies`` (signal
panel -> target weights), ``simulator.simulate`` and ``summary`` (KPIs, go/no-go). It is a
tool and presumes no decision horizon: ``signal`` and ``freq`` are required.

The signal panel is built **one asset at a time** (``evaluation.panel.asset_features`` on
that asset's bars up to ``end``, only the requested column kept), so no more than the wide
daily frames and one asset's feature frame are in memory (8 GB RAM). The features are
truncation-invariant, so a value at *t* is what was knowable at *t*'s close; together with
the adapter's timing contract (a row dated *t* executes at the next open) a run with
``end=s`` agrees with the full run on every daily row up to *s*.

Read-only on the DB. Pure compute is in ``render_report`` and ``_clamp_*`` helpers; no
``dropna`` anywhere (missing signals and prices stay NaN and make an asset ineligible).
"""
from __future__ import annotations

import functools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from investbot.backtest import summary
from investbot.backtest.inputs import BacktestInputs, load_backtest_inputs
from investbot.backtest.simulator import FIRST_EQUITY_REGIME_DATE, SimulationResult, simulate
from investbot.backtest.strategies import (
    DEFAULT_QUANTILE,
    equal_weight_universe,
    rebalance_sessions,
    signal_panel,
    top_quantile_weights,
)
from investbot.data.quality import SplitIssue
from investbot.evaluation import loader
from investbot.evaluation.baselines import compute_baselines
from investbot.evaluation.metrics import MIN_NAMES
from investbot.evaluation.panel import SCORE_COLUMNS, SIGNAL_PREFIX, asset_features
from investbot.scoring.technical.confluence import SIGNALS
from investbot.scoring.technical.scorer import MIN_BARS

STRATEGY = "strategy"
TAX_NAIVE = "strategy_tax_naive"        # same schedule, simulated with tax=False (no tax at all)
BENCHMARK = "benchmark"

#: Cost rates of the dated equity regimes before this date are partly copies of today's.
COSTS_VERIFIED_FROM = date(2024, 7, 23)


@dataclass
class BacktestRun:
    """Everything one ``run_backtest`` produced. The three results share inputs and sessions."""

    params: dict
    strategy: SimulationResult
    strategy_tax_naive: SimulationResult
    benchmark: SimulationResult
    compare: pd.DataFrame                      # rows STRATEGY, TAX_NAIVE, BENCHMARK
    go_no_go: pd.DataFrame                     # §9.2 table on the STRATEGY row (after-tax)
    fingerprint: dict
    split_issues: list[SplitIssue]
    missing: list[str]                         # universe tickers with no price rows
    stale_marks: dict[str, pd.DataFrame]       # stale-mark episodes per run label
    weights: pd.DataFrame                      # strategy target weights (decision dates x asset_id)
    benchmark_weights: pd.DataFrame
    complete_from: date                        # membership is known only from this date
    tickers: dict[int, str] = field(default_factory=dict)
    empty_in_window: list[str] = field(default_factory=list)   # members with no bar in the window
    start_clamped: bool = False                # start moved to FIRST_EQUITY_REGIME_DATE


# ──────────────────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def valid_signals() -> tuple[str, ...]:
    """Columns ``run_backtest`` accepts: ``TECH_SCORE``, family scores, ``SIG_*``, ``BASE_*``."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    tiny = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                        index=idx)
    base = tuple(compute_baselines(tiny))
    sig = tuple(SIGNAL_PREFIX + s.name for s in SIGNALS)
    return (*SCORE_COLUMNS, *sig, *base)


def _check_signal(signal: str) -> None:
    valid = valid_signals()
    if signal not in valid:
        raise ValueError(f"unknown signal {signal!r}; valid columns: {', '.join(valid)}")


def _signal_series(session: Session, inp: BacktestInputs, signal: str,
                   end: date | None) -> pd.DataFrame:
    """Sessions x asset_id panel of ``signal``, built one asset at a time from its own bars."""
    frames: dict[str, pd.DataFrame] = {}
    for ticker in inp.tickers.values():
        bars = loader.load_daily_history(session, ticker, end)
        if len(bars) < MIN_BARS:
            continue                                   # too short to score: NaN, never eligible
        feats = asset_features(bars)
        del bars
        if signal in feats.columns:
            frames[ticker] = feats[[signal]]
        del feats
    if not frames:
        raise ValueError(f"signal {signal!r} is not available for any asset in the universe")
    stacked = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    stacked.index = stacked.index.set_names(["date", "ticker"])
    ids = {t: a for a, t in inp.tickers.items()}
    panel = signal_panel(stacked, signal, ids)
    sessions = inp.open_.index
    first = sessions[0]
    stray = panel.index[(panel.index >= first)].difference(sessions)
    if len(stray):
        raise ValueError(f"signal sessions missing from the price frames, e.g. {stray[0].date()} "
                         f"({len(stray)} in all)")
    return panel.reindex(index=sessions, columns=inp.open_.columns)


# ──────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────

def _first_execution(sessions: pd.DatetimeIndex, rebalance: pd.DatetimeIndex) -> date | None:
    """Date of the session the first decision executes on (None: no later session)."""
    if len(rebalance) == 0:
        return None
    i = int(sessions.searchsorted(rebalance[0], side="right"))
    return sessions[i].date() if i < len(sessions) else None


def _drop_empty_columns(inp: BacktestInputs) -> tuple[BacktestInputs, list[str]]:
    """Drop assets with no finite close in the window (their columns are all NaN)."""
    empty = [a for a in inp.close.columns if not inp.close[a].notna().any()]
    if not empty:
        return inp, []
    names = sorted(inp.tickers[a] for a in empty)
    keep = [a for a in inp.close.columns if a not in set(empty)]
    return replace(
        inp, open_=inp.open_[keep], close=inp.close[keep], dividends=inp.dividends[keep],
        membership=inp.membership[keep],
        tickers={a: t for a, t in inp.tickers.items() if a not in set(empty)},
        fmv_2018_01_31={a: v for a, v in inp.fmv_2018_01_31.items() if a not in set(empty)},
    ), names


def _simulate(label: str, *args, **kwargs) -> SimulationResult:
    """``simulate`` with the run's label in the error (a ruined portfolio cannot be simulated)."""
    try:
        return simulate(*args, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"simulating the {label} run failed: {e}. A high-turnover schedule can exhaust the "
            "capital in costs, and the simulator does not handle a portfolio with NAV <= 0 yet "
            "(BACKLOG, task 600 follow-ups); try a lower rebalance frequency."
        ) from e


def _notice(text: str) -> None:
    print(text, file=sys.stderr)


# ──────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────

def run_backtest(
    session: Session,
    *,
    signal: str,
    freq: str,
    start: date | None,
    end: date | None,
    universe: str = "index",
    quantile: float = DEFAULT_QUANTILE,
    initial_capital: float = 500_000.0,
    higher_is_better: bool = True,
    min_names: int = MIN_NAMES,
    index_code: str = "NIFTY50",
    notice: Callable[[str], None] = _notice,
) -> BacktestRun:
    """Run ``signal`` at ``freq`` through the after-tax simulator (see the module docstring).

    ``universe="index"`` runs the stale-membership pre-flight (exits 1 when
    ``index_membership`` differs from the change list, as ``main.py evaluate``) and uses
    point-in-time members; ``"active"`` uses today's active stocks (survivorship bias).
    ``start`` is clamped (and ``notice`` told) when the first execution session would
    precede ``FIRST_EQUITY_REGIME_DATE``. ``min_names``: fewer eligible names on a
    rebalance session means cash (``strategies``). Raises ValueError for an unknown
    signal (listing the valid columns), an empty window or an unknown universe.
    """
    _check_signal(signal)
    if universe == "index":
        from investbot.data.membership_check import require_membership_sync
        if require_membership_sync(session, index_code, what="backtest") is None:
            raise ValueError(f"index_membership has no {index_code} rows; run "
                             "`python main.py setup` first")

    inp = load_backtest_inputs(session, index_code=index_code, universe=universe,
                               start=start, end=end)
    requested_start, clamped = start, False
    sessions = inp.open_.index
    first_exec = _first_execution(sessions, rebalance_sessions(sessions, freq)) if len(sessions) else None
    if first_exec is not None and first_exec < FIRST_EQUITY_REGIME_DATE:
        start, clamped = FIRST_EQUITY_REGIME_DATE, True
        notice(f"start moved to {start} (first execution session {first_exec} is before "
               f"{FIRST_EQUITY_REGIME_DATE}, the first date with a modelled equity regime)")
        inp = load_backtest_inputs(session, index_code=index_code, universe=universe,
                                   start=start, end=end)
    inp, empty = _drop_empty_columns(inp)
    sessions = inp.open_.index
    if len(sessions) == 0 or inp.open_.shape[1] == 0:
        raise ValueError("no price data in the requested window")
    if len(rebalance_sessions(sessions, freq)) == 0:
        raise ValueError(f"no complete {freq!r} period in the window: nothing to rebalance")

    panel = _signal_series(session, inp, signal, end)
    weights = top_quantile_weights(panel, inp.membership, freq=freq, quantile=quantile,
                                   min_names=min_names, higher_is_better=higher_is_better)
    bench_w = equal_weight_universe(inp.membership, inp.close, freq=freq, min_names=min_names)
    del panel

    sim = dict(initial_capital=initial_capital, fmv_2018_01_31=inp.fmv_2018_01_31)
    results = {
        STRATEGY: _simulate(STRATEGY, weights, inp.open_, inp.close, inp.dividends, tax=True, **sim),
        TAX_NAIVE: _simulate(TAX_NAIVE, weights, inp.open_, inp.close, inp.dividends, tax=False,
                             **sim),
        BENCHMARK: _simulate(BENCHMARK, bench_w, inp.open_, inp.close, inp.dividends, tax=True,
                             **sim),
    }
    table = summary.compare(results, benchmark=BENCHMARK)
    params = {
        "signal": signal, "freq": freq, "universe": universe, "index_code": index_code,
        "quantile": quantile, "initial_capital": initial_capital,
        "higher_is_better": higher_is_better, "min_names": min_names,
        "requested_start": requested_start, "start": start, "end": end,
        "first_session": sessions[0].date(), "last_session": sessions[-1].date(),
        "n_sessions": len(sessions), "n_assets": inp.open_.shape[1],
        "n_rebalances": len(weights),
    }
    return BacktestRun(
        params=params, strategy=results[STRATEGY], strategy_tax_naive=results[TAX_NAIVE],
        benchmark=results[BENCHMARK], compare=table,
        go_no_go=summary.go_no_go(table.loc[STRATEGY]), fingerprint=inp.fingerprint,
        split_issues=inp.split_issues, missing=inp.missing,
        stale_marks={k: v.stale_marks for k, v in results.items()},
        weights=weights, benchmark_weights=bench_w, complete_from=inp.complete_from,
        tickers=inp.tickers, empty_in_window=empty, start_clamped=clamped,
    )


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

SECTIONS = (
    "Parameters",
    "Comparison",
    "Go / no-go (after-tax strategy, master plan §9.2)",
    "Fiscal-year tax (strategy)",
    "Monthly returns (strategy, after tax)",
    "Concentration (Herfindahl)",
    "Data caveats",
)

_PCT = {"cagr", "ann_vol", "max_drawdown", "var95", "cvar95", "win_rate", "turnover_annual"}


def _fmt(v, pct: bool = False, digits: int = 2) -> str:
    if v is None or isinstance(v, str):
        return "n/a" if v is None else v
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "n/a"
    return f"{x * 100:.{digits}f}%" if pct else f"{x:,.{digits}f}"


def _md_table(frame: pd.DataFrame, index_name: str = "", formats: dict | None = None) -> str:
    """Markdown table of ``frame``; ``formats`` maps a column to a ``value -> str`` callable."""
    formats = formats or {}
    head = [index_name, *[str(c) for c in frame.columns]]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for label, row in frame.iterrows():
        cells = [str(label)]
        for c in frame.columns:
            cells.append(formats.get(c, _fmt)(row[c]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _comparison_table(run: BacktestRun) -> str:
    c = run.compare
    clean = c.loc[TAX_NAIVE, "cagr_after_tax"]       # tax=False run: no tax at all
    rows = {}
    for label in (STRATEGY, TAX_NAIVE, BENCHMARK):
        r = c.loc[label]
        rows[label] = {
            "CAGR after tax": r["cagr_after_tax"],
            "CAGR pre-tax": r["cagr_pre_tax"],
            "Tax drag (pre − after)": r["tax_drag_cagr"],
            "Excess vs benchmark": r["excess_cagr_after_tax"],
            "Vol": r["ann_vol_after_tax"],
            "Sharpe": r["sharpe_after_tax"],
            "Max DD": r["max_drawdown_after_tax"],
            "Win rate": r["win_rate_after_tax"],
            "Turnover / yr": r["turnover_annual_after_tax"],
            "Costs (₹)": r["total_costs_after_tax"],
            "Tax paid (₹)": r["total_tax_paid_after_tax"],
            "Terminal (₹)": r["terminal_value_after_tax"],
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    pct = {k: (lambda v: _fmt(v, pct=True)) for k in
           ("CAGR after tax", "CAGR pre-tax", "Tax drag (pre − after)", "Excess vs benchmark",
            "Vol", "Max DD", "Win rate", "Turnover / yr")}
    pct.update({k: (lambda v: _fmt(v, digits=0)) for k in ("Costs (₹)", "Tax paid (₹)", "Terminal (₹)")})
    labels = {STRATEGY: "strategy (after tax)", TAX_NAIVE: "strategy, `tax=False`",
              BENCHMARK: "benchmark (equal-weight universe, after tax)"}
    table.index = [labels[i] for i in table.index]
    gap = c.loc[STRATEGY, "cagr_after_tax"] - clean
    note = (f"\nClean tax drag of the strategy (CAGR of the `tax=False` run minus the after-tax "
            f"CAGR): {_fmt(-gap, pct=True)}. The \"pre-tax\" column is `nav` before the open "
            "FY's accrual and the liquidation tax only; tax paid in earlier FYs has already "
            "left it (DECISIONS_PENDING \"Backtest KPI summary: pre-tax curve\"). The "
            "`tax=False` row is the clean pre-tax figure.")
    return _md_table(table, "run", pct) + "\n" + note


def _go_table(run: BacktestRun) -> str:
    g = run.go_no_go.set_index("metric")
    pct = {"excess_cagr_after_tax", "max_drawdown", "win_rate", "tax_alpha"}
    lines = ["| metric | value | minimum | target | meets minimum | meets target |",
             "|---|---|---|---|---|---|"]
    for m, r in g.iterrows():
        is_pct = m in pct
        lines.append(f"| {m} | {_fmt(r['value'], pct=is_pct)} | {_fmt(r['minimum'], pct=is_pct)} | "
                     f"{_fmt(r['target'], pct=is_pct)} | {_fmt(r['meets_minimum'])} | "
                     f"{_fmt(r['meets_target'])} |")
    return "\n".join(lines) + (
        "\n\n`tax_alpha` needs a tax-aware and a tax-naive `tax=True` run, which this driver "
        "does not produce (the `tax=False` run is a pre-tax run, not a tax-naive one), so it "
        "is n/a. Excess CAGR is against the equal-weight universe, not an index level.")


def _fy_table(run: BacktestRun) -> str:
    t = summary.fy_tax_table(run.strategy)
    if len(t) == 0:
        return "No fiscal years (no execution)."
    return _md_table(t, "fiscal year", {c: (lambda v: _fmt(v, digits=0)) for c in t.columns
                                        if c != "paid"})


def _monthly_table(run: BacktestRun) -> str:
    t = summary.monthly_table(run.strategy, after_tax=True)
    if len(t) == 0:
        return "No monthly returns."
    t = t.copy()
    t.columns = [str(c) for c in t.columns]
    return _md_table(t, "year", {c: (lambda v: _fmt(v, pct=True, digits=1)) for c in t.columns})


def _hhi_table(run: BacktestRun) -> str:
    rows = {}
    for label, w in (("strategy", run.weights), ("benchmark", run.benchmark_weights)):
        h = summary.concentration(w)
        finite = h[np.isfinite(h.to_numpy())]
        n_cash = int(len(h) - len(finite))
        rows[label] = {
            "decisions": len(h), "cash rows": n_cash,
            "mean": finite.mean() if len(finite) else math.nan,
            "median": finite.median() if len(finite) else math.nan,
            "min": finite.min() if len(finite) else math.nan,
            "max": finite.max() if len(finite) else math.nan,
            "≈ names held (1 / mean)": 1.0 / finite.mean() if len(finite) else math.nan,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    f = {c: (lambda v: _fmt(v, digits=3)) for c in ("mean", "median", "min", "max")}
    f["≈ names held (1 / mean)"] = lambda v: _fmt(v, digits=1)
    f["decisions"] = f["cash rows"] = lambda v: _fmt(v, digits=0)
    return _md_table(table, "weights", f)


def _caveats(run: BacktestRun) -> str:
    p = run.params
    n_members = len(run.tickers) + len(run.missing)
    dates = run.weights.index
    n_pre = int((dates < pd.Timestamp(run.complete_from)).sum()) if len(dates) else 0
    share = n_pre / len(dates) if len(dates) else math.nan
    lines = []
    if p["universe"] == "active":
        lines.append("- **Universe `active` is today's active stocks**: survivorship bias, "
                     "every name is a member on every date.")
    lines.append(
        f"- **Universe members without prices: {len(run.missing)}** of {n_members}"
        + (f" ({', '.join(sorted(run.missing))})" if run.missing else "")
        + ". They never enter either portfolio, so a past index member without stored prices "
        "is silently absent (survivorship bias until the past-member ingest has run).")
    if run.empty_in_window:
        lines.append(f"- Members with prices but no bar inside the window, dropped: "
                     f"{len(run.empty_in_window)} ({', '.join(run.empty_in_window)}).")
    lines.append(
        f"- **Rebalance dates before {run.complete_from} (membership change list complete from "
        f"then): {n_pre} of {len(dates)} ({_fmt(share, pct=True, digits=1)})**; before it, "
        "membership comes from stored stints that are not sourced completely.")
    lines.append(f"- **Unadjusted-split issues (`detect_unadjusted_splits`): "
                 f"{len(run.split_issues)}**; prices are not adjusted or masked, so a flagged "
                 "split breaks the share basis of a held position.")
    s = run.stale_marks
    lines.append(
        f"- **Stale-mark episodes** (a held asset whose bars stop or have a gap, marked at the "
        f"last close): strategy {len(s[STRATEGY])}, benchmark {len(s[BENCHMARK])}.")
    for label in (STRATEGY, BENCHMARK):
        for r in s[label].head(10).itertuples(index=False):
            lines.append(f"  - {label}: {run.tickers.get(int(r.asset_id), r.asset_id)} stale from "
                         f"{pd.Timestamp(r.first_stale_date).date()} (last close "
                         f"{r.last_valid_close:,.2f} on {pd.Timestamp(r.last_valid_date).date()})")
    lines += [
        f"- Non-STT cost rates before {COSTS_VERIFIED_FROM} (stamp duty, exchange charges, "
        "SEBI fee, GST / service tax) are **unverified** copies of today's rates "
        "(BACKLOG \"Transaction-cost regime dating\").",
        "- **Dividend income tax is not modelled**: dividends are credited gross.",
        "- **Stale marks are held at the last close** (no forced exit on delisting or merger; "
        "decision pending, DECISIONS_PENDING \"Backtest simulator: stale marks\").",
        "- No slippage model: fills are at the stored open.",
    ]
    return "\n".join(lines)


def render_report(run: BacktestRun) -> str:
    """Markdown report of ``run`` (pure). Section headings are ``SECTIONS``."""
    p = run.params
    fp = run.fingerprint
    start_note = ""
    if run.start_clamped:
        start_note = (f"; start requested {p['requested_start'] or 'earliest'}, moved to "
                      f"{p['start']}, the first date with a modelled equity regime")
    params = [
        f"- Signal: `{p['signal']}` ({'higher' if p['higher_is_better'] else 'lower'} is better)",
        f"- Rebalance frequency: `{p['freq']}` ({p['n_rebalances']} decision dates)",
        f"- Selection: top {_fmt(p['quantile'], pct=True, digits=0)} equal-weight, cash below "
        f"{p['min_names']} eligible names",
        f"- Universe: `{p['universe']}`" + (f" ({p['index_code']}, point in time)"
                                            if p["universe"] == "index" else "")
        + f", {p['n_assets']} assets with prices",
        f"- Window: {p['first_session']} to {p['last_session']} ({p['n_sessions']} sessions"
        f"{start_note})",
        f"- Initial capital: ₹{p['initial_capital']:,.0f}",
        f"- DB: latest price date {fp.get('db_max_date')}, {fp.get('db_price_rows')} price rows, "
        f"{fp.get('db_action_rows')} corporate-action rows",
    ]
    parts = [
        f"# Backtest: {p['signal']} / {p['freq']}",
        "Tool output, not a result: the numbers carry the caveats below.",
        f"## {SECTIONS[0]}", "\n".join(params),
        f"## {SECTIONS[1]}", _comparison_table(run),
        f"## {SECTIONS[2]}", _go_table(run),
        f"## {SECTIONS[3]}", _fy_table(run),
        f"## {SECTIONS[4]}", _monthly_table(run),
        f"## {SECTIONS[5]}", _hhi_table(run),
        f"## {SECTIONS[6]}", _caveats(run),
    ]
    return "\n\n".join(parts) + "\n"
