"""
KPI summary of simulator runs (master plan §9.1 / §9.2).

Pure: no DB, no network, no file I/O; a ``SimulationResult`` is only read, never changed.
Every return-based figure comes from ``investbot.evaluation.performance`` applied to
``simulator.daily_returns(result, after_tax)`` with ``periods_per_year = 252``; the leading
NaN of that series stays in place (``performance`` ignores non-finite values). Nothing is
dropped, ranked or filtered, and nothing here draws a conclusion: ``go_no_go`` only tabulates
the §9.2 thresholds against the numbers it is given.

Two curves
    ``nav`` is cash plus holdings: tax already **paid** in an April has left the cash, tax
    still **accrued** has not. ``nav_after_tax = nav - tax_accrued``. So ``after_tax=False``
    is "before the tax of the open fiscal year and the terminal liquidation", not "before
    all tax": for a multi-year run the pre-tax curve still carries the tax paid in earlier
    FYs, and ``tax_drag_cagr`` in ``compare`` measures only the unpaid part (the last FY's
    accrual and the liquidation tax). A clean pre-tax figure is a ``tax=False`` run of the
    same schedule (docs/DECISIONS_PENDING.md, "Backtest KPI summary: pre-tax curve").

Units
    Returns, CAGR, volatility, VaR/CVaR and drawdown are fractions (0.05 = 5%). VaR/CVaR are
    **daily** positive losses. ``max_drawdown`` is <= 0 as ``performance.drawdown_stats``
    returns it. ``max_dd_duration_days`` counts trading sessions of the daily curve, not
    calendar days. ``start`` / ``end`` are ``pd.Timestamp``; the money columns are rupees.

Ruined runs (``result.ruined_on`` set, the curve reaches 0 or goes negative)
    Nothing raises. ``daily_returns`` is NaN for every return off a non-positive base, so
    the return series ends with the drop into the first non-positive value (a return
    ``<= -1``) and is NaN after it. On that series: ``sharpe``, ``sortino``, ``ann_vol`` and
    VaR/CVaR are computed over the finite returns up to and including that drop (dominated
    by it; NaN with fewer than 2 finite returns); ``max_drawdown`` is ``<= -1`` (exactly -1
    when the curve hits 0, below -1 when it goes negative), so it reads as a total loss;
    ``cagr`` is NaN when the terminal value is < 0 and -1 when it is exactly 0; ``calmar``
    is NaN when ``cagr`` is NaN, else ``cagr / abs(max_drawdown)``. ``turnover_annual`` is
    NaN when the mean ``nav`` is <= 0. ``monthly_table`` has the drop in its month and NaN
    after it. ``compare`` carries ``ruined_on`` so a reader sees which rows these are.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from investbot.backtest.simulator import SimulationResult, daily_returns
from investbot.evaluation import performance
from investbot.evaluation.metrics import RISK_FREE_RATE_ANNUAL as RISK_FREE_ANNUAL
from investbot.evaluation.metrics import TRADING_DAYS as TRADING_DAYS_PER_YEAR

DAYS_PER_YEAR = 365.25  # calendar years of the CAGR: (end - start).days / 365.25
VAR_LEVEL = 0.95

KPI_KEYS = (
    "start", "end", "years", "initial_nav", "terminal_value", "cagr", "ann_vol", "sharpe",
    "sortino", "max_drawdown", "max_dd_duration_days", "calmar", "var95", "cvar95",
    "total_costs", "total_dividends", "total_tax_paid", "terminal_tax_accrued",
    "liquidation_tax", "turnover_annual", "win_rate", "avg_win_loss_ratio", "profit_factor",
    "n_realisations", "n_stale_episodes",
)

# Master plan §9.2. Applied to after-tax figures only. ``column`` is the name in a ``compare``
# row (or any Series the caller builds); ``magnitude`` compares abs(value) (a drawdown is
# stored <= 0); all comparisons are strict.
GO_NO_GO: dict[str, dict] = {
    "excess_cagr_after_tax": {"column": "excess_cagr_after_tax", "minimum": 0.02, "target": 0.05,
                              "op": ">", "magnitude": False},
    "max_drawdown": {"column": "max_drawdown_after_tax", "minimum": 0.25, "target": 0.20,
                     "op": "<", "magnitude": True},
    "sharpe": {"column": "sharpe_after_tax", "minimum": 0.8, "target": 1.2,
               "op": ">", "magnitude": False},
    "win_rate": {"column": "win_rate_after_tax", "minimum": 0.45, "target": 0.55,
                 "op": ">", "magnitude": False},
    "tax_alpha": {"column": "tax_alpha", "minimum": 0.005, "target": 0.015,
                  "op": ">", "magnitude": False},
}
GO_NO_GO_COLUMNS = ["metric", "value", "minimum", "target", "meets_minimum", "meets_target"]

FY_TAX_COLUMNS = [
    "realized_stcg", "realized_ltcg", "ltcg_exemption_used", "realized_stcl", "realized_ltcl",
    "carried_forward_stcl", "carried_forward_ltcl", "estimated_tax_liability", "paid",
]


def _cagr(terminal_value: float, initial_nav: float, years: float) -> float:
    if not (years > 0 and initial_nav > 0 and math.isfinite(terminal_value)):
        return math.nan
    if terminal_value < 0:
        return math.nan
    return float((terminal_value / initial_nav) ** (1.0 / years) - 1.0)


def kpis(
    result: SimulationResult,
    *,
    after_tax: bool = True,
    rf_annual: float = RISK_FREE_ANNUAL,
) -> dict[str, float]:
    """The §9.1 KPIs of one run, in ``KPI_KEYS`` order.

    ``after_tax=True`` uses the ``nav_after_tax`` curve and
    ``terminal_value = nav - tax_accrued - liquidation_tax`` on the last row; ``False`` uses
    ``nav`` and ``terminal_value = nav`` (see the module docstring for what that omits).
    ``initial_nav`` is the first row of the chosen curve, the base of the return series: a
    decision that executes on the first session leaves its costs and its open-to-close move
    outside the base. ``cagr = (terminal_value / initial_nav) ** (1 / years) - 1`` with
    ``years = (end - start).days / 365.25``, so the liquidation tax is inside the after-tax
    CAGR; ``calmar`` uses that CAGR. NaN where undefined (zero years, no variance, ...).

    Costs, dividends, tax paid, accrued and liquidation tax, turnover, realisation counts
    and ``n_stale_episodes`` are totals of the run and equal for both settings.
    ``turnover_annual`` = sum of ``trades.value`` (rebalance legs and tax-cover sales alike)
    / mean daily ``nav`` / ``years``.

    Win-rate fields come from ``performance.trade_stats`` over ``[r.gain for r in
    result.realisations]``: each realisation, i.e. each lot slice closed by a sale, is one
    trade. The gain is net of trading costs and before tax, so the three fields are the same
    for ``after_tax`` True and False. ``n_realisations`` is the number of such slices.
    """
    daily = result.daily
    if len(daily) == 0:
        raise ValueError("result.daily has no rows")
    curve = daily["nav_after_tax" if after_tax else "nav"]
    r = daily_returns(result, after_tax)
    start, end = daily.index[0], daily.index[-1]
    years = (end - start).days / DAYS_PER_YEAR
    initial_nav = float(curve.iloc[0])
    if after_tax:
        terminal_value = float(curve.iloc[-1]) - float(result.liquidation_tax)
    else:
        terminal_value = float(curve.iloc[-1])
    cagr = _cagr(terminal_value, initial_nav, years)

    finite = r.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    ann_vol = (
        float(finite.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if finite.size >= 2 else math.nan
    )
    dd = performance.drawdown_stats(r)
    var95, cvar95 = performance.var_cvar(r, VAR_LEVEL)

    mean_nav = float(daily["nav"].mean())
    traded = float(result.trades["value"].sum()) if len(result.trades) else 0.0
    turnover = traded / mean_nav / years if years > 0 and mean_nav > 0 else math.nan
    stats = performance.trade_stats([x.gain for x in result.realisations])

    return {
        "start": start,
        "end": end,
        "years": years,
        "initial_nav": initial_nav,
        "terminal_value": terminal_value,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": performance.sharpe(r, TRADING_DAYS_PER_YEAR, rf_annual),
        "sortino": performance.sortino(r, TRADING_DAYS_PER_YEAR, rf_annual),
        "max_drawdown": dd["max_drawdown"],
        "max_dd_duration_days": dd["max_dd_duration"],
        "calmar": performance.calmar(cagr, dd["max_drawdown"]),
        "var95": var95,
        "cvar95": cvar95,
        "total_costs": float(daily["costs"].sum()),
        "total_dividends": float(daily["dividends"].sum()),
        "total_tax_paid": float(daily["tax_paid"].sum()),
        "terminal_tax_accrued": float(daily["tax_accrued"].iloc[-1]),
        "liquidation_tax": float(result.liquidation_tax),
        "turnover_annual": turnover,
        "win_rate": stats["win_rate"],
        "avg_win_loss_ratio": stats["avg_win_loss_ratio"],
        "profit_factor": stats["profit_factor"],
        "n_realisations": len(result.realisations),
        "n_stale_episodes": len(result.stale_marks),
    }


def monthly_table(result: SimulationResult, after_tax: bool = True) -> pd.DataFrame:
    """Year x month (1..12) grid of ``performance.monthly_returns`` plus a ``year`` column.

    ``year`` compounds the months present in that row (``prod(1 + m) - 1``); a month with
    no finite daily return stays NaN. The first month holds only the returns after the
    first session (the base row has none).
    """
    months = performance.monthly_returns(daily_returns(result, after_tax))
    cols = list(range(1, 13))
    if len(months) == 0:
        return pd.DataFrame(columns=[*cols, "year"], dtype=float)
    keyed = pd.Series(
        months.to_numpy(),
        index=pd.MultiIndex.from_arrays([months.index.year, months.index.month]),
    )
    grid = keyed.unstack().reindex(columns=cols)
    grid.index.name = None
    grid.columns.name = None
    year = (1.0 + grid).prod(axis=1, skipna=True, min_count=1) - 1.0
    return pd.concat([grid, year.rename("year")], axis=1)


def fy_tax_table(result: SimulationResult) -> pd.DataFrame:
    """One row per ``result.fiscal_years`` entry, indexed by the FY label.

    Columns are the ``FiscalYearTax`` fields ``realized_stcg``, ``realized_ltcg``,
    ``ltcg_exemption_used``, ``realized_stcl``, ``realized_ltcl``, ``carried_forward_stcl``
    and ``carried_forward_ltcl`` (the carry-forward out of that FY),
    ``estimated_tax_liability``, plus ``paid``: False only for the last FY, which is still
    open (its liability is the run's ``terminal_tax_accrued``, not paid). Empty for ``tax=False``.
    """
    labels = list(result.fiscal_years)
    cols: dict[str, list] = {c: [] for c in FY_TAX_COLUMNS}
    for i, label in enumerate(labels):
        fy = result.fiscal_years[label]
        for c in FY_TAX_COLUMNS[:-1]:
            cols[c].append(float(getattr(fy, c)))
        cols["paid"].append(i < len(labels) - 1)
    out = pd.DataFrame(cols, index=pd.Index(labels, name="fiscal_year"), columns=FY_TAX_COLUMNS)
    out["paid"] = out["paid"].astype(bool)
    return out


def compare(
    results: dict[str, SimulationResult], *, benchmark: str | None = None
) -> pd.DataFrame:
    """One row per label: ``kpis`` after tax and before tax side by side.

    Columns: every KPI with the suffix ``_after_tax``, then every KPI with ``_pre_tax``, then
    ``tax_drag_cagr = cagr_pre_tax - cagr_after_tax`` (what the module docstring says that
    covers). With ``benchmark`` (a key of ``results``, whatever run the caller chose, not
    necessarily an index) also ``excess_cagr_after_tax`` = the row's ``cagr_after_tax`` minus
    the benchmark row's. The last column is ``ruined_on``: the run's ``ruined_on`` as an ISO
    date string, or None (module docstring, "Ruined runs"). Rows keep the order of
    ``results``; nothing is ranked or filtered.
    """
    if benchmark is not None and benchmark not in results:
        raise ValueError(f"benchmark {benchmark!r} is not one of the results: {list(results)}")
    rows: dict[str, dict] = {}
    for label, res in results.items():
        after = kpis(res, after_tax=True)
        pre = kpis(res, after_tax=False)
        row = {f"{k}_after_tax": v for k, v in after.items()}
        row.update({f"{k}_pre_tax": v for k, v in pre.items()})
        row["tax_drag_cagr"] = pre["cagr"] - after["cagr"]
        rows[label] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    if benchmark is not None:
        table["excess_cagr_after_tax"] = (
            table["cagr_after_tax"] - table.loc[benchmark, "cagr_after_tax"]
        )
    table["ruined_on"] = pd.Series(
        [None if res.ruined_on is None else res.ruined_on.isoformat() for res in results.values()],
        index=table.index, dtype=object,
    )
    return table


def tax_alpha(aware: SimulationResult, naive: SimulationResult) -> float:
    """After-tax CAGR of the tax-aware run minus that of the tax-naive run (annual, fraction).

    Both must be ``tax=True`` runs of the same strategy that differ only in the tax handling
    of their decisions; a ``tax=False`` run is a pre-tax run, not a tax-naive one. Put the
    result in a row as ``tax_alpha`` for ``go_no_go``.
    """
    return kpis(aware, after_tax=True)["cagr"] - kpis(naive, after_tax=True)["cagr"]


def go_no_go(row: pd.Series) -> pd.DataFrame:
    """The §9.2 thresholds against one row of after-tax figures (e.g. a ``compare`` row).

    Columns ``metric, value, minimum, target, meets_minimum, meets_target``, one row per
    ``GO_NO_GO`` entry. All comparisons are strict (a value equal to a threshold does not
    meet it). ``max_drawdown`` is shown as a positive loss (``abs`` of the stored figure) and
    must be below the thresholds. A metric the row lacks, or whose value is not finite, gives
    NaN and ``meets_* = None``, never False. Use after-tax columns only: ``excess_cagr_after_tax``
    is measured against **whichever benchmark run the caller passed to ``compare``**; we have
    no NIFTY50 index level, so it is "buy-and-hold NIFTY50" only if the caller supplies that.
    ``tax_alpha`` is absent unless the caller adds it (see ``tax_alpha``). ``win_rate`` is the
    per-lot-slice win rate, which does not change with tax.
    """
    records = []
    for metric, spec in GO_NO_GO.items():
        raw = row.get(spec["column"], math.nan)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            value = math.nan
        elif spec["magnitude"]:
            value = abs(value)
        meets = {}
        for level in ("minimum", "target"):
            if math.isnan(value):
                meets[level] = None
            elif spec["op"] == ">":
                meets[level] = bool(value > spec[level])
            else:
                meets[level] = bool(value < spec[level])
        records.append(
            {
                "metric": metric,
                "value": value,
                "minimum": spec["minimum"],
                "target": spec["target"],
                "meets_minimum": meets["minimum"],
                "meets_target": meets["target"],
            }
        )
    table = pd.DataFrame(records, columns=GO_NO_GO_COLUMNS)
    table["meets_minimum"] = table["meets_minimum"].astype(object)
    table["meets_target"] = table["meets_target"].astype(object)
    return table


def concentration(target_weights: pd.DataFrame) -> pd.Series:
    """Herfindahl index per decision date (``performance.herfindahl``): 1/n for n equal weights."""
    return performance.herfindahl(target_weights)
