"""
After-tax portfolio simulator (master plan Phase 7, §9.1): target weights -> trades,
FIFO lots, costs, tax and a daily pre-tax / after-tax equity curve.

Pure compute: no DB, no network, inputs are never mutated. The simulator does not know
where the weights come from and does not presume a decision horizon: a daily, weekly or
monthly schedule is just a ``target_weights`` frame with more or fewer rows.

Timing (the lookahead contract)
    Weights dated *t* are decided after the close of session *t* and executed at the
    **open of the next session after t**. A decision date that is not a session is handled
    like the last session before it (a Saturday decision executes on Monday's open); one
    dated before the first session executes at the first open. When several decision dates
    map to the same execution session the latest one wins. A decision whose execution
    session is past the last session is never executed. So nothing in the ``daily`` row of
    a session *s* depends on a ``target_weights`` row dated *s* or later, nor on any price,
    dividend or weight after *s*: truncating every input at *s* reproduces rows <= *s*.

Order of events on a session *s*
    1. Fiscal-year rollover (first session on or after 1 April): every FY closed since the
       previous session gets ``compute_fiscal_year_tax`` (carry-forward chained) and
       ``dividend_tax_for_fy`` on its gross dividends.
    2. Dividends: ``qty held at the previous close x dividend`` (gross) is added to cash.
    3. Tax payment (``tax=True``): the closed FY's ``estimated_tax_liability`` plus its
       dividend tax is paid from cash. If cash is short, every holding with a valid open is sold pro rata at the open
       so the proceeds net of sell costs cover the shortfall; those sales are ordinary
       realisations of the new FY. If even a full sale cannot cover it, cash goes negative.
    4. Rebalance, if a decision executes at *s* and the portfolio is not ruined: NAV at the
       open = cash + sum(qty x open) (an asset with no valid open is valued at its last valid
       close). Target quantity = max(0, w x NAV / open). Sells first (``sell_fifo`` with ``estimate_costs("SELL", ...)``), then
       buys; if buys plus their costs exceed the cash, all buys are scaled by one common
       factor so cash ends >= 0. One order per asset leg for brokerage. A leg whose
       |change| is below ``_REL_QTY_TOL`` of the larger of held and target quantity (float
       noise) or whose value is below ``min_trade_value`` is skipped. A sell never asks
       ``sell_fifo`` for more than the held quantity.
    5. Ruin: if the NAV at the open of a rebalance session is ``<= RUIN_NAV``, *s* is
       ``ruined_on`` and the decision is not executed. From then on no buy is placed again:
       on every session on or after ``ruined_on`` (from the next session's open when the
       ruin is found at a close) every holding with a valid open is sold in full, whatever
       ``target_weights`` says and regardless of ``min_trade_value``; a holding with no valid
       open is sold at the first later session that has one.
    6. Close: ``nav`` = cash + sum(qty x close). A held asset with no valid close is marked
       at its last valid close (``stale_marks``). If ``nav <= RUIN_NAV`` and the portfolio
       is not ruined yet, *s* is ``ruined_on``.

Ruin (``ruined_on``)
    A schedule whose costs exhaust the capital (e.g. daily rebalancing of a small capital
    against the flat brokerage) is a reported outcome, not an error: the run continues to
    the last session as a liquidated portfolio. Dividends are still credited, the FY tax is
    still computed and paid when due (cash may go negative there, as for any run), and
    ``nav`` may be negative. Ruin on *s* uses only the NAV from prices up to *s* (the open
    of *s* for a rebalance session, otherwise its close), so it keeps the lookahead
    contract. The convention is reversible (docs/DECISIONS_PENDING.md, "Backtest simulator:
    ruin convention").

Quantities are fractional. Stored ``open``/``close`` are on today's share basis (Yahoo
restates them for later splits and bonuses), so the whole-share counts an investor held
historically cannot be recovered; rounding to whole units on today's basis would be
arbitrary. ``close`` is the stored close, never ``adj_close``: dividends are paid in cash
via ``dividends``.

Costs and tax regime: every leg uses ``tax_rules.regime_for("equity", trade_date)``. Only
the STT is dated before 2018; the non-STT charges (stamp duty, exchange charge, SEBI fee,
GST) of the older regimes are copies of today's rates and are unverified (BACKLOG
"Transaction-cost regime dating"). ``regime_for`` has no equity regime before 2004-10-01,
so ``simulate`` raises ValueError if the first execution session is before that date.

Tax (``tax=True``): every ``Realisation`` is collected; at each FY end the FY's tax is
computed with the previous FY's ``carry_forward_out`` and paid on the first session on or
after 1 April. ``tax_accrued`` is the liability of the current FY's realisations so far
(recomputed only on days with realisations) plus the dividend tax of the current FY's
dividends so far (recomputed only on days with a dividend); ``nav_after_tax = nav -
tax_accrued``.

Dividend tax (``tax=True`` and ``dividend_slab_rate`` not None): gross dividends are summed
per FY of their ex-date (the session they are credited on) and taxed with
``dividend_tax.dividend_tax_for_fy(fy, gross, slab_rate=dividend_slab_rate,
surcharge_rate=surcharge_rate)``: exempt before FY 2016-17, s.115BBDA (10% above ₹10 lakh)
in FY 2016-17 to 2019-20, the slab rate from FY 2020-21. The ₹10 lakh threshold sees only
this portfolio's dividends. The tax joins the FY's capital-gains tax in the April payment
and in ``tax_accrued``, but is kept apart from ``FiscalYearTax`` (``result.dividend_tax``).
``liquidation_tax`` has no dividend part: a terminal sale pays no dividend.
``dividend_slab_rate=None`` leaves dividends untaxed (``result.dividend_tax`` still lists
the gross per FY, with 0.0 tax).
``fmv_2018_01_31`` (per asset_id, same share basis as the prices) is passed to
``sell_fifo`` for the grandfathered cost of pre-2018-02-01 lots. ``tax=False`` still
charges costs and collects realisations but never computes or pays tax.

Not modelled: the s.57 interest-deduction cap, the basic exemption and s.87A rebate for
dividend income, and the DDT the company paid before FY 2020-21 (the stored dividend is
what the shareholder received; ``dividends`` in ``daily`` are reported gross of the
shareholder's own dividend tax); TDS on dividends (a prepayment of the same tax); DP charges; the TDS credit (equity has none);
interest on cash; advance-tax instalments (the whole liability is paid once, in April).
A held asset whose prices stop is kept at its last close; there is no forced liquidation
rule (docs/DECISIONS_PENDING.md, "Backtest simulator").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from investbot.trading import tax_rules
from investbot.trading.dividend_tax import DIVIDEND_SLAB_RATE, dividend_tax_for_fy
from investbot.trading.lot_tracker import Lot, Realisation, sell_fifo
from investbot.trading.tax_ledger import (
    CarryForward,
    FiscalYearTax,
    compute_fiscal_year_tax,
    incremental_tax,
)
from investbot.trading.transaction_costs import deductible_costs, estimate_costs

FIRST_EQUITY_REGIME_DATE = date(2004, 10, 1)  # regime_for("equity", d) raises before this
_WEIGHT_SUM_TOL = 1e-9
_REL_QTY_TOL = 1e-9  # a leg smaller than this share of max(held, target) is float noise
_SCALE_MARGIN = 1e-12  # float headroom (share of the legs' value) so cash ends >= 0
RUIN_NAV: float = 0.0  # NAV at or below this (at a rebalance open or a close) ruins the run

DAILY_COLUMNS = [
    "cash",
    "holdings_value",
    "nav",
    "costs",
    "dividends",
    "tax_paid",
    "tax_accrued",
    "nav_after_tax",
]
TRADE_COLUMNS = ["date", "asset_id", "side", "quantity", "price", "value", "costs"]
STALE_COLUMNS = ["asset_id", "first_stale_date", "last_valid_date", "last_valid_close"]


@dataclass
class SimulationResult:
    """Output of ``simulate``.

    ``fiscal_years`` holds every FY from the first execution session's FY to the last
    session's FY. All but the last are closed and paid; the last one is still open (its
    liability, plus its dividend tax, is the final ``tax_accrued``, not paid). Empty when
    ``tax=False``. ``dividend_tax`` maps the same FY labels to (gross dividends, dividend
    tax); it is not part of ``FiscalYearTax.estimated_tax_liability``.
    Terminal after-tax value = ``nav - tax_accrued - liquidation_tax`` on the last row;
    ``liquidation_tax`` is negative when the terminal sale would harvest a loss against
    gains already realised in the last FY. ``trades`` lists rebalance legs and the April
    tax-cover sales in execution order. ``stale_marks`` has one row per stale episode.
    ``ruined_on`` is the first session on which the portfolio counted as ruined (NAV at a
    rebalance open or at a close ``<= RUIN_NAV``), None if it never was; from it on the
    run is a liquidated portfolio (see the module docstring).
    """

    daily: pd.DataFrame
    trades: pd.DataFrame
    realisations: list[Realisation] = field(default_factory=list)
    fiscal_years: dict[str, FiscalYearTax] = field(default_factory=dict)
    stale_marks: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=STALE_COLUMNS))
    liquidation_tax: float = 0.0
    ruined_on: date | None = None
    dividend_tax: dict[str, tuple[float, float]] = field(default_factory=dict)


def _fy_start(label: str) -> int:
    return int(label.split("-")[0])


def _fy_label(start: int) -> str:
    return f"{start}-{(start + 1) % 100:02d}"


def _check_frame(name: str, frame: pd.DataFrame, ref: pd.DataFrame) -> None:
    if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
        raise ValueError(f"{name} must have the same index and columns as open_")


def _execution_schedule(
    target_weights: pd.DataFrame, sessions: pd.DatetimeIndex, assets: pd.Index
) -> dict[int, np.ndarray]:
    """Map each executing decision to its session position: {session_idx: weight vector}."""
    unknown = target_weights.columns.difference(assets)
    if len(unknown):
        raise ValueError(f"target_weights has assets with no prices: {list(unknown)}")
    w = target_weights.reindex(columns=assets).fillna(0.0)
    values = w.to_numpy(dtype=float, copy=True)
    if (values < 0).any():
        raise ValueError("target weights must be >= 0")
    sums = values.sum(axis=1)
    if (sums > 1.0 + _WEIGHT_SUM_TOL).any():
        raise ValueError("target weight rows must sum to <= 1 (the remainder is cash)")
    dates = pd.DatetimeIndex(w.index)
    if dates.has_duplicates:
        raise ValueError("target_weights has duplicate decision dates")
    order = np.argsort(dates.to_numpy(), kind="stable")
    # last session <= decision date, then the next one; before the first session -> 0
    exec_idx = sessions.searchsorted(dates.to_numpy(), side="right")
    schedule: dict[int, np.ndarray] = {}
    for i in order:  # ascending decision dates, so the latest decision wins
        s = int(exec_idx[i])
        if s < len(sessions):
            schedule[s] = values[i]
    return schedule


def _compressed(sums: dict[tuple[date, str, bool], float]) -> list[Realisation]:
    """One synthetic realisation per (sell_date, term, gain > 0) with the summed gain.

    ``compute_fiscal_year_tax`` reads only those fields of a realisation (the regime comes
    from the sell date, the bucket from the term and the sign), so the liability equals the
    one of the full list up to float summation order, at a fraction of the cost.
    """
    return [
        Realisation(
            lot_id="", asset_id=-1, buy_date=d, sell_date=d, quantity=0.0, cost_basis=0.0,
            sale_value=0.0, sell_costs=0.0, gain=g, term=term,
        )
        for (d, term, _), g in sums.items()
    ]


def _cost_line(side: str, regime: tax_rules.TaxRegimeParams) -> tuple[float, float]:
    """``estimate_costs(side, v).total == a * v + b`` for one order: returns (a, b)."""
    b = estimate_costs(side, 0.0, regime).total
    a = estimate_costs(side, 1.0, regime).total - b
    return a, b


def simulate(
    target_weights: pd.DataFrame,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
    *,
    initial_capital: float = 500_000.0,
    tax: bool = True,
    fmv_2018_01_31: dict[int, float] | None = None,
    min_trade_value: float = 0.0,
    surcharge_rate: float = 0.0,
    dividend_slab_rate: float | None = DIVIDEND_SLAB_RATE,
) -> SimulationResult:
    """Replay a target-weight schedule on daily prices (see the module docstring).

    ``target_weights``: decision dates x asset_id, weights >= 0, row sum <= 1 (+1e-9), the
    remainder is cash, NaN = 0. ``open_`` / ``close`` / ``dividends``: sessions x asset_id
    (same index and columns); ``dividends`` is gross cash per share on the ex-date bar.
    ``dividend_slab_rate``: slab rate on dividends from FY 2020-21 (module docstring);
    None = dividends untaxed. ``surcharge_rate`` applies to both capital-gains and
    dividend tax. Raises ValueError for malformed inputs, a rate outside [0, 1] or a first
    execution before 2004-10-01.
    """
    _check_frame("close", close, open_)
    if dividends is not None:
        _check_frame("dividends", dividends, open_)
    sessions = pd.DatetimeIndex(open_.index)
    if len(sessions) == 0:
        raise ValueError("open_ has no sessions")
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("sessions must be strictly increasing")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if min_trade_value < 0:
        raise ValueError("min_trade_value must be >= 0")
    if dividend_slab_rate is not None and not 0.0 <= dividend_slab_rate <= 1.0:
        raise ValueError("dividend_slab_rate must be in [0, 1] or None")

    assets = open_.columns
    asset_ids = [int(a) for a in assets]
    n_sess, n_assets = len(sessions), len(assets)
    schedule = _execution_schedule(target_weights, sessions, assets)
    session_dates = [ts.date() for ts in sessions]
    if schedule:
        first_exec = session_dates[min(schedule)]
        if first_exec < FIRST_EQUITY_REGIME_DATE:
            raise ValueError(
                f"first execution session {first_exec} is before {FIRST_EQUITY_REGIME_DATE}: "
                "no equity tax/cost regime is modelled there"
            )
        first_fy_start = _fy_start(tax_rules.fiscal_year(first_exec))
    else:
        first_fy_start = None

    opens = open_.to_numpy(dtype=float, copy=True)
    closes = close.to_numpy(dtype=float, copy=True)
    divs = (
        np.zeros((n_sess, n_assets))
        if dividends is None
        else np.nan_to_num(dividends.to_numpy(dtype=float, copy=True), nan=0.0)
    )
    fmv = fmv_2018_01_31 or {}

    # state
    cash = float(initial_capital)
    qty = np.zeros(n_assets)
    lots: list[list[Lot]] = [[] for _ in range(n_assets)]
    lot_counter = [0]
    mark = np.full(n_assets, np.nan)  # last valid close (or buy price before the first close)
    mark_date: list[date | None] = [None] * n_assets
    stale_open: set[int] = set()  # positions of held assets inside a stale-mark episode
    stale_rows: list[dict] = []
    realisations: list[Realisation] = []
    fy_realisations: list[Realisation] = []
    fy_sums: dict[tuple[date, str, bool], float] = {}  # compressed fy_realisations for tax_accrued
    fiscal_years: dict[str, FiscalYearTax] = {}
    carry_in: CarryForward | None = None
    tax_accrued = 0.0
    cg_accrued = 0.0  # capital-gains part of tax_accrued
    div_accrued = 0.0  # dividend part of tax_accrued
    fy_dividends = 0.0  # gross dividends of the open FY
    dividend_tax: dict[str, tuple[float, float]] = {}
    ruined_on: date | None = None
    regime_cache: dict[date, tax_rules.TaxRegimeParams] = {}

    def div_tax(label: str, gross: float) -> float:
        if dividend_slab_rate is None:
            return 0.0
        return dividend_tax_for_fy(
            label, gross, slab_rate=dividend_slab_rate, surcharge_rate=surcharge_rate
        )

    trade_rows: dict[str, list] = {c: [] for c in TRADE_COLUMNS}
    out = {c: np.zeros(n_sess) for c in DAILY_COLUMNS}

    def regime(d: date) -> tax_rules.TaxRegimeParams:
        if d not in regime_cache:
            regime_cache[d] = tax_rules.regime_for("equity", d)
        return regime_cache[d]

    def record(d: date, j: int, side: str, q: float, price: float, value: float, c: float) -> None:
        for col, v in zip(TRADE_COLUMNS, (d, asset_ids[j], side, q, price, value, float(c))):
            trade_rows[col].append(v)

    def sell(j: int, q: float, price: float, d: date) -> float:
        """Sell q units of asset j at price; returns the cost total. Updates cash/qty/lots."""
        nonlocal cash
        q, price = float(q), float(price)
        reg = regime(d)
        value = q * price
        costs = estimate_costs("SELL", value, reg)
        rs, new_lots = sell_fifo(
            lots[j], q, price, d, reg, sell_costs=costs, fmv_2018_01_31=fmv.get(asset_ids[j])
        )
        lots[j] = [lot for lot in new_lots if lot.remaining_qty > 0]
        realisations.extend(rs)
        fy_realisations.extend(rs)
        for r in rs:
            key = (r.sell_date, r.term, r.gain > 0)
            fy_sums[key] = fy_sums.get(key, 0.0) + r.gain
        qty[j] = 0.0 if not lots[j] else qty[j] - q
        cash += value - costs.total
        record(d, j, "SELL", q, price, value, costs.total)
        return costs.total

    def buy(j: int, q: float, price: float, d: date) -> float:
        nonlocal cash
        q, price = float(q), float(price)
        reg = regime(d)
        value = q * price
        costs = estimate_costs("BUY", value, reg)
        lot_counter[0] += 1
        lots[j].append(
            Lot(
                lot_id=f"{asset_ids[j]}-{lot_counter[0]}",
                asset_id=asset_ids[j],
                buy_date=d,
                buy_price=price,
                quantity=q,
                remaining_qty=q,
                buy_costs=deductible_costs(costs),
            )
        )
        qty[j] += q
        cash -= value + costs.total
        if math.isnan(mark[j]):
            mark[j] = price
        record(d, j, "BUY", q, price, value, costs.total)
        return costs.total

    prev_fy: str | None = None
    for s in range(n_sess):
        d = session_dates[s]
        op = opens[s]
        tradable = ~np.isnan(op) & (op > 0)
        day_costs = 0.0
        day_tax_paid = 0.0
        n_realised = len(fy_realisations)
        fy = tax_rules.fiscal_year(d)

        # 1. fiscal-year rollover
        due = 0.0
        if prev_fy is not None and fy != prev_fy:
            for start in range(_fy_start(prev_fy), _fy_start(fy)):
                if not tax or first_fy_start is None or start < first_fy_start:
                    continue
                label = _fy_label(start)
                closed = compute_fiscal_year_tax(
                    fy_realisations if label == prev_fy else [],
                    label,
                    carry_in,
                    surcharge_rate=surcharge_rate,
                )
                fiscal_years[label] = closed
                carry_in = closed.carry_forward_out
                due += closed.estimated_tax_liability
                gross_div = fy_dividends if label == prev_fy else 0.0
                tax_div = div_tax(label, gross_div)
                dividend_tax[label] = (gross_div, tax_div)
                due += tax_div
            fy_realisations = []
            fy_sums = {}
            n_realised = 0
            tax_accrued = 0.0
            cg_accrued = 0.0
            div_accrued = 0.0
            fy_dividends = 0.0
        prev_fy = fy

        # 2. dividends on the ex-date bar, on the quantity held at the previous close
        day_div = float(qty @ divs[s])
        cash += day_div
        fy_dividends += day_div

        # 3. tax payment, selling pro rata at the open if cash is short
        if due > 0:
            shortfall = due - cash
            held = np.flatnonzero((qty > 0) & tradable)
            if shortfall > 0 and held.size:
                a, b = _cost_line("SELL", regime(d))
                gross = float(qty[held] @ op[held])
                frac = (shortfall + held.size * b) / (gross * (1.0 - a)) + _SCALE_MARGIN
                frac = min(1.0, frac)
                for j in held:
                    q = qty[j] if frac >= 1.0 else qty[j] * frac
                    day_costs += sell(j, q, op[j], d)
            cash -= due
            day_tax_paid = due

        # 4. rebalance
        weights = schedule.get(s) if ruined_on is None else None
        if weights is not None:
            value_at = np.where(tradable, op, mark)
            held_value = np.where(qty > 0, qty * np.nan_to_num(value_at, nan=0.0), 0.0)
            nav_open = cash + float(held_value.sum())
            if nav_open <= RUIN_NAV:
                ruined_on = d
                weights = None
        if weights is not None:
            target = np.where(tradable, weights * nav_open / np.where(tradable, op, 1.0), qty)
            target = np.maximum(target, 0.0)
            delta = target - qty
            noise = np.abs(delta) <= _REL_QTY_TOL * np.maximum(np.abs(qty), np.abs(target))
            small = np.abs(delta) * np.where(tradable, op, 0.0) < min_trade_value
            skip = ~tradable | noise | small
            for j in np.flatnonzero(~skip & (delta < 0)):
                q = qty[j] if target[j] <= _REL_QTY_TOL * qty[j] else min(-delta[j], qty[j])
                day_costs += sell(j, q, op[j], d)
            buys = np.flatnonzero(~skip & (delta > 0))
            if buys.size:
                values = delta[buys] * op[buys]
                a, b = _cost_line("BUY", regime(d))
                need = float(values.sum()) * (1.0 + a) + buys.size * b
                scale = 1.0
                if need > cash:
                    scale = max(0.0, (cash - buys.size * b) / (float(values.sum()) * (1.0 + a)))
                    scale *= 1.0 - _SCALE_MARGIN  # float headroom so cash ends >= 0
                if scale > 0:
                    for j, v in zip(buys, values):
                        day_costs += buy(j, v * scale / op[j], op[j], d)

        # 5. ruined: sell every holding with a valid open in full, never buy again
        if ruined_on is not None:
            for j in np.flatnonzero((qty > 0) & tradable):
                day_costs += sell(j, qty[j], op[j], d)

        # 6. close: marks, stale marks, NAV
        cl = closes[s]
        valid_close = ~np.isnan(cl) & (cl > 0)
        for j in np.flatnonzero(valid_close):
            mark[j] = cl[j]
            mark_date[j] = d
            stale_open.discard(j)
        for j in np.flatnonzero(~valid_close & (qty > 0)):
            if j not in stale_open:
                stale_open.add(j)
                stale_rows.append(
                    {
                        "asset_id": asset_ids[j],
                        "first_stale_date": d,
                        "last_valid_date": mark_date[j],
                        "last_valid_close": float(mark[j]),
                    }
                )
        holdings = float(np.where(qty > 0, qty * np.nan_to_num(mark, nan=0.0), 0.0).sum())

        if tax and len(fy_realisations) != n_realised:
            cg_accrued = compute_fiscal_year_tax(
                _compressed(fy_sums), fy, carry_in, surcharge_rate=surcharge_rate
            ).estimated_tax_liability
        if tax and day_div != 0.0:
            div_accrued = div_tax(fy, fy_dividends)
        tax_accrued = cg_accrued + div_accrued

        nav = cash + holdings
        if ruined_on is None and nav <= RUIN_NAV:
            ruined_on = d  # liquidation starts at the next open: this open has passed
        out["cash"][s] = cash
        out["holdings_value"][s] = holdings
        out["nav"][s] = nav
        out["costs"][s] = day_costs
        out["dividends"][s] = day_div
        out["tax_paid"][s] = day_tax_paid
        out["tax_accrued"][s] = tax_accrued
        out["nav_after_tax"][s] = nav - tax_accrued

    liquidation_tax = 0.0
    if tax and first_fy_start is not None:
        last = session_dates[-1]
        fy = tax_rules.fiscal_year(last)
        fiscal_years[fy] = compute_fiscal_year_tax(
            fy_realisations, fy, carry_in, surcharge_rate=surcharge_rate
        )
        dividend_tax[fy] = (fy_dividends, div_tax(fy, fy_dividends))
        reg = regime(last)
        candidates: list[Realisation] = []
        for j in np.flatnonzero(qty > 0):
            if math.isnan(mark[j]):
                continue
            q, price = float(qty[j]), float(mark[j])
            costs = estimate_costs("SELL", q * price, reg)
            rs, _ = sell_fifo(
                lots[j], q, price, last, reg,
                sell_costs=costs, fmv_2018_01_31=fmv.get(asset_ids[j]),
            )
            candidates.extend(rs)
        if candidates:
            liquidation_tax = incremental_tax(
                _compressed(fy_sums), candidates, fy, carry_in, surcharge_rate=surcharge_rate
            )

    daily = pd.DataFrame(out, index=sessions, columns=DAILY_COLUMNS)
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    stale = pd.DataFrame(stale_rows, columns=STALE_COLUMNS)
    return SimulationResult(
        daily=daily,
        trades=trades,
        realisations=realisations,
        fiscal_years=fiscal_years,
        stale_marks=stale,
        liquidation_tax=liquidation_tax,
        ruined_on=ruined_on,
        dividend_tax=dividend_tax,
    )


def daily_returns(result: SimulationResult, after_tax: bool = True) -> pd.Series:
    """Simple daily returns of ``nav_after_tax`` (or ``nav``); the first session is NaN.

    Ready for ``performance.sharpe(r, periods_per_year=252)`` and friends, which ignore
    the NaN. Nothing is dropped. A return whose base (the previous session's value) is
    ``<= 0`` is NaN: a simple return off a non-positive base has no meaning. So on a ruined
    curve the return into the first non-positive value is ``<= -1`` (a total loss) and every
    later return off a non-positive value is NaN. A curve that stays positive is unaffected.
    """
    nav = result.daily["nav_after_tax" if after_tax else "nav"]
    base = nav.shift(1)
    return (nav / base - 1.0).where(base > 0)
