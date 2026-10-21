"""
Backtest driver (investbot/backtest/runner.py, ``main.py backtest``).

In-memory SQLite with six synthetic assets over ~3 years, one of them a past member whose
stint ends mid-sample. No network, no real DB.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.backtest import runner
from investbot.backtest.runner import (
    BENCHMARK, SECTIONS, STRATEGY, TAX_NAIVE, BacktestRun, render_report, run_backtest,
    valid_signals,
)
from investbot.data.models import (
    Asset, AssetClass, Base, CorporateAction, Exchange, IndexMembership, PriceDaily,
)

DATES = pd.bdate_range("2021-01-01", "2023-12-29")
STINT_END = date(2022, 6, 30)
KW = dict(signal="BASE_MOM_20", min_names=3, quantile=0.4)


def _bars(dates: pd.DatetimeIndex, seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(len(dates))))
    open_ = close * (1 + 0.002 * rng.standard_normal(len(dates)))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1e6}, index=dates)


def _fill(s, ticker, bars, active=True, stints=(), actions=()):
    a = Asset(exchange_id=s.info["exchange_id"], ticker=ticker, name=ticker,
              asset_type="stock", is_active=active)
    s.add(a)
    s.flush()
    s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high, low=r.low,
                          close=r.close, adj_close=r.close, volume=int(r.volume))
               for d, r in bars.iterrows()])
    for f, t in stints:
        s.add(IndexMembership(index_code="NIFTY50", asset_id=a.id, effective_from=f,
                              effective_to=t, source="test", verified=True))
    for ex, kind, value in actions:
        s.add(CorporateAction(asset_id=a.id, ex_date=ex, action_type=kind, value=value,
                              source="test"))
    s.commit()
    return a.id


def _new_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    s.info["exchange_id"] = ex.id
    return s


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch):
    """The synthetic stints are not the real change list; the pre-flight has its own test."""
    from investbot.data import membership_check
    monkeypatch.setattr(membership_check, "require_membership_sync",
                        lambda *a, **k: object())


@pytest.fixture(scope="module")
def db():
    s = _new_session()
    ids = {}
    drifts = [0.0006, 0.0003, 0.0, -0.0002, 0.0004, 0.0002]
    for i, name in enumerate(["A", "B", "C", "D", "E"]):
        ids[name] = _fill(s, f"{name}.NS", _bars(DATES, i, drifts[i]),
                          stints=[(DATES[0].date(), None)],
                          actions=[(date(2022, 3, 15), "dividend", 2.0)] if name == "A" else ())
    # past member: stint ends mid-sample, prices continue (so a held name is easy to leak)
    ids["P"] = _fill(s, "P.NS", _bars(DATES, 9, drifts[5]), active=False,
                     stints=[(DATES[0].date(), STINT_END)])
    yield s, ids
    s.close()


@pytest.fixture(scope="module")
def full(db):
    """Full-window runs, computed once (feature building dominates the runtime)."""
    from investbot.data import membership_check
    s, _ = db
    with pytest.MonkeyPatch.context() as mp:     # the autouse patch is function-scoped
        mp.setattr(membership_check, "require_membership_sync", lambda *a, **k: object())
        return {f: run_backtest(s, freq=f, start=None, end=None, **KW) for f in ("M", "D")}


# --- run_backtest ----------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["M", "D"])
def test_runs_for_monthly_and_daily(full, freq):
    run = full[freq]
    assert isinstance(run, BacktestRun)
    assert list(run.compare.index) == [STRATEGY, TAX_NAIVE, BENCHMARK]
    assert len(run.strategy.daily) == len(run.benchmark.daily) == len(run.strategy_tax_naive.daily)
    assert run.strategy.daily.index.equals(run.benchmark.daily.index)
    # tax=False: nothing paid or accrued, same trades as the taxed run's decisions
    assert (run.strategy_tax_naive.daily["tax_paid"] == 0).all()
    assert (run.strategy_tax_naive.daily["tax_accrued"] == 0).all()
    assert run.strategy_tax_naive.fiscal_years == {}
    assert run.strategy.daily["tax_paid"].sum() + run.strategy.daily["tax_accrued"].iloc[-1] >= 0
    assert list(run.go_no_go["metric"]) == ["excess_cagr_after_tax", "max_drawdown", "sharpe",
                                            "win_rate", "tax_alpha"]
    assert run.fingerprint["db_price_rows"] == len(DATES) * 6
    assert run.missing == [] and run.empty_in_window == []
    assert not run.start_clamped
    # pure benchmark: same CAGR rows exist and excess of the benchmark over itself is 0
    assert run.compare.loc[BENCHMARK, "excess_cagr_after_tax"] == pytest.approx(0.0)


def test_benchmark_and_strategy_hold_only_members(db, full):
    s, ids = db
    run = full["M"]
    p = ids["P"]
    for w in (run.benchmark_weights, run.weights):
        after = w.loc[w.index > pd.Timestamp(STINT_END), p]
        assert (after == 0).all()
        assert (w.sum(axis=1).round(9).isin([0.0, 1.0])).all()
    before = run.benchmark_weights.loc[run.benchmark_weights.index <= pd.Timestamp(STINT_END), p]
    assert (before > 0).any()                      # P is held while it is a member
    assert run.benchmark_weights.loc[run.benchmark_weights.index > pd.Timestamp(STINT_END)
                                     ].gt(0).sum(axis=1).max() == 5
    # the strategy picks ceil(0.4 x n) names (2 of 5, 3 of 6) at most
    n_held = (run.weights > 0).sum(axis=1)
    assert n_held.max() <= 3 and n_held.min() >= 2


@pytest.mark.parametrize("freq,end", [("M", date(2022, 9, 14)), ("D", date(2022, 9, 14))])
def test_no_lookahead_end_truncation(db, full, freq, end):
    s, ids = db
    full = full[freq]
    cut = run_backtest(s, freq=freq, start=None, end=end, **KW)
    stamp = pd.Timestamp(end)
    assert cut.strategy.daily.index[-1] <= stamp
    for label in ("strategy", "strategy_tax_naive", "benchmark"):
        a = getattr(cut, label).daily
        b = getattr(full, label).daily.loc[:a.index[-1]]
        pd.testing.assert_frame_equal(a, b)
    assert len(cut.strategy.daily) > 200


def test_unknown_signal_lists_valid_columns(db):
    s, _ = db
    with pytest.raises(ValueError) as e:
        run_backtest(s, signal="NOPE", freq="M", start=None, end=None)
    msg = str(e.value)
    for col in ("TECH_SCORE", "BASE_MOM_12_1", "SIG_" + runner.SIGNALS[0].name):
        assert col in msg
    assert "TECH_SCORE" in valid_signals() and "REGIME" not in valid_signals()


def test_unknown_universe_and_freq_raise(db):
    s, _ = db
    with pytest.raises(ValueError):
        run_backtest(s, freq="M", start=None, end=None, universe="bogus", **KW)
    with pytest.raises(ValueError):
        run_backtest(s, freq="Q", start=None, end=None, **KW)


def test_stale_membership_exits(db, monkeypatch):
    from investbot.data import membership_check
    s, _ = db
    monkeypatch.undo()                              # the real pre-flight: stints are not the change list
    assert membership_check.require_membership_sync is not None
    with pytest.raises(SystemExit) as e:
        run_backtest(s, freq="M", start=None, end=None, **KW)
    assert e.value.code == 1


def test_start_clamped_to_first_equity_regime():
    s = _new_session()
    dates = pd.bdate_range("2004-07-01", periods=160)
    for i, name in enumerate("ABCD"):
        _fill(s, f"{name}.NS", _bars(dates, i, 0.0005 * i))
    told: list[str] = []
    run = run_backtest(s, signal="BASE_MOM_20", freq="D", start=date(2004, 7, 1), end=None,
                       universe="active", min_names=3, quantile=0.5, notice=told.append)
    assert run.start_clamped and len(told) == 1 and "2004-10-01" in told[0]
    assert run.params["first_session"] >= date(2004, 10, 1)
    assert run.strategy.daily.index[0].date() >= date(2004, 10, 1)
    s.close()


def test_simulator_error_names_the_run(monkeypatch):
    s = _new_session()
    dates = pd.bdate_range("2021-01-01", periods=120)
    for i, name in enumerate("ABCD"):
        _fill(s, f"{name}.NS", _bars(dates, i, 0.0), stints=[(dates[0].date(), None)])

    def boom(*a, **k):
        raise ValueError("quantity must be > 0")
    monkeypatch.setattr(runner, "simulate", boom)
    with pytest.raises(ValueError, match="strategy run failed: quantity must be > 0"):
        run_backtest(s, signal="BASE_MOM_20", freq="M", start=None, end=None, min_names=3,
                     quantile=0.5)
    s.close()


def test_drops_member_without_bar_in_window(db):
    s, ids = db
    run = run_backtest(s, freq="M", start=date(2022, 9, 1), end=None, **KW)
    assert ids["P"] not in run.tickers              # stint ended before the window: not in universe
    s2 = _new_session()
    _fill(s2, "A.NS", _bars(DATES, 1, 0.0), stints=[(DATES[0].date(), None)])
    for n in "BCD":
        _fill(s2, f"{n}.NS", _bars(DATES, ord(n), 0.0), stints=[(DATES[0].date(), None)])
    _fill(s2, "Z.NS", _bars(DATES[:100], 5, 0.0), active=False, stints=[(DATES[0].date(), None)])
    run2 = run_backtest(s2, freq="M", start=date(2023, 6, 1), end=None, signal="BASE_MOM_20",
                        min_names=3, quantile=0.5)
    assert run2.empty_in_window == ["Z.NS"]
    assert "Z.NS" not in run2.tickers.values()
    s2.close()


# --- report ----------------------------------------------------------------------------------


def test_report_has_every_section_and_caveats(full):
    run = full["M"]
    text = render_report(run)
    for heading in SECTIONS:
        assert f"## {heading}" in text
    assert "BASE_MOM_20" in text and "latest price date" in text
    for phrase in ("Universe members without prices: 0", "Unadjusted-split issues",
                   "Stale-mark episodes", "unverified", "Dividend income tax is not modelled",
                   "held at the last close", "Rebalance dates before 1998-10-07"):
        assert phrase in text
    assert text.endswith("\n") and "nan" not in text.lower().replace("finance", "")
    assert render_report(run) == text               # pure


def test_report_counts_missing_pre_complete_and_splits(full):
    run = replace(full["M"], missing=["GONE.NS", "OLD.NS"], split_issues=[object()] * 3,
                  complete_from=date(2022, 1, 1))
    text = render_report(run)
    assert "Universe members without prices: 2** of 8" in text
    assert "(GONE.NS, OLD.NS)" in text
    assert "(`detect_unadjusted_splits`): 3**" in text
    n_pre = int((run.weights.index < pd.Timestamp("2022-01-01")).sum())
    assert f"Rebalance dates before 2022-01-01 (membership change list complete from then): " \
           f"{n_pre} of {len(run.weights)}" in text.replace("**", "")
    assert n_pre > 0


# --- CLI -------------------------------------------------------------------------------------


def test_cli_help_exits_zero():
    r = subprocess.run([sys.executable, "main.py", "backtest", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--signal" in r.stdout and "--freq" in r.stdout


def test_cli_requires_signal_and_freq():
    r = subprocess.run([sys.executable, "main.py", "backtest"], capture_output=True, text=True)
    assert r.returncode == 2 and "--signal" in r.stderr
