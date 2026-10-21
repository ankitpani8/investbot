"""
DB-backed inputs for the backtest (investbot/backtest/inputs.py).

Pure helpers are tested on synthetic Series; the loader on an in-memory SQLite DB seeded here.
No network.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.backtest.inputs import (
    BacktestInputs,
    asset_columns,
    load_backtest_inputs,
    union_sessions,
    wide_frame,
)
from investbot.backtest.simulator import simulate
from investbot.data.models import (
    Asset,
    AssetClass,
    Base,
    CorporateAction,
    Exchange,
    IndexMembership,
    PriceDaily,
)

D = pd.bdate_range("2023-01-02", periods=40)


def _series(dates, values) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def _bars(dates, base: float, high_bump: float = 2.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    close = base + np.arange(len(idx), dtype=float)
    return pd.DataFrame({"open": close - 0.5, "high": close + high_bump, "low": close - 1.0,
                         "close": close, "volume": 1000.0}, index=idx)


# --- pure helpers ---------------------------------------------------------------------------


def test_union_sessions_sorted_and_deduplicated():
    a = pd.DatetimeIndex(["2023-01-03", "2023-01-05"])
    b = pd.DatetimeIndex(["2023-01-02", "2023-01-03"])
    out = union_sessions([a, b])
    assert list(out) == [pd.Timestamp(x) for x in ("2023-01-02", "2023-01-03", "2023-01-05")]
    assert len(union_sessions([])) == 0


def test_wide_frame_union_index_and_nan_gaps():
    a = _series(["2023-01-02", "2023-01-03", "2023-01-04"], [1, 2, 3])
    b = _series(["2023-01-03", "2023-01-05"], [10, 20])
    w = wide_frame({7: a, 3: b})
    assert list(w.columns) == [3, 7]                          # ascending asset_id
    assert w.columns.dtype == np.int64
    assert list(w.index) == [pd.Timestamp(x) for x in
                             ("2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05")]
    assert w[7].isna().tolist() == [False, False, False, True]
    assert w[3].isna().tolist() == [True, False, True, False]
    assert w.loc["2023-01-03"].tolist() == [10, 2]


def test_wide_frame_empty():
    w = wide_frame({})
    assert w.shape == (0, 0) and isinstance(w.index, pd.DatetimeIndex)


def test_asset_columns_start_cut_keeps_fmv_and_pre_start_dividend():
    bars = _bars(pd.bdate_range("2018-01-25", periods=20), 100.0)
    actions = pd.DataFrame({"action_type": ["dividend"], "value": [5.0]},
                           index=pd.DatetimeIndex(["2018-01-30"], name="ex_date"))
    cols = asset_columns(bars, actions, start=date(2018, 2, 5))
    assert cols.close.index[0] == pd.Timestamp("2018-02-05")
    assert cols.dividends.eq(0.0).all()                       # not pushed onto the first kept bar
    assert cols.fmv == bars.loc["2018-01-31", "high"]


# --- DB fixture -----------------------------------------------------------------------------


def _session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    s.info["exchange_id"] = ex.id
    return s


def _add_asset(s, ticker: str, bars: pd.DataFrame | None, active: bool = True,
               stints=(), actions=()) -> int:
    a = Asset(exchange_id=s.info["exchange_id"], ticker=ticker, name=ticker, asset_type="stock",
              is_active=active)
    s.add(a)
    s.flush()
    if bars is not None:
        s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high, low=r.low,
                              close=r.close, adj_close=r.close, volume=int(r.volume))
                   for d, r in bars.iterrows()])
    for f, t in stints:
        s.add(IndexMembership(index_code="NIFTY50", asset_id=a.id, effective_from=f,
                              effective_to=t, source="test", verified=True))
    for ex_date, kind, value in actions:
        s.add(CorporateAction(asset_id=a.id, ex_date=ex_date, action_type=kind, value=value,
                              source="test"))
    s.commit()
    return a.id


@pytest.fixture
def db():
    s = _session()
    # A: all sessions, dividend on D[10]; B: starts later and skips a session;
    # C: past member (stint D[5]..D[14]), delisted bars stop at D[20]; M: member, no prices.
    ids = {}
    ids["A"] = _add_asset(s, "A.NS", _bars(D, 100.0), stints=[(D[0].date(), None)],
                          actions=[(D[10].date(), "dividend", 3.0)])
    ids["B"] = _add_asset(s, "B.NS", _bars(D[[3, 4, 6, 7, 8, 9, 10] + list(range(11, 40))], 50.0),
                          stints=[(D[3].date(), None)])
    ids["C"] = _add_asset(s, "C.NS", _bars(D[:21], 20.0), active=False,
                          stints=[(D[5].date(), D[14].date())])
    ids["M"] = _add_asset(s, "M.NS", None, stints=[(D[0].date(), None)])
    ids["X"] = _add_asset(s, "X.NS", _bars(D, 10.0))          # active, never a member
    yield s, ids
    s.close()


# --- loader ---------------------------------------------------------------------------------


def test_union_index_and_nan_gaps(db):
    s, ids = db
    inp = load_backtest_inputs(s)
    assert list(inp.open_.columns) == sorted([ids["A"], ids["B"], ids["C"]])
    assert inp.open_.index.equals(D)                          # union of A (all) and the rest
    assert inp.open_.index.equals(inp.close.index) and inp.close.columns.equals(inp.open_.columns)
    assert inp.dividends.index.equals(D) and inp.membership.index.equals(D)
    assert inp.open_[ids["B"]].iloc[:3].isna().all()          # B has no bars before D[3]
    assert np.isnan(inp.open_.loc[D[5], ids["B"]])            # skipped session
    assert inp.open_[ids["C"]].iloc[21:].isna().all()         # C's bars stop
    assert inp.open_.loc[D[0], ids["A"]] == pytest.approx(99.5)
    assert inp.close.loc[D[4], ids["B"]] == pytest.approx(51.0)
    assert inp.tickers == {ids["A"]: "A.NS", ids["B"]: "B.NS", ids["C"]: "C.NS"}
    assert isinstance(inp.open_.index, pd.DatetimeIndex)


def test_missing_ticker_listed_and_warned(db, caplog):
    s, ids = db
    with caplog.at_level(logging.WARNING, logger="investbot.backtest.inputs"):
        inp = load_backtest_inputs(s)
    assert inp.missing == ["M.NS"]
    assert ids["M"] not in inp.open_.columns
    assert "M.NS" in caplog.text


def test_past_member_membership_only_inside_stint(db):
    s, ids = db
    m = load_backtest_inputs(s).membership
    assert m.dtypes.eq(bool).all()
    c = m[ids["C"]]
    assert c.loc[D[5]:D[14]].all()                            # effective_from session counts
    assert not c.loc[:D[4]].any() and not c.loc[D[15]:].any()
    b = m[ids["B"]]
    assert not b.loc[:D[2]].any() and b.loc[D[3]:].all()
    assert m[ids["A"]].all()


def test_universe_window_drops_non_overlapping_stints(db):
    s, ids = db
    inp = load_backtest_inputs(s, start=D[20].date())
    assert ids["C"] not in inp.open_.columns                  # stint ended D[14]
    assert inp.open_.index[0] == D[20]


def test_active_universe_all_members(db):
    s, ids = db
    inp = load_backtest_inputs(s, universe="active")
    # active stocks with prices: A, B, X (M has none; C is inactive)
    assert list(inp.open_.columns) == sorted([ids["A"], ids["B"], ids["X"]])
    assert inp.membership.to_numpy().all() and inp.membership.shape == inp.open_.shape
    assert inp.missing == ["M.NS"]
    with pytest.raises(ValueError):
        load_backtest_inputs(s, universe="bogus")


def test_dividends_on_ex_date_bar_zero_elsewhere_nan_off_bar(db):
    s, ids = db
    dv = load_backtest_inputs(s).dividends
    assert dv.loc[D[10], ids["A"]] == 3.0
    assert (dv[ids["A"]].drop(D[10]) == 0.0).all()
    assert (dv[ids["B"]].dropna() == 0.0).all()               # bars without dividend are 0.0
    assert dv.loc[D[5], ids["B"]] != dv.loc[D[5], ids["B"]]   # NaN: B has no bar there
    assert dv.loc[D[30], ids["C"]] != dv.loc[D[30], ids["C"]]


def test_dividend_before_start_not_credited_to_first_bar(db):
    s, ids = db
    dv = load_backtest_inputs(s, start=D[12].date()).dividends
    assert (dv[ids["A"]] == 0.0).all()


def test_fmv_from_bars_before_start():
    s = _session()
    dates = pd.bdate_range("2018-01-22", periods=30)
    a = _add_asset(s, "F.NS", _bars(dates, 100.0), stints=[(date(2018, 1, 22), None)])
    n = _add_asset(s, "N.NS", _bars(pd.bdate_range("2018-02-05", periods=5), 10.0),
                   stints=[(date(2018, 2, 5), None)])
    inp = load_backtest_inputs(s, start=date(2018, 3, 1))
    assert inp.open_.index[0] >= pd.Timestamp("2018-03-01")
    expect = _bars(dates, 100.0).loc["2018-01-31", "high"]
    assert inp.fmv_2018_01_31 == {a: pytest.approx(expect)}   # N has no bar <= 2018-01-31: left out
    assert n not in inp.fmv_2018_01_31
    assert all(not np.isnan(v) for v in inp.fmv_2018_01_31.values())


def test_unadjusted_split_reported_prices_unchanged():
    s = _session()
    dates = pd.bdate_range("2023-01-02", periods=30)
    bars = _bars(dates, 100.0)
    bars.iloc[15:, bars.columns.get_indexer(["open", "high", "low", "close"])] /= 2.0
    bars.loc[dates[14], "close"] = 100.0                      # prev close / ex open == 2.0
    bars.loc[dates[15], "open"] = 50.0
    a = _add_asset(s, "S.NS", bars, stints=[(dates[0].date(), None)],
                   actions=[(dates[15].date(), "split", 2.0)])
    inp = load_backtest_inputs(s)
    assert [(i.ticker, i.ex_date, i.factor) for i in inp.split_issues] == [
        ("S.NS", dates[15].date(), 2.0)]
    np.testing.assert_array_equal(inp.close[a].to_numpy(), bars["close"].to_numpy())
    np.testing.assert_array_equal(inp.open_[a].to_numpy(), bars["open"].to_numpy())


def test_end_truncation_equals_prefix_of_full_load(db):
    s, _ = db
    full = load_backtest_inputs(s)
    for cut in (D[12], D[21], D[30]):
        part = load_backtest_inputs(s, end=cut.date())
        assert part.open_.index[-1] <= cut
        for name in ("open_", "close", "dividends", "membership"):
            f, p = getattr(full, name), getattr(part, name)
            expect = f.loc[:cut, p.columns]
            pd.testing.assert_frame_equal(p, expect, check_freq=False)
    # a column that only appears in the full load has no rows <= cut
    early = load_backtest_inputs(s, end=D[2].date())
    assert set(early.open_.columns) <= set(full.open_.columns)


def test_fingerprint_and_complete_from(db):
    s, _ = db
    inp = load_backtest_inputs(s)
    assert inp.fingerprint["db_price_rows"] == 40 + 36 + 21 + 40
    assert inp.fingerprint["db_action_rows"] == 1
    assert set(inp.fingerprint) == {"db_max_date", "db_price_rows", "db_action_rows"}
    assert isinstance(inp.complete_from, date)


def test_no_dropna_in_module():
    import inspect

    from investbot.backtest import inputs
    assert "dropna" not in inspect.getsource(inputs).replace("No ``dropna``", "")


def test_smoke_feeds_simulate(db):
    s, ids = db
    inp: BacktestInputs = load_backtest_inputs(s)
    cols = list(inp.open_.columns)
    w = pd.DataFrame(0.0, index=[D[1], D[25]], columns=cols)
    w.loc[D[1], ids["A"]] = 0.5
    w.loc[D[1], ids["B"]] = 0.4
    w.loc[D[25], ids["A"]] = 0.9
    res = simulate(w, inp.open_, inp.close, inp.dividends, fmv_2018_01_31=inp.fmv_2018_01_31)
    assert len(res.daily) == len(D)
