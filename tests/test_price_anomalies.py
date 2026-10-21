"""
Price-path anomaly audit (`investbot.data.price_anomalies`, section of `main.py dq`).
Synthetic frames for the pure detector, in-memory SQLite for the loader and the CLI.
"""
from __future__ import annotations

import ast
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.data import price_anomalies as pa
from investbot.data.models import Asset, Base, CorporateAction, PriceDaily
from investbot.data.price_anomalies import (
    ANOMALY_KINDS, PriceAnomaly, audit_price_anomalies, detect_price_anomalies,
    largest_anomalies, summarise_anomalies,
)

DAYS = pd.bdate_range("2001-03-01", periods=60)


def _bars(closes, opens=None) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    o = c if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": o, "high": c, "low": c, "close": c, "volume": 1000.0},
                        index=DAYS[:len(c)])


def _walk(n=30, level=100.0) -> list[float]:
    """A quiet path: +-0.5% alternating, no big move, no repeated close."""
    out, p = [], level
    for i in range(n):
        p *= 1.005 if i % 2 == 0 else 0.996
        out.append(p)
    return out


def _actions(*rows) -> pd.DataFrame:
    """rows: (day index, type, value)."""
    return pd.DataFrame(
        {"action_type": [r[1] for r in rows], "value": [float(r[2]) for r in rows]},
        index=pd.DatetimeIndex([DAYS[r[0]] for r in rows], name="ex_date"),
        columns=["action_type", "value"],
    )


NO_ACTIONS = _actions()


def _kinds(found):
    return [(a.kind, a.start) for a in found]


def test_kinds_are_stable_strings():
    assert ANOMALY_KINDS == ("spike_reversal", "extreme_move", "stale_run")


def test_single_bad_tick_that_reverts_is_one_spike():
    c = _walk()
    base = c[10]
    c[10] = base * np.exp(0.30)                      # bad tick, back to normal next bar
    found = detect_price_anomalies(_bars(c), NO_ACTIONS, "X.NS")
    assert [(a.kind, a.start, a.end, a.sessions) for a in found] == [
        ("spike_reversal", DAYS[10].date(), DAYS[10].date(), 1)]
    assert found[0].ticker == "X.NS" and found[0].value == pytest.approx(0.30, abs=0.01)


def test_reverting_bar_of_a_big_spike_is_not_a_second_extreme_move():
    c = _walk()
    c[10] = c[9] * np.exp(0.60)                      # |ret| >= 0.40 both ways
    found = detect_price_anomalies(_bars(c), NO_ACTIONS)
    assert _kinds(found) == [("spike_reversal", DAYS[10].date())]


def test_genuine_25pct_move_that_stays_is_not_a_spike():
    c = _walk()
    c[10:] = [x * 1.25 for x in c[10:]]
    assert detect_price_anomalies(_bars(c), NO_ACTIONS) == []


def test_partial_reversion_below_share_is_not_a_spike():
    c = _walk()
    c[10] = c[9] * np.exp(0.30)
    c[11:] = [x * np.exp(0.30 * 0.5) for x in c[11:]]   # gives back only half
    assert detect_price_anomalies(_bars(c), NO_ACTIONS) == []


def test_plus_60pct_jump_without_action_is_extreme_move():
    c = _walk()
    c[10:] = [x * np.exp(0.60) for x in c[10:]]
    found = detect_price_anomalies(_bars(c), NO_ACTIONS)
    assert [(a.kind, a.start, a.sessions) for a in found] == [("extreme_move", DAYS[10].date(), 1)]
    assert found[0].value == pytest.approx(0.60, abs=0.02)


@pytest.mark.parametrize("ex_offset", [-1, 0, 1])
def test_extreme_move_within_one_session_of_an_action_is_explained(ex_offset):
    c = _walk()
    c[10:] = [x * np.exp(0.60) for x in c[10:]]
    for kind in ("split", "bonus", "dividend"):
        assert detect_price_anomalies(_bars(c), _actions((10 + ex_offset, kind, 2.0))) == []


def test_extreme_move_two_sessions_from_an_action_is_still_flagged():
    c = _walk()
    c[10:] = [x * np.exp(0.60) for x in c[10:]]
    found = detect_price_anomalies(_bars(c), _actions((12, "split", 2.0)))
    assert _kinds(found) == [("extreme_move", DAYS[10].date())]


def test_split_ex_date_is_not_an_extreme_move_adjusted_or_unadjusted():
    # Unadjusted 1:5 split: close drops 80% on the ex-date (|log ret| 1.6), open = new level.
    c = _walk()
    c[10:] = [x / 5 for x in c[10:]]
    o = list(c)
    o[10] = c[9] / 5
    actions = _actions((10, "split", 5.0))
    assert detect_price_anomalies(_bars(c, o), actions) == []
    # Adjusted history: the same ex-date with a smooth path flags nothing either.
    assert detect_price_anomalies(_bars(_walk()), actions) == []


def test_six_bar_frozen_close_is_one_stale_run():
    c = _walk()
    for i in range(10, 16):
        c[i] = 123.45
    found = detect_price_anomalies(_bars(c), NO_ACTIONS)
    assert [(a.kind, a.start, a.end, a.sessions) for a in found] == [
        ("stale_run", DAYS[10].date(), DAYS[15].date(), 6)]
    assert found[0].value == 123.45


def test_four_bar_frozen_close_is_not_flagged():
    c = _walk()
    for i in range(10, 14):
        c[i] = 123.45
    assert detect_price_anomalies(_bars(c), NO_ACTIONS) == []


def test_stale_run_tolerates_float_noise_and_is_cut_by_a_nan():
    c = _walk()
    for i in range(10, 20):
        c[i] = 50.0 * (1 + 1e-12 * (i % 2))
    c[14] = np.nan
    found = [a for a in detect_price_anomalies(_bars(c), NO_ACTIONS) if a.kind == "stale_run"]
    assert [(a.start, a.end, a.sessions) for a in found] == [(DAYS[15].date(), DAYS[19].date(), 5)]


def test_nan_and_zero_closes_give_no_flag_and_no_exception():
    c = _walk()
    c[10] = np.nan
    c[20] = 0.0
    c[21] = -3.0
    bars = _bars(c)
    found = detect_price_anomalies(bars, NO_ACTIONS)
    assert found == []
    assert len(bars) == 30 and bars["close"].isna().sum() == 1     # nothing dropped
    zeros = detect_price_anomalies(_bars([0.0] * 10), NO_ACTIONS)  # a flat zero is not a frozen price
    assert zeros == []


def test_empty_and_tiny_frames():
    assert detect_price_anomalies(_bars([]), NO_ACTIONS) == []
    assert detect_price_anomalies(_bars([100.0]), NO_ACTIONS) == []
    assert detect_price_anomalies(_bars([100.0, 150.0]), NO_ACTIONS)[0].kind == "extreme_move"
    # a spike on the last bar has no t+1 to revert: not a spike, and < 0.40 so not extreme
    c = _walk()
    c[-1] = c[-2] * np.exp(0.30)
    assert detect_price_anomalies(_bars(c), NO_ACTIONS) == []


def test_inputs_are_not_modified():
    c = _walk()
    c[10] = c[9] * np.exp(0.5)
    for i in range(20, 26):
        c[i] = 77.0
    bars, actions = _bars(c), _actions((5, "dividend", 1.0))
    bars_before, actions_before = bars.copy(), actions.copy()
    detect_price_anomalies(bars, actions)
    pd.testing.assert_frame_equal(bars, bars_before)
    pd.testing.assert_frame_equal(actions, actions_before)


def test_truncation_invariance_except_for_the_next_bar_of_a_spike():
    """Flags at bar t reappear on a truncation at t+1 or later; spikes need t+1 and are the only kind that look ahead."""
    c = _walk(50)
    c[10] = c[9] * np.exp(0.5)
    c[30:] = [x * np.exp(0.7) for x in c[30:]]       # a level break that stays
    for i in range(40, 46):
        c[i] = 11.0
    bars = _bars(c)
    full = detect_price_anomalies(bars, NO_ACTIONS)
    for cut in (12, 31, 44, 50):
        part = detect_price_anomalies(bars.iloc[:cut], NO_ACTIONS)
        cut_day = DAYS[cut - 1].date()
        expected = [a for a in full if a.end <= cut_day and a.kind != "stale_run"]
        got = [a for a in part if a.kind != "stale_run"]
        # spikes at the last bar cannot be judged; extremes never depend on later bars
        assert [a for a in got if a.kind == "extreme_move"] == [a for a in expected if a.kind == "extreme_move"]
        assert [a for a in got if a.kind == "spike_reversal"] == [
            a for a in expected if a.kind == "spike_reversal" and a.start < cut_day]


def test_summary_counts_by_year_and_distinct_tickers():
    d = lambda y: date(y, 6, 1)  # noqa: E731
    anomalies = [
        PriceAnomaly("A", "spike_reversal", d(2001), d(2001), 1, 0.3),
        PriceAnomaly("A", "spike_reversal", d(2001), d(2001), 1, -0.3),
        PriceAnomaly("B", "spike_reversal", d(2003), d(2003), 1, 0.3),
        PriceAnomaly("B", "stale_run", d(2003), d(2003), 6, 10.0),
    ]
    s = summarise_anomalies(anomalies)
    assert list(s.columns) == list(ANOMALY_KINDS)
    assert list(s.index) == [2001, 2003, "total", "tickers"]
    assert s.loc[2001, "spike_reversal"] == 2 and s.loc[2003, "stale_run"] == 1
    assert s.loc["total", "spike_reversal"] == 3 and s.loc["tickers", "spike_reversal"] == 2
    assert s.loc["tickers", "extreme_move"] == 0
    empty = summarise_anomalies([])
    assert list(empty.index) == ["total", "tickers"] and int(empty.to_numpy().sum()) == 0


def test_largest_ranks_returns_by_magnitude_then_stale_runs():
    d = date(2001, 1, 1)
    items = [
        PriceAnomaly("A", "stale_run", d, d, 9, 2500.0),
        PriceAnomaly("A", "spike_reversal", d, d, 1, 0.25),
        PriceAnomaly("A", "extreme_move", d, d, 1, -0.9),
    ]
    assert [a.value for a in largest_anomalies(items, 2)] == [-0.9, 0.25]
    assert largest_anomalies(items)[-1].kind == "stale_run"


def test_module_has_no_dropna_and_states_the_lookahead_note():
    src = Path(pa.__file__).read_text()
    tree = ast.parse(src)
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "dropna" not in calls
    doc = pa.__doc__
    assert "t+1" in doc and "offline" in doc.lower() and "backtest mask" in doc


# ──────────────────────────────────────────────────────────
# Loader and CLI
# ──────────────────────────────────────────────────────────

@pytest.fixture
def Session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def _add_asset(s, ticker, closes, *, active=True, start=DAYS[0], nan_at=None):
    a = Asset(exchange_id=1, ticker=ticker, name=ticker, asset_type="stock", is_active=active)
    s.add(a)
    s.flush()
    for i, c in enumerate(closes):
        s.add(PriceDaily(asset_id=a.id, date=DAYS[i].date(), open=c, high=c, low=c, close=c,
                         adj_close=c, volume=1000))
    s.commit()
    return a


def _spiky():
    c = _walk()
    c[10] = c[9] * np.exp(0.5)
    return c


def test_audit_reads_active_and_inactive_assets_and_filters_by_ticker(Session):
    with Session() as s:
        _add_asset(s, "A.NS", _spiky())
        _add_asset(s, "OLD.NS", _spiky(), active=False)
        _add_asset(s, "CLEAN.NS", _walk())
        s.add(Asset(exchange_id=1, ticker="EMPTY.NS", name="e", asset_type="stock"))
        s.commit()
        found = audit_price_anomalies(s)
        assert [(a.ticker, a.kind) for a in found] == [("A.NS", "spike_reversal"), ("OLD.NS", "spike_reversal")]
        assert [a.ticker for a in audit_price_anomalies(s, tickers=["OLD.NS"])] == ["OLD.NS"]
        assert audit_price_anomalies(s, tickers=[]) == []


def test_audit_uses_stored_corporate_actions(Session):
    c = _walk()
    c[10:] = [x * np.exp(0.6) for x in c[10:]]
    with Session() as s:
        a = _add_asset(s, "A.NS", c)
        assert [x.kind for x in audit_price_anomalies(s)] == ["extreme_move"]
        s.add(CorporateAction(asset_id=a.id, ex_date=DAYS[10].date(), action_type="split", value=2.0))
        s.commit()
        assert audit_price_anomalies(s) == []


@pytest.fixture
def main_mod(Session, monkeypatch):
    import main
    monkeypatch.setattr(main, "SessionLocal", Session)
    return main


def test_dq_exit_code_and_existing_lines_unchanged_when_only_anomalies_exist(
        main_mod, Session, monkeypatch, capsys, tmp_path):
    with Session() as s:
        _add_asset(s, "A.NS", _spiky())
    csv = tmp_path / "anomalies.csv"
    monkeypatch.setattr(sys, "argv", ["main.py", "dq", "--anomalies-csv", str(csv)])
    main_mod.cmd_dq()                                   # no SystemExit: exit code 0
    out = capsys.readouterr().out
    assert "Data quality OK: no bad price rows." in out
    assert "Split check OK: no unadjusted splits." in out
    assert "Price anomalies (report-only)" in out and "spike_reversal 1 (1 tickers)" in out
    assert out.index("Split check OK") < out.index("Price anomalies (report-only)")
    rows = pd.read_csv(csv)
    assert list(rows.columns) == ["ticker", "kind", "start", "end", "sessions", "value",
                                  "market_share", "market_wide"]
    assert rows["kind"].tolist() == ["spike_reversal"]
    with Session() as s:
        assert s.query(PriceDaily).count() == 30         # nothing purged


def test_dq_purge_does_not_touch_anomalies_and_clean_data_prints_none(main_mod, Session, monkeypatch, capsys):
    with Session() as s:
        _add_asset(s, "A.NS", _spiky())
    monkeypatch.setattr(sys, "argv", ["main.py", "dq", "--purge"])
    main_mod.cmd_dq()
    capsys.readouterr()
    with Session() as s:
        assert s.query(PriceDaily).count() == 30
        s.query(PriceDaily).delete()
        s.commit()
    monkeypatch.setattr(sys, "argv", ["main.py", "dq"])
    main_mod.cmd_dq()
    assert "none found" in capsys.readouterr().out


# ──────────────────────────────────────────────────────────
# Market-wide tag
# ──────────────────────────────────────────────────────────

def _flat(n=30, level=100.0) -> list[float]:
    """Quiet path with no hard mover (+-0.5% alternating)."""
    return _walk(n, level)


def _crash(closes, at, log_ret):
    c = list(closes)
    for i in range(at, len(c)):
        c[i] = c[i] * float(np.exp(log_ret))
    return c


def _counts(frames) -> pd.DataFrame:
    total = frames[0]
    for f in frames[1:]:
        total = total.add(f, fill_value=0).astype(int)
    return total


def test_daily_move_counts_columns_and_validity():
    c = _flat(10)
    c[3] = c[2] * np.exp(0.12)
    c[6] = c[5] * np.exp(-0.10)
    c[8] = np.nan
    c[1] = 0.0
    bars = _bars(c)
    out = pa.daily_move_counts(bars)
    assert list(out.columns) == ["priced", "up", "down"]
    assert out.index.equals(pd.DatetimeIndex(bars.index).normalize())
    assert len(out) == len(bars)
    assert out["priced"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    # the one-bar jumps revert on the next bar, which is a hard move the other way
    assert out["up"].tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert out["down"].tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert all(dt.kind == "i" for dt in out.dtypes)


def test_daily_move_counts_empty_and_single_bar():
    assert len(pa.daily_move_counts(_bars([]))) == 0
    assert pa.daily_move_counts(_bars([100.0]))["priced"].tolist() == [0]


def _universe(n_tickers, at, own_drop, others_drop):
    """n tickers; ticker 0 falls by own_drop at bar `at`, the others by others_drop (None = flat)."""
    frames, found = [], []
    own = {}
    for k in range(n_tickers):
        drop = own_drop if k == 0 else others_drop
        c = _flat(30) if drop is None else _crash(_flat(30), at, drop)
        bars = _bars(c)
        f = pa.daily_move_counts(bars)
        frames.append(f)
        if k == 0:
            found = detect_price_anomalies(bars, _actions(), ticker="T0")
            own = {("T0", a.start): tuple(int(v) for v in f.loc[pd.Timestamp(a.start)]) for a in found}
    return found, _counts(frames), own


def test_market_wide_crash_is_tagged_with_full_share():
    found, counts, own = _universe(10, 12, np.log(0.55), np.log(0.85))
    assert [a.kind for a in found] == ["extreme_move"]
    tagged = pa.tag_market_wide(found, counts, own)
    assert tagged[0].market_wide is True and tagged[0].market_share == pytest.approx(9 / 9)
    assert found[0].market_wide is False and np.isnan(found[0].market_share)    # input untouched
    assert tagged[0] is not found[0]


def test_lone_move_is_not_market_wide():
    found, counts, own = _universe(10, 12, np.log(0.55), None)
    tagged = pa.tag_market_wide(found, counts, own)
    assert tagged[0].market_wide is False and tagged[0].market_share == 0.0


def test_self_is_excluded_from_the_share():
    found, counts, own = _universe(5, 12, np.log(0.55), None)
    assert int(counts.loc[pd.Timestamp(found[0].start), "down"]) == 1
    for om in (own, None):                       # derived self row gives the same answer
        tagged = pa.tag_market_wide(found, counts, om)
        assert tagged[0].market_share == 0.0 and tagged[0].market_wide is False


def test_opposite_direction_movers_do_not_count():
    found, counts, own = _universe(10, 12, np.log(0.55), np.log(1.15))
    tagged = pa.tag_market_wide(found, counts, own)
    assert tagged[0].market_share == 0.0 and tagged[0].market_wide is False


def test_too_few_other_tickers_is_never_market_wide():
    found, counts, own = _universe(5, 12, np.log(0.55), np.log(0.85))      # 4 others, share 1.0
    tagged = pa.tag_market_wide(found, counts, own)
    assert tagged[0].market_share == pytest.approx(1.0) and tagged[0].market_wide is False
    found, counts, own = _universe(6, 12, np.log(0.55), np.log(0.85))      # 5 others
    assert pa.tag_market_wide(found, counts, own)[0].market_wide is True


def test_share_below_threshold_is_not_market_wide():
    frames, own = [], {}
    found = []
    for k in range(11):                          # 10 others, 1 of them moves = 10% < 20%
        c = _flat(30)
        if k in (0, 1):
            c = _crash(c, 12, np.log(0.55) if k == 0 else np.log(0.85))
        bars = _bars(c)
        f = pa.daily_move_counts(bars)
        frames.append(f)
        if k == 0:
            found = detect_price_anomalies(bars, _actions(), ticker="T0")
            own = {("T0", found[0].start): tuple(int(v) for v in f.loc[pd.Timestamp(found[0].start)])}
    tagged = pa.tag_market_wide(found, _counts(frames), own)
    assert tagged[0].market_share == pytest.approx(0.1) and tagged[0].market_wide is False


def test_stale_run_keeps_nan_and_false():
    c = _flat(20)
    c[5:12] = [c[5]] * 7
    bars = _bars(c)
    found = detect_price_anomalies(bars, _actions(), ticker="T0")
    assert [a.kind for a in found] == ["stale_run"]
    counts = _counts([pa.daily_move_counts(bars)] * 8)
    tagged = pa.tag_market_wide(found, counts)
    assert np.isnan(tagged[0].market_share) and tagged[0].market_wide is False


def test_tagging_with_no_other_priced_ticker_or_missing_date_does_not_raise():
    a = PriceAnomaly("T0", "extreme_move", DAYS[3].date(), DAYS[3].date(), 1, -0.6)
    only_self = pa.daily_move_counts(_bars(_crash(_flat(10), 3, -0.6)))
    t = pa.tag_market_wide([a], only_self)[0]
    assert np.isnan(t.market_share) and t.market_wide is False
    empty = pd.DataFrame({"priced": [], "up": [], "down": []}, dtype=int)
    t = pa.tag_market_wide([a], empty)[0]
    assert np.isnan(t.market_share) and t.market_wide is False
    assert pa.tag_market_wide([], empty) == []


def test_spike_reversal_is_tagged_too():
    found, counts, own = _universe(8, 12, np.log(0.55), np.log(0.85))
    spike = _flat(30)
    spike[12] = spike[11] * np.exp(-0.5)
    f = pa.daily_move_counts(_bars(spike))
    a = detect_price_anomalies(_bars(spike), _actions(), ticker="S")
    assert [x.kind for x in a] == ["spike_reversal"]
    # the spike ticker is the 9th in the cross-section; the 8 others all fell 15% that day
    tagged = pa.tag_market_wide(a, _counts([counts, f]), {("S", a[0].start): (1, 0, 1)})
    assert tagged[0].market_share == pytest.approx(1.0) and tagged[0].market_wide is True


def test_market_tag_is_same_day_cross_section_only():
    """Truncation: counts built from bars up to day t (plus self) tag a day-t extreme_move the same."""
    found, counts, own = _universe(8, 12, np.log(0.55), np.log(0.85))
    full = pa.tag_market_wide(found, counts, own)[0]
    cut = counts.loc[: pd.Timestamp(found[0].start)]
    assert pa.tag_market_wide(found, cut, own)[0].market_share == full.market_share


def _crash_day_db(s, n_tickers=4, lone_extra=False):
    for k in range(n_tickers):
        c = _crash(_walk(), 10, np.log(0.85))
        if k == 0:
            c = _crash(_walk(), 10, np.log(0.55))
        _add_asset(s, f"T{k}.NS", c)
    if lone_extra:
        c = _walk()
        c[20:] = [x * np.exp(0.6) for x in c[20:]]
        _add_asset(s, "LONE.NS", c)


def test_audit_tags_across_assets(Session):
    with Session() as s:
        _crash_day_db(s, n_tickers=7, lone_extra=True)       # 8 assets
        found = audit_price_anomalies(s)
        by = {(a.ticker, a.start): a for a in found}
        crash = by[("T0.NS", DAYS[10].date())]
        assert crash.kind == "extreme_move"
        assert crash.market_share == pytest.approx(6 / 7) and crash.market_wide is True
        lone = by[("LONE.NS", DAYS[20].date())]
        assert lone.market_share == 0.0 and lone.market_wide is False
        # tickers= narrows the cross-section to the loaded assets
        few = audit_price_anomalies(s, tickers=["T0.NS", "T1.NS"])
        assert few[0].market_share == pytest.approx(1.0) and few[0].market_wide is False
        assert audit_price_anomalies(s, tickers=[]) == []


def test_dq_prints_market_summary_marker_and_csv_columns(main_mod, Session, monkeypatch, capsys, tmp_path):
    with Session() as s:
        _crash_day_db(s, n_tickers=7, lone_extra=True)
    csv = tmp_path / "a.csv"
    monkeypatch.setattr(sys, "argv", ["main.py", "dq", "--anomalies-csv", str(csv)])
    main_mod.cmd_dq()
    out = capsys.readouterr().out
    assert "1 of 2 return anomalies are market-wide (>=20% of other priced tickers moved >=10% the same way" in out
    assert "mkt 86%" in out and "own 0%" in out
    rows = pd.read_csv(csv)
    assert list(rows.columns)[:6] == ["ticker", "kind", "start", "end", "sessions", "value"]
    assert list(rows.columns)[6:] == ["market_share", "market_wide"]
    assert rows.set_index("ticker").loc["T0.NS", "market_wide"]


def test_dq_marker_for_stale_run_is_a_dash(main_mod, Session, monkeypatch, capsys):
    c = _walk()
    c[5:12] = [c[5]] * 7
    with Session() as s:
        _add_asset(s, "S.NS", c)
    monkeypatch.setattr(sys, "argv", ["main.py", "dq"])
    main_mod.cmd_dq()
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "stale_run" in l and "close" in l][0]
    assert line.rstrip().endswith("-")
    assert "0 of 0 return anomalies are market-wide" in out
