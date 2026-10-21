"""
Price coverage of membership stints (`investbot.data.membership_coverage`, `main.py coverage`).
Synthetic frames for the pure functions, an in-memory SQLite session for the loaders and
the CLI. No network, no real DB.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from investbot.data import membership_coverage as mc
from investbot.data.membership_coverage import (
    COVERAGE_EDGE_SESSIONS, COVERAGE_OK, coverage_by_year, format_coverage, load_bar_dates,
    load_stints, session_calendar, stint_coverage,
)
from investbot.data.models import Asset, Base, IndexMembership, PriceDaily
from investbot.data.seed import run_seed

ROOT = Path(__file__).resolve().parents[1]
DAYS = pd.bdate_range("2020-01-01", periods=100)  # day i of the synthetic market


def d(i: int) -> pd.Timestamp:
    return DAYS[i]


def _bars(ticker: str, idx) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": [DAYS[i] for i in idx]})


def _stints(*rows) -> pd.DataFrame:
    """rows: (ticker, from_idx, to_idx | None)."""
    return pd.DataFrame({
        "ticker": [r[0] for r in rows],
        "effective_from": pd.to_datetime([DAYS[r[1]] for r in rows]),
        "effective_to": pd.to_datetime([DAYS[r[2]] if r[2] is not None else pd.NaT for r in rows]),
    })


def _run(stints: pd.DataFrame, bars: list[pd.DataFrame], **kw) -> pd.DataFrame:
    """ANCHOR trades every day (it defines the calendar) and is not a member."""
    bar_dates = pd.concat([_bars("ANCHOR", range(100)), *bars], ignore_index=True)
    return stint_coverage(stints, bar_dates, session_calendar(bar_dates), **kw).set_index("ticker")


# ── contract ──────────────────────────────────────────────

def test_stint_columns_are_a_contract():
    out = _run(_stints(("A", 0, 10)), [_bars("A", range(11))])
    assert list(out.reset_index().columns) == [
        "ticker", "effective_from", "effective_to", "sessions", "priced_sessions", "coverage",
        "first_bar", "last_bar", "missing_at_start", "missing_at_end", "late_start", "early_end",
        "status"]
    assert mc.STINT_COLUMNS == list(out.reset_index().columns)


def test_year_columns_are_a_contract():
    bar_dates = _bars("A", range(100))
    out = coverage_by_year(_stints(("A", 0, None)), bar_dates, session_calendar(bar_dates))
    assert list(out.columns) == ["year", "sessions", "members_median", "members_min",
                                 "priced_median", "priced_min", "priced_share_mean"]


def test_constants():
    assert COVERAGE_OK == 0.98 and COVERAGE_EDGE_SESSIONS == 5


def test_session_calendar_is_sorted_union_of_all_tickers():
    bars = pd.concat([_bars("A", [5, 2]), _bars("B", [2, 9]), _bars("A", [5])])
    cal = session_calendar(bars)
    assert list(cal) == [d(2), d(5), d(9)]
    assert cal.is_monotonic_increasing and cal.is_unique


# ── statuses ──────────────────────────────────────────────

def test_each_status():
    stints = _stints(
        ("NOPRICE", 10, 60),
        ("OUT", 30, 60),
        ("EARLY", 10, 60),
        ("LATE", 10, 60),
        ("HOLES", 10, 60),
        ("GOOD", 10, 60),
    )
    holes = [i for i in range(10, 61) if i not in (20, 30, 40, 50)]  # 47 of 51 priced = 92%
    out = _run(stints, [
        _bars("OUT", range(0, 10)),
        _bars("EARLY", range(10, 31)),           # bars stop mid-stint: renamed symbol pattern
        _bars("LATE", range(25, 61)),
        _bars("HOLES", holes),
        _bars("GOOD", range(10, 61)),
    ])
    assert out["status"].to_dict() == {
        "NOPRICE": "no_prices", "OUT": "outside_stint", "EARLY": "early_end",
        "LATE": "late_start", "HOLES": "holes", "GOOD": "ok"}

    early = out.loc["EARLY"]
    assert (early["sessions"], early["priced_sessions"]) == (51, 21)
    assert early["missing_at_start"] == 0 and early["missing_at_end"] == 30
    assert early["first_bar"] == d(10) and early["last_bar"] == d(30)
    assert early["coverage"] == pytest.approx(21 / 51)

    late = out.loc["LATE"]
    assert late["missing_at_start"] == 15 and late["missing_at_end"] == 0
    assert bool(late["late_start"]) and not bool(late["early_end"])

    holes_row = out.loc["HOLES"]
    assert holes_row["coverage"] == pytest.approx(47 / 51) and holes_row["coverage"] < COVERAGE_OK
    assert holes_row["missing_at_start"] == 0 and holes_row["missing_at_end"] == 0

    nop = out.loc["NOPRICE"]
    assert pd.isna(nop["first_bar"]) and pd.isna(nop["last_bar"])
    assert nop["priced_sessions"] == 0 and nop["coverage"] == 0.0

    outside = out.loc["OUT"]
    assert outside["priced_sessions"] == 0 and outside["first_bar"] == d(0) and outside["last_bar"] == d(9)


def test_edge_threshold_is_strictly_more_than_five_sessions():
    out = _run(_stints(("FIVE", 10, 60), ("SIX", 10, 60)),
               [_bars("FIVE", range(15, 61)), _bars("SIX", range(16, 61))])
    assert out.loc["FIVE", "missing_at_start"] == 5 and not out.loc["FIVE", "late_start"]
    assert out.loc["SIX", "missing_at_start"] == 6 and out.loc["SIX", "late_start"]
    assert out.loc["FIVE", "status"] == "holes"  # 46 of 51 = 90%, edges are fine
    assert out.loc["SIX", "status"] == "late_start"


def test_early_end_wins_over_late_start():
    out = _run(_stints(("BOTH", 10, 60)), [_bars("BOTH", range(25, 41))])
    row = out.loc["BOTH"]
    assert bool(row["late_start"]) and bool(row["early_end"])
    assert row["status"] == "early_end"


def test_open_ended_stint_counts_to_last_calendar_date_or_as_of():
    stints = _stints(("OPEN", 50, None))
    bars = [_bars("OPEN", range(50, 100))]
    out = _run(stints, bars)
    assert out.loc["OPEN", "sessions"] == 50 and out.loc["OPEN", "status"] == "ok"
    assert pd.isna(out.loc["OPEN", "effective_to"])  # the input is echoed, not rewritten

    capped = _run(stints, bars, as_of=d(70).date())
    assert capped.loc["OPEN", "sessions"] == 21 and capped.loc["OPEN", "priced_sessions"] == 21
    assert capped.loc["OPEN", "missing_at_end"] == 0


def test_open_ended_stint_whose_bars_stop_is_early_end():
    out = _run(_stints(("OPEN", 50, None)), [_bars("OPEN", range(50, 70))])
    assert out.loc["OPEN", "missing_at_end"] == 30 and out.loc["OPEN", "status"] == "early_end"


def test_ticker_with_two_stints_gets_a_row_each():
    stints = _stints(("TWO", 0, 20), ("TWO", 60, 80))
    bars = [_bars("TWO", range(0, 21))]  # bars only for the first stint
    out = stint_coverage(stints, pd.concat([_bars("ANCHOR", range(100)), *bars]),
                         session_calendar(_bars("ANCHOR", range(100))))
    assert list(out["ticker"]) == ["TWO", "TWO"]
    assert list(out["status"]) == ["ok", "outside_stint"]
    assert list(out["sessions"]) == [21, 21] and list(out["priced_sessions"]) == [21, 0]
    # first_bar / last_bar are the ticker's overall bars, repeated on both rows
    assert list(out["first_bar"]) == [d(0), d(0)] and list(out["last_bar"]) == [d(20), d(20)]


def test_stint_after_the_last_calendar_date_is_kept_with_nan_coverage():
    bar_dates = pd.concat([_bars("ANCHOR", range(50)), _bars("NEW", range(10, 20))])
    stints = pd.DataFrame({
        "ticker": ["NEW", "FUTURE", "FUTURE_NOBARS"],
        "effective_from": pd.to_datetime([d(10), d(60), d(60)]),
        "effective_to": pd.to_datetime([d(30), pd.NaT, pd.NaT]),
    })
    out = stint_coverage(stints, bar_dates, session_calendar(bar_dates)).set_index("ticker")
    assert len(out) == 3
    for t in ("FUTURE", "FUTURE_NOBARS"):
        assert out.loc[t, "sessions"] == 0 and out.loc[t, "status"] == "ok"
        assert np.isnan(out.loc[t, "coverage"])


def test_empty_inputs_keep_the_columns():
    empty_bars = pd.DataFrame({"ticker": [], "date": []})
    cal = session_calendar(empty_bars)
    out = stint_coverage(_stints(("A", 0, 5)), empty_bars, cal)
    assert len(out) == 1 and list(out.columns) == mc.STINT_COLUMNS
    assert out.loc[0, "status"] in ("no_prices", "ok")
    none = stint_coverage(_stints()[:0], empty_bars, cal)
    assert len(none) == 0 and list(none.columns) == mc.STINT_COLUMNS
    assert list(coverage_by_year(_stints()[:0], empty_bars, cal).columns) == mc.YEAR_COLUMNS


# ── per year ──────────────────────────────────────────────

def _year_example():
    """
    Calendar (union of A's bars): 2020-12-30, 12-31, 2021-01-01, 01-04, 01-05, 01-06.
    A: open stint from 12-30, bars every day.  B: stint 12-31..01-04, bars on 12-31 and 01-04.
    members/priced per date: 1/1, 2/2, 2/1, 2/2, 1/1, 1/1.
    """
    days = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-04",
                           "2021-01-05", "2021-01-06"])
    bar_dates = pd.concat([
        pd.DataFrame({"ticker": "A", "date": days}),
        pd.DataFrame({"ticker": "B", "date": pd.to_datetime(["2020-12-31", "2021-01-04"])}),
    ], ignore_index=True)
    stints = pd.DataFrame({
        "ticker": ["A", "B"],
        "effective_from": pd.to_datetime(["2020-12-30", "2020-12-31"]),
        "effective_to": pd.to_datetime([pd.NaT, "2021-01-04"]),
    })
    return stints, bar_dates


def test_by_year_on_a_hand_computed_example():
    stints, bar_dates = _year_example()
    out = coverage_by_year(stints, bar_dates, session_calendar(bar_dates)).set_index("year")
    assert list(out.index) == [2020, 2021]
    r20, r21 = out.loc[2020], out.loc[2021]
    assert r20["sessions"] == 2
    assert (r20["members_median"], r20["members_min"]) == (1.5, 1)
    assert (r20["priced_median"], r20["priced_min"]) == (1.5, 1)
    assert r20["priced_share_mean"] == pytest.approx(1.0)
    assert r21["sessions"] == 4
    assert (r21["members_median"], r21["members_min"]) == (1.5, 1)
    assert (r21["priced_median"], r21["priced_min"]) == (1.0, 1)
    assert r21["priced_share_mean"] == pytest.approx((0.5 + 1 + 1 + 1) / 4)

    sc = stint_coverage(stints, bar_dates, session_calendar(bar_dates)).set_index("ticker")
    assert sc.loc["B", "sessions"] == 3 and sc.loc["B", "priced_sessions"] == 2
    assert sc.loc["B", "coverage"] == pytest.approx(2 / 3) and sc.loc["B", "status"] == "holes"
    assert sc.loc["A", "sessions"] == 6 and sc.loc["A", "status"] == "ok"


def test_by_year_stint_bounds_are_inclusive():
    stints, bar_dates = _year_example()
    cal = session_calendar(bar_dates)
    out = coverage_by_year(stints.iloc[[1]], bar_dates, cal).set_index("year")
    # B alone: members on 12-31, 01-01 and 01-04 (both end points included)
    assert out.loc[2020, "members_min"] == 0 and out.loc[2020, "members_median"] == 0.5
    assert out.loc[2021, "members_median"] == 0.5 and out.loc[2021, "members_min"] == 0


def test_by_year_truncating_bars_leaves_earlier_years_unchanged():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    bars = pd.concat([
        pd.DataFrame({"ticker": t, "date": dates[rng.random(len(dates)) < p]})
        for t, p in (("A", 1.0), ("B", 0.9), ("C", 0.5))
    ], ignore_index=True)
    stints = pd.DataFrame({
        "ticker": ["A", "B", "C", "A"],
        "effective_from": pd.to_datetime(["2018-01-01", "2018-06-01", "2019-03-01", "2021-05-01"]),
        "effective_to": pd.to_datetime(["2020-12-31", "2021-07-01", pd.NaT, pd.NaT]),
    })
    full = coverage_by_year(stints, bars, session_calendar(bars)).set_index("year")

    cut = pd.Timestamp("2020-09-15")
    trunc = bars[bars["date"] <= cut]
    part = coverage_by_year(stints, trunc, session_calendar(trunc)).set_index("year")
    assert list(part.index) == [2018, 2019, 2020]
    pd.testing.assert_frame_equal(part.loc[[2018, 2019]], full.loc[[2018, 2019]])
    assert part.loc[2020, "sessions"] < full.loc[2020, "sessions"]  # d's own year may differ


# ── format ────────────────────────────────────────────────

def test_format_lists_non_ok_stints_sorted_by_status_then_start():
    stints = _stints(("GOOD", 10, 60), ("LATE2", 20, 60), ("NOPRICE", 30, 60),
                     ("EARLY", 40, 60), ("LATE1", 10, 60), ("NOPRICE0", 5, 60))
    bar_dates = pd.concat([_bars("ANCHOR", range(100)), _bars("GOOD", range(10, 61)),
                           _bars("LATE2", range(40, 61)), _bars("LATE1", range(40, 61)),
                           _bars("EARLY", range(40, 50))])
    cal = session_calendar(bar_dates)
    sc = stint_coverage(stints, bar_dates, cal)
    lines = format_coverage(sc, coverage_by_year(stints, bar_dates, cal))
    text = "\n".join(lines)
    assert re.search(r"no_prices\s+2", text) and re.search(r"ok\s+1", text)
    order = [next(i for i, l in enumerate(lines) if l.strip().startswith(t + " ") and "..".join(["", ""]) in l)
             for t in ("NOPRICE0", "NOPRICE", "EARLY", "LATE1", "LATE2")]
    assert order == sorted(order)
    assert not any(l.strip().startswith("GOOD ") for l in lines)
    assert any("2020" in l and "%" in l for l in lines)  # per-year table
    assert "0.0%" in text and "open" not in text


def test_format_shows_open_stints_and_truncates():
    stints = _stints(*[(f"T{i:02d}", i, None) for i in range(10)])
    bar_dates = _bars("ANCHOR", range(100))
    cal = session_calendar(bar_dates)
    sc = stint_coverage(stints, bar_dates, cal)
    lines = format_coverage(sc, coverage_by_year(stints, bar_dates, cal), max_rows=3)
    assert sum(1 for l in lines if l.strip().startswith("T0")) == 3
    assert any("and 7 more" in l for l in lines)
    assert any("..open" in l for l in lines)


def test_format_all_ok():
    stints = _stints(("A", 0, None))
    bar_dates = _bars("A", range(100))
    cal = session_calendar(bar_dates)
    lines = format_coverage(stint_coverage(stints, bar_dates, cal), coverage_by_year(stints, bar_dates, cal))
    assert any("No stint with a coverage problem" in l for l in lines)


# ── no lookahead / purity ─────────────────────────────────

def test_stint_coverage_is_pure():
    stints = _stints(("A", 10, 60))
    bar_dates = pd.concat([_bars("ANCHOR", range(100)), _bars("A", range(10, 61))])
    before = (stints.copy(), bar_dates.copy())
    _ = stint_coverage(stints, bar_dates, session_calendar(bar_dates))
    pd.testing.assert_frame_equal(stints, before[0])
    pd.testing.assert_frame_equal(bar_dates, before[1])


def test_no_scoring_or_evaluation_module_imports_it():
    for pkg in ("scoring", "evaluation"):
        for f in (ROOT / "investbot" / pkg).rglob("*.py"):
            assert "membership_coverage" not in f.read_text(), f
    src = (ROOT / "investbot" / "data" / "membership_coverage.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+investbot\.evaluation", src, re.M)
    assert not re.search(r"^\s*(from|import)\s+investbot\.scoring", src, re.M)


def test_module_has_no_dropna_and_loaders_are_last():
    src = (ROOT / "investbot" / "data" / "membership_coverage.py").read_text()
    assert "dropna" not in src
    first_loader = src.index("def load_stints")
    for name in ("session_calendar", "stint_coverage", "coverage_by_year", "format_coverage"):
        assert src.index(f"def {name}") < first_loader


# ── DB loaders and the CLI ────────────────────────────────

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.engine_ = engine
    yield s
    s.close()
    engine.dispose()


def _add_bars(session, ticker, days, asset_type=None):
    asset = session.query(Asset).filter_by(ticker=ticker).one()
    if asset_type:
        asset.asset_type = asset_type
    for day in days:
        session.add(PriceDaily(asset_id=asset.id, date=day, open=1.0, high=2.0, low=0.5,
                               close=1.5, adj_close=1.5, volume=10))
    session.flush()


@pytest.fixture()
def seeded(session):
    run_seed(session)
    session.commit()
    return session


def _counts(session):
    return (session.query(Asset).count(), session.query(IndexMembership).count(),
            session.query(PriceDaily).count())


def test_load_stints_matches_index_membership(seeded):
    df = load_stints(seeded)
    assert list(df.columns) == ["ticker", "effective_from", "effective_to"]
    assert len(df) == seeded.query(IndexMembership).count()
    assert df["effective_to"].isna().sum() == 50  # current members have an open end
    assert pd.api.types.is_datetime64_any_dtype(df["effective_from"])
    assert len(load_stints(seeded, "NOT_AN_INDEX")) == 0


def test_load_bar_dates_stocks_only_and_selects_no_ohlcv(seeded):
    tickers = sorted(load_stints(seeded)["ticker"].unique())
    stock, other = tickers[0], tickers[1]
    _add_bars(seeded, stock, [date(2024, 1, 2), date(2024, 1, 3)])
    _add_bars(seeded, other, [date(2024, 1, 2)], asset_type="crypto")
    seeded.commit()

    statements: list[str] = []
    event.listen(seeded.engine_, "before_cursor_execute",
                 lambda conn, cur, stmt, params, ctx, many: statements.append(stmt))
    df = load_bar_dates(seeded)
    assert list(df.columns) == ["ticker", "date"]
    assert df["ticker"].tolist() == [stock, stock]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    sql = " ".join(statements).lower()
    assert "price_daily.date" in sql
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        assert f"price_daily.{col}" not in sql


@pytest.fixture()
def main_mod(seeded, monkeypatch):
    import main
    monkeypatch.setattr(main, "SessionLocal", lambda: seeded)
    monkeypatch.setattr(seeded, "close", lambda: None)
    return main


def _calendar_days():
    return list(pd.bdate_range("2024-01-01", periods=30).date)


def test_cmd_coverage_prints_report_writes_csv_and_changes_nothing(main_mod, seeded, tmp_path, capsys):
    current = load_stints(seeded).query("effective_to != effective_to")["ticker"].tolist()
    days = _calendar_days()
    _add_bars(seeded, current[0], days)           # one current member fully priced
    _add_bars(seeded, current[1], days[:10])      # bars stop early
    seeded.commit()
    before = _counts(seeded)

    csv = tmp_path / "cov.csv"
    main_mod.cmd_coverage(["--csv", str(csv)])    # returns normally: exit 0
    out = capsys.readouterr().out

    assert "OUT OF SYNC" not in out
    assert "membership stints" in out and "no_prices" in out and "early_end" in out
    assert "Members with a bar, per calendar year" in out
    assert _counts(seeded) == before

    written = pd.read_csv(csv)
    assert list(written.columns) == mc.STINT_COLUMNS
    assert len(written) == len(load_stints(seeded))
    row = written[written["ticker"] == current[1]].iloc[-1]
    assert row["status"] == "early_end" and row["priced_sessions"] == 10
    assert written[written["ticker"] == current[0]].iloc[-1]["status"] == "ok"


def test_cmd_coverage_warns_on_stale_table_and_still_runs(main_mod, seeded, capsys):
    row = seeded.query(IndexMembership).filter(IndexMembership.effective_to.isnot(None)).first()
    seeded.delete(row)
    seeded.commit()
    main_mod.cmd_coverage([])
    out = capsys.readouterr().out
    assert "OUT OF SYNC" in out and "python main.py setup" in out
    assert "Price coverage of" in out


def test_cmd_coverage_on_an_empty_database(session, monkeypatch, capsys):
    import main
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(session, "close", lambda: None)
    main.cmd_coverage([])
    assert "Price coverage of 0 membership stints" in capsys.readouterr().out


def test_status_prints_the_stints_line(main_mod, seeded, capsys):
    main_mod.cmd_status()
    n = seeded.query(IndexMembership).count()
    assert f"Stints not fully priced: {n} of {n} (python main.py coverage)" in capsys.readouterr().out


def test_coverage_is_in_the_cli_and_docs():
    import main
    assert "coverage" in main.COMMANDS
    assert "python main.py coverage" in main.__doc__
    assert re.search(r"main\.py\s+CLI:.*\| coverage", (ROOT / "CLAUDE.md").read_text())
