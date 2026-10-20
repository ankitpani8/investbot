"""
`check_membership_sync` / `format_sync` and the stale-membership guards in `main.py`
(`status`, `ingest --members`, `evaluate`). In-memory SQLite, no network, no real DB.
"""
from __future__ import annotations

import dataclasses
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import object_session, sessionmaker

from investbot.data import index_membership as im
from investbot.data.membership_check import (
    MembershipSync, check_membership_sync, expected_intervals, format_sync,
)
from investbot.data.models import Asset, Base, IndexMembership, PriceDaily
from investbot.data.seed import run_seed


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture()
def synced(session):
    run_seed(session)
    return session


def _snapshot(session):
    return sorted((r.asset_id, r.index_code, r.effective_from, r.effective_to, r.source, r.verified)
                  for r in session.query(IndexMembership))


def _ticker(row) -> str:
    return object_session(row).get(Asset, row.asset_id).ticker


def _label(row) -> str:
    to = row.effective_to
    return f"{_ticker(row)} {row.effective_from}..{to if to is not None else 'open'}"


def _closed_stint(session) -> IndexMembership:
    return session.query(IndexMembership).filter(IndexMembership.effective_to.isnot(None)).first()


def test_expected_intervals_come_from_the_reconstruction():
    pd.testing.assert_frame_equal(expected_intervals(), im.nifty50_intervals())
    with pytest.raises(ValueError):
        expected_intervals("NIFTY999")


def test_in_sync_right_after_sync(synced):
    s = check_membership_sync(synced)
    assert s.in_sync and s.index_code == "NIFTY50"
    assert s.db_rows == s.expected_rows == len(im.nifty50_intervals())
    assert s.missing == s.extra == s.members_without_asset == ()
    assert format_sync(s)[0].endswith(f"in sync with the change list ({s.db_rows} stints).")
    assert not any("Fix" in line for line in format_sync(s))


def test_deleted_stint_is_missing(synced):
    row = _closed_stint(synced)
    label = _label(row)
    synced.delete(row)
    synced.commit()
    s = check_membership_sync(synced)
    assert not s.in_sync
    assert s.missing == (label,) and s.extra == ()
    assert s.db_rows == s.expected_rows - 1


def test_altered_effective_to_is_missing_and_extra(synced):
    row = _closed_stint(synced)
    want = _label(row)
    row.effective_to = date(2030, 1, 1)
    synced.commit()
    s = check_membership_sync(synced)
    assert not s.in_sync
    assert s.missing == (want,)
    assert s.extra == (f"{_ticker(row)} {row.effective_from}..2030-01-01",)
    assert s.db_rows == s.expected_rows


def test_open_ended_stint_compared_as_null(synced):
    row = synced.query(IndexMembership).filter(IndexMembership.effective_to.is_(None)).first()
    row.effective_to = date(2030, 1, 1)
    synced.commit()
    s = check_membership_sync(synced)
    assert s.missing == (f"{_ticker(row)} {row.effective_from}..open",)


def test_change_list_ticker_without_asset_row(synced):
    ticker = _ticker(_closed_stint(synced))
    asset = synced.query(Asset).filter_by(ticker=ticker).one()
    synced.query(IndexMembership).filter_by(asset_id=asset.id).delete()
    synced.delete(asset)
    synced.commit()
    s = check_membership_sync(synced)
    assert not s.in_sync
    assert s.members_without_asset == (ticker,)
    assert ticker not in s.members_without_prices
    assert all(m.startswith(f"{ticker} ") for m in s.missing) and s.missing


def test_member_without_prices_is_still_in_sync(synced):
    s = check_membership_sync(synced)
    assert s.in_sync and s.members_without_prices        # no price rows were seeded at all
    ticker = s.members_without_prices[0]
    asset = synced.query(Asset).filter_by(ticker=ticker).one()
    synced.add(PriceDaily(asset_id=asset.id, date=date(2026, 1, 2), close=1.0))
    synced.commit()
    s2 = check_membership_sync(synced)
    assert s2.in_sync
    assert ticker not in s2.members_without_prices
    assert len(s2.members_without_prices) == len(s.members_without_prices) - 1


def test_empty_table_is_out_of_sync_with_everything_missing(synced):
    synced.query(IndexMembership).delete()
    synced.commit()
    s = check_membership_sync(synced)
    assert not s.in_sync
    assert s.db_rows == 0 and len(s.missing) == s.expected_rows and s.extra == ()


def test_check_is_read_only(synced):
    row = _closed_stint(synced)
    row.effective_to = date(2030, 1, 1)
    synced.commit()
    before = _snapshot(synced)
    n_assets = synced.query(Asset).count()
    check_membership_sync(synced)
    assert not synced.new and not synced.dirty and not synced.deleted
    assert _snapshot(synced) == before
    assert synced.query(Asset).count() == n_assets


def _sync(**kw) -> MembershipSync:
    base = dict(index_code="NIFTY50", in_sync=False, db_rows=0, expected_rows=0, missing=(),
                extra=(), members_without_asset=(), members_without_prices=())
    return MembershipSync(**{**base, **kw})


def test_format_sync_truncates_long_lists_and_names_the_fix():
    missing = tuple(f"T{i}.NS 2020-01-01..open" for i in range(25))
    lines = format_sync(_sync(missing=missing, expected_rows=25), max_items=10)
    assert sum(line.startswith("    T") for line in lines) == 10
    assert "    … and 15 more" in lines
    assert any("python main.py setup" in line for line in lines)
    short = format_sync(_sync(missing=missing[:3]), max_items=10)
    assert not any("more" in line for line in short)


def test_membership_sync_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _sync().in_sync = True


# ── main.py guards ────────────────────────────────────────

@pytest.fixture()
def main_mod(session, monkeypatch):
    import main
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(session, "close", lambda: None)
    return main


@pytest.fixture()
def no_download(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("download attempted")
    from investbot.data.ingestion import prices
    monkeypatch.setattr(prices, "ingest_all_prices", boom)
    monkeypatch.setattr(prices.yf, "download", boom, raising=False)


def test_ingest_members_exits_1_on_stale_table_without_downloading(
        synced, main_mod, no_download, capsys):
    synced.delete(_closed_stint(synced))
    synced.commit()
    with pytest.raises(SystemExit) as e:
        main_mod.cmd_ingest(["--members"])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "OUT OF SYNC" in out and "python main.py setup" in out


def test_ingest_members_exits_1_on_empty_table(session, main_mod, no_download, capsys):
    run_seed(session)
    session.query(IndexMembership).delete()
    session.commit()
    with pytest.raises(SystemExit) as e:
        main_mod.cmd_ingest(["--members"])
    assert e.value.code == 1
    assert "OUT OF SYNC" in capsys.readouterr().out


def test_ingest_members_proceeds_when_in_sync(synced, main_mod, monkeypatch):
    from investbot.data.ingestion import prices
    calls = []
    stats = {k: 0 for k in ("daily_inserted", "daily_updated", "daily_rejected", "actions_inserted",
                            "actions_updated", "full_downloads", "incremental_downloads")}
    stats.update(errors=[], no_data=[])
    monkeypatch.setattr(prices, "ingest_all_prices", lambda *a, **k: calls.append(k) or stats)
    main_mod.cmd_ingest(["--members"])
    assert len(calls) == 1


def test_status_reports_sync(synced, main_mod, capsys):
    main_mod.cmd_status()
    assert "in sync with nifty50_changes.csv: yes" in capsys.readouterr().out
    synced.delete(_closed_stint(synced))
    synced.commit()
    main_mod.cmd_status()
    out = capsys.readouterr().out
    assert "in sync with nifty50_changes.csv: no" in out and "python main.py setup" in out


@pytest.fixture()
def eval_stubs(monkeypatch):
    from investbot.evaluation import evaluate, report
    calls = []
    monkeypatch.setattr(report, "run_evaluation",
                        lambda *a, **k: calls.append(k) or ("r.md", "m.csv", None, None))
    monkeypatch.setattr(evaluate, "headline", lambda t: pd.DataFrame({"promising": [True]}))
    monkeypatch.setattr(report, "universe_note", lambda info: "note")
    monkeypatch.setattr(report, "dividend_warning", lambda info: "")
    return calls


def _evaluate(main_mod, monkeypatch, *flags):
    monkeypatch.setattr("sys.argv", ["main.py", "evaluate", *flags])
    main_mod.cmd_evaluate()


def test_evaluate_exits_1_on_stale_table(synced, main_mod, eval_stubs, monkeypatch, capsys):
    synced.delete(_closed_stint(synced))
    synced.commit()
    with pytest.raises(SystemExit) as e:
        _evaluate(main_mod, monkeypatch)
    assert e.value.code == 1
    assert "OUT OF SYNC" in capsys.readouterr().out
    assert eval_stubs == []


def test_evaluate_continues_with_allow_stale_membership(
        synced, main_mod, eval_stubs, monkeypatch, capsys):
    synced.delete(_closed_stint(synced))
    synced.commit()
    _evaluate(main_mod, monkeypatch, "--allow-stale-membership")
    out = capsys.readouterr().out
    assert "WARNING" in out and "OUT OF SYNC" in out
    assert len(eval_stubs) == 1 and eval_stubs[0]["universe"] == "index"


def test_evaluate_with_empty_table_keeps_the_fallback(session, main_mod, eval_stubs, monkeypatch, capsys):
    run_seed(session)
    session.query(IndexMembership).delete()
    session.commit()
    _evaluate(main_mod, monkeypatch)
    out = capsys.readouterr().out
    assert "WARNING: index_membership has no NIFTY50 rows" in out
    assert eval_stubs[0]["universe"] == "active"


def test_evaluate_in_sync_runs_without_a_flag(synced, main_mod, eval_stubs, monkeypatch):
    _evaluate(main_mod, monkeypatch)
    assert eval_stubs[0]["universe"] == "index"
