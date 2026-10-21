"""
`require_membership_sync` and its use by `python -m scripts.regime_study` (task 480) and
`main.py evaluate`. In-memory SQLite, no network, no real DB.
"""
from __future__ import annotations

import types

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.data.membership_check import (
    check_membership_sync, format_sync, require_membership_sync,
)
from investbot.data.models import Base, IndexMembership
from investbot.data.seed import run_seed
from scripts import regime_study


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


@pytest.fixture()
def stale(synced):
    synced.delete(synced.query(IndexMembership)
                  .filter(IndexMembership.effective_to.isnot(None)).first())
    synced.commit()
    return synced


def _collect():
    lines: list[str] = []
    return lines, lines.append


# ── helper ────────────────────────────────────────────────

def test_no_rows_returns_none_and_prints_nothing(session):
    run_seed(session)
    session.query(IndexMembership).delete()
    session.commit()
    lines, out = _collect()
    assert require_membership_sync(session, out=out) is None
    assert lines == []


def test_in_sync_returns_sync_and_prints_nothing(synced):
    lines, out = _collect()
    sync = require_membership_sync(synced, out=out)
    assert sync is not None and sync.in_sync
    assert lines == []


def test_stale_exits_1_with_report_and_refusal(stale):
    lines, out = _collect()
    with pytest.raises(SystemExit) as e:
        require_membership_sync(stale, out=out, what="run the regime study")
    assert e.value.code == 1
    assert lines == ["\n".join(format_sync(check_membership_sync(stale))),
                     "Refusing to run the regime study on stale membership "
                     "(--allow-stale-membership to continue anyway)."]


def test_default_verb_is_evaluate(stale):
    lines, out = _collect()
    with pytest.raises(SystemExit):
        require_membership_sync(stale, out=out)
    assert lines[-1].startswith("Refusing to evaluate on stale membership")


def test_stale_with_allow_stale_warns_and_returns_sync(stale):
    lines, out = _collect()
    sync = require_membership_sync(stale, allow_stale=True, out=out)
    assert sync is not None and not sync.in_sync
    text = "\n".join(lines)
    assert "WARNING" in text and "OUT OF SYNC" in text
    assert "Continuing on stale membership (--allow-stale-membership)." in text
    assert "Refusing" not in text


# ── regime study runner ───────────────────────────────────

@pytest.fixture()
def runner(monkeypatch):
    """Stubs the session factory and every heavy step; records the calls."""
    calls: dict[str, list] = {"panels": [], "run": []}

    def install(session):
        from investbot.data import db
        monkeypatch.setattr(db, "SessionLocal", lambda: session)
        monkeypatch.setattr(session, "close", lambda: None)

    def boom(*a, **k):
        calls["panels"].append((a, k))
        raise AssertionError("build_study_panels called")

    res = types.SimpleNamespace(
        tests=pd.DataFrame({"verdict": ["a"]}), h6=pd.DataFrame())
    monkeypatch.setattr(regime_study, "build_study_panels", boom)
    monkeypatch.setattr(regime_study, "run",
                        lambda *a, **k: calls["run"].append(k) or ("r.md", res))
    calls["install"] = install
    return calls


def test_runner_exits_1_on_stale_table_before_loading_prices(stale, runner, capsys):
    runner["install"](stale)
    with pytest.raises(SystemExit) as e:
        regime_study.main([])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "OUT OF SYNC" in out
    assert "Refusing to run the regime study on stale membership" in out
    assert runner["panels"] == [] and runner["run"] == []


def test_runner_exits_1_on_stale_table_for_active_universe_too(stale, runner, capsys):
    runner["install"](stale)
    with pytest.raises(SystemExit) as e:
        regime_study.main(["--universe", "active"])
    assert e.value.code == 1
    assert runner["run"] == []


def test_runner_proceeds_with_allow_stale_membership(stale, runner, capsys):
    runner["install"](stale)
    regime_study.main(["--allow-stale-membership"])
    out = capsys.readouterr().out
    assert "WARNING" in out and "OUT OF SYNC" in out
    assert len(runner["run"]) == 1 and runner["run"][0]["universe"] == "index"


def test_runner_in_sync_runs_silently_on_the_index_universe(synced, runner, capsys):
    runner["install"](synced)
    regime_study.main([])
    out = capsys.readouterr().out
    assert "WARNING" not in out and "OUT OF SYNC" not in out
    assert runner["run"][0]["universe"] == "index"


def test_runner_with_empty_table_keeps_the_fallback(session, runner, capsys):
    run_seed(session)
    session.query(IndexMembership).delete()
    session.commit()
    runner["install"](session)
    regime_study.main([])
    assert "WARNING: index_membership has no NIFTY50 rows" in capsys.readouterr().out
    assert runner["run"][0]["universe"] == "active"
