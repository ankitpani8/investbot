"""
NIFTY50 constituent check: parsing, comparison, CLI exit codes and the scheduler job.

No network: every test injects `fetcher`. The check is read-only, so no DB is needed.
"""
from __future__ import annotations

import copy
import csv
import io
import logging
from datetime import date
from pathlib import Path

import pytest

from investbot.config import settings
from investbot.config import tickers as tickers_module
from investbot.config.tickers import NIFTY50_TICKERS
from investbot.data import constituents_check as cc
from investbot.data.constituents_check import (
    ConstituentDiff, ConstituentFetchError, ConstituentParseError,
    check_nifty50, compare_constituents, fetch_index_csv, parse_constituents,
)
from investbot.data.index_membership import (
    NIFTY50_CHANGES_CSV, current_tickers, members_as_of, nifty50_intervals,
)
from investbot.scheduler import jobs

HEADER = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]


def _csv(symbols, header=HEADER, symbol_col="Symbol") -> str:
    """NSE-format CSV text (company names carry commas and dots, so rows need quoting)."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n")
    w.writerow(header)
    for i, s in enumerate(symbols):
        row = {"Company Name": f"{s.title()} Ltd., India", "Industry": "Services",
               symbol_col: s, "Series": "EQ", "ISIN Code": f"INE{i:06d}A01024"}
        w.writerow([row.get(h, "") for h in header])
    return out.getvalue()


def _our_symbols() -> list[str]:
    return [t.removesuffix(".NS") for t in current_tickers()]


def _nse_fixture(**kw) -> str:
    return _csv(_our_symbols(), **kw)


# ── parsing ──────────────────────────────────────────────

def test_real_format_csv_matches_our_list():
    nse = parse_constituents(_nse_fixture())
    assert len(nse) == 50
    diff = compare_constituents(nse, current_tickers())
    assert diff == ConstituentDiff(added=[], removed=[], matches=True, members_disagree=[])


def test_real_format_with_bom_and_padded_headers():
    text = "﻿" + _csv(_our_symbols(), header=["Company Name ", " Industry", " Symbol ", "Series", "ISIN Code"],
                           symbol_col=" Symbol ")
    assert parse_constituents(text) == current_tickers()


def test_one_swapped_symbol_gives_one_added_one_removed():
    symbols = _our_symbols()
    gone = symbols[10]
    symbols[10] = "NEWCO"
    diff = compare_constituents(parse_constituents(_csv(symbols)), current_tickers())
    assert diff.added == ["NEWCO.NS"]
    assert diff.removed == [gone + ".NS"]
    assert not diff.matches


def test_ampersand_and_hyphen_symbols_map():
    symbols = _our_symbols()
    assert "M&M" in symbols and "BAJAJ-AUTO" in symbols
    nse = parse_constituents(_csv(symbols))
    assert "M&M.NS" in nse and "BAJAJ-AUTO.NS" in nse


def test_symbol_overrides_are_applied(monkeypatch):
    monkeypatch.setitem(cc.SYMBOL_OVERRIDES, "NEWNAME", "OLDNAME.NS")
    symbols = _our_symbols()
    symbols[0] = "NEWNAME"
    nse = parse_constituents(_csv(symbols))
    assert nse[0] == "OLDNAME.NS"
    assert "NEWNAME.NS" not in nse


def test_symbol_overrides_empty_by_default():
    assert cc.SYMBOL_OVERRIDES == {}


@pytest.mark.parametrize("body", [
    "<html><head><title>Access Denied</title></head><body>Access Denied</body></html>",
    "",
    "   \n",
    _csv(_our_symbols(), header=["Company Name", "Industry", "Ticker", "Series", "ISIN Code"],
         symbol_col="Ticker"),
    _csv(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "SBIN", "LT", "ONGC", "NTPC", "WIPRO"]),
    _csv([f"S{i}" for i in range(56)]),
    _csv(["RELIANCE"] * 50),
], ids=["html", "empty", "blank", "no-symbol-column", "ten-rows", "fifty-six-rows", "duplicates"])
def test_bad_bodies_raise_parse_error(body):
    with pytest.raises(ConstituentParseError):
        parse_constituents(body)


def test_demerger_window_with_51_names_is_accepted():
    assert len(parse_constituents(_csv(_our_symbols() + ["DUMMYSYM"]))) == 51


# ── fetching ─────────────────────────────────────────────

def test_fetch_uses_injected_fetcher_once():
    calls = []
    text = fetch_index_csv(fetcher=lambda url: calls.append(url) or "body")
    assert text == "body"
    assert calls == [cc.NIFTY50_LIST_URL]


def test_fetch_error_is_wrapped():
    def boom(url):
        raise TimeoutError("timed out")
    with pytest.raises(ConstituentFetchError, match="timed out"):
        fetch_index_csv(fetcher=boom)
    with pytest.raises(ConstituentFetchError):
        check_nifty50(fetcher=boom)


def test_default_fetcher_sends_browser_user_agent(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"\xef\xbb\xbfSymbol\nA\n"

    def fake_urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(cc.urllib.request, "urlopen", fake_urlopen)
    assert fetch_index_csv(timeout=7) == "Symbol\nA\n"
    assert seen["ua"].startswith("Mozilla/5.0")
    assert seen["timeout"] == 7


# ── comparison and the membership chain ──────────────────

def test_members_disagree_is_empty_for_current_data():
    diff = check_nifty50(fetcher=lambda url: _nse_fixture(), today=date.today())
    assert diff.matches
    assert diff.members_disagree == []
    assert set(current_tickers()) == members_as_of(nifty50_intervals(), date.today())


def test_members_disagree_reports_symmetric_difference():
    diff = compare_constituents(["A.NS"], ["A.NS"], as_of_members={"A.NS", "B.NS"})
    assert diff.members_disagree == ["B.NS"]
    assert diff.matches


def test_compare_lists_are_sorted():
    diff = compare_constituents(["Z.NS", "B.NS", "M.NS"], ["M.NS", "Q.NS", "A.NS"])
    assert diff.added == ["B.NS", "Z.NS"]
    assert diff.removed == ["A.NS", "Q.NS"]


# ── read-only ────────────────────────────────────────────

def _db_stat():
    url = settings.DATABASE_URL
    if not url.startswith("sqlite:///"):
        return None
    p = Path(url.removeprefix("sqlite:///"))
    if not p.exists():
        return None
    st = p.stat()
    return st.st_size, st.st_mtime_ns


def test_mismatch_run_changes_nothing(capsys):
    tickers_before = copy.deepcopy(tickers_module.NIFTY50_TICKERS)
    changes_before = NIFTY50_CHANGES_CSV.read_bytes()
    db_before = _db_stat()

    symbols = _our_symbols()
    symbols[3] = "NEWCO"
    text = _csv(symbols)
    assert not check_nifty50(fetcher=lambda url: text).matches
    jobs.run_constituents_check(fetcher=lambda url: text)
    with pytest.raises(SystemExit):
        import main
        main.cmd_constituents(fetcher=lambda url: text)

    assert tickers_module.NIFTY50_TICKERS == tickers_before == NIFTY50_TICKERS
    assert NIFTY50_CHANGES_CSV.read_bytes() == changes_before
    assert _db_stat() == db_before


# ── CLI ──────────────────────────────────────────────────

def _run_cli(fetcher):
    import main
    with pytest.raises(SystemExit) as exc:
        main.cmd_constituents(fetcher=fetcher)
    return exc.value.code


def test_cli_exit_0_on_match(capsys):
    import main
    main.cmd_constituents(fetcher=lambda url: _nse_fixture())   # returns normally = exit 0
    out = capsys.readouterr().out
    assert "NSE count:     50" in out and "Our count:     50" in out
    assert "OK" in out


def test_cli_exit_1_on_mismatch(capsys):
    symbols = _our_symbols()
    gone = symbols[0]
    symbols[0] = "NEWCO"
    assert _run_cli(lambda url: _csv(symbols)) == 1
    out = capsys.readouterr().out
    assert "NEWCO.NS" in out and gone + ".NS" in out
    for part in ("nifty50_changes.csv", "NIFTY50_TICKERS", "CURRENT_AS_OF"):
        assert part in out


def test_cli_exit_2_on_fetch_error(capsys):
    def boom(url):
        raise OSError("network down")
    assert _run_cli(boom) == 2
    assert "network down" in capsys.readouterr().out


def test_cli_exit_2_on_parse_error(capsys):
    assert _run_cli(lambda url: "<html>blocked</html>") == 2


def test_cli_is_registered_and_documented():
    import main
    assert main.COMMANDS["constituents"] is main.cmd_constituents
    assert "main.py constituents" in main.__doc__


# ── scheduler job ────────────────────────────────────────

def test_job_logs_info_on_match(caplog):
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        diff = jobs.run_constituents_check(fetcher=lambda url: _nse_fixture())
    assert diff.matches
    assert [r.levelno for r in caplog.records if "constituent check" in r.getMessage()] == [logging.INFO]


def test_job_logs_one_error_line_on_mismatch(caplog):
    symbols = _our_symbols()
    gone = symbols[5]
    symbols[5] = "NEWCO"
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        jobs.run_constituents_check(fetcher=lambda url: _csv(symbols))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    msg = errors[0].getMessage()
    assert "NEWCO.NS" in msg and gone + ".NS" in msg


def test_job_warns_and_does_not_raise_on_fetch_error(caplog):
    def boom(url):
        raise ConnectionError("refused")
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        assert jobs.run_constituents_check(fetcher=boom) is None
    assert [r.levelno for r in caplog.records if "constituent check" in r.getMessage()] == [logging.WARNING]


def test_job_warns_on_parse_error(caplog):
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        assert jobs.run_constituents_check(fetcher=lambda url: "<html/>") is None
    assert caplog.records[-1].levelno == logging.WARNING


def test_job_wrapper_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(cc, "check_nifty50", boom)
    jobs._job_constituents_check()   # logged, not raised


def test_job_registered_saturday_10_ist():
    scheduler = jobs.create_scheduler()
    job = scheduler.get_job("constituents_check")
    assert job is not None
    assert job.func is jobs._job_constituents_check
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["day_of_week"] == "sat"
    assert fields["hour"] == "10"
    assert fields["minute"] == "0"
    assert str(job.trigger.timezone) == "Asia/Kolkata"
    assert len(scheduler.get_jobs()) == 4
