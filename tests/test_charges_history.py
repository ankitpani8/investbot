"""Dated non-STT charges: the committed file, the lookup and the regime overlay. No DB, no network."""

from __future__ import annotations

import csv
import dataclasses
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from investbot.trading import charges_history as ch
from investbot.trading.tax_rules import regime_for, regime_params

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return ch.load_charge_history()


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "charges.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ch.COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({**_ROW, **r})
    return path


_ROW = {
    "charge": "exchange_txn_charge",
    "effective_from": "2020-01-01",
    "effective_to": "",
    "rate": "0.00003",
    "side": "both",
    "basis": "traded_value",
    "source_url": "https://example.org/circular.pdf",
    "source_title": "t",
    "note": "n",
}


# ---------------------------------------------------------------- the committed file

def test_committed_file_validates(history):
    assert list(history.columns) == list(ch.COLUMNS)
    assert set(history["charge"]) == set(ch.CHARGES)
    assert (history["source_url"].str.startswith("http")).all()
    assert (history["source_title"].str.len() > 0).all()
    assert history["rate"].between(0.0, 1.0, inclusive="left").all()


def test_each_charge_has_one_open_ended_current_row_per_side(history):
    for (charge, side), rows in ch._by_effective_side(history):
        open_rows = rows[rows["effective_to"].map(lambda v: v is None)]
        assert len(open_rows) == 1, (charge, side)
        assert open_rows.iloc[0]["effective_from"] == rows["effective_from"].max()


def _gaps(history: pd.DataFrame) -> list[tuple[str, str, date, date]]:
    found = []
    for (charge, side), rows in ch._by_effective_side(history):
        spans = sorted(zip(rows["effective_from"], rows["effective_to"]), key=lambda s: s[0])
        for (_s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            if s2 > e1 + timedelta(days=1):
                found.append((charge, side, e1 + timedelta(days=1), s2 - timedelta(days=1)))
    return sorted(found)


def test_rows_are_contiguous_except_known_gaps(history):
    assert _gaps(history) == sorted(ch.KNOWN_GAPS)


def test_first_sourced_dates_match_the_file(history):
    first = {
        key: rows["effective_from"].min() for key, rows in ch._by_effective_side(history)
    }
    assert first == ch.FIRST_SOURCED


def test_known_values_at_documented_change_dates(history):
    def on(d, side="buy"):
        return ch.charges_on(d, history, side=side)

    # NSE circulars: Rs 3.50/lakh to 30 Sep 2009, 3.25 from 1 Oct 2009, 3.45 from 1 Jan 2021 ...
    assert on(date(2009, 9, 30))["exchange_txn_charge"] == pytest.approx(3.5e-5)
    assert on(date(2009, 10, 1))["exchange_txn_charge"] == pytest.approx(3.25e-5)
    assert on(date(2020, 12, 31))["exchange_txn_charge"] == pytest.approx(3.25e-5)
    assert on(date(2021, 1, 1))["exchange_txn_charge"] == pytest.approx(3.45e-5)
    assert on(date(2024, 9, 30))["exchange_txn_charge"] == pytest.approx(3.22e-5)
    assert on(date(2024, 10, 1))["exchange_txn_charge"] == pytest.approx(2.97e-5)
    # SEBI turnover fee: Rs 10, 15, 10, 5 (Jun 2020-Mar 2021), 10 per crore
    assert on(date(2017, 3, 31))["sebi_turnover_fee"] == pytest.approx(1e-6)
    assert on(date(2017, 4, 1))["sebi_turnover_fee"] == pytest.approx(1.5e-6)
    assert on(date(2019, 4, 1))["sebi_turnover_fee"] == pytest.approx(1e-6)
    assert on(date(2020, 6, 1))["sebi_turnover_fee"] == pytest.approx(5e-7)
    assert on(date(2021, 4, 1))["sebi_turnover_fee"] == pytest.approx(1e-6)
    # service tax / GST
    assert on(date(2017, 6, 30))["gst_or_service_tax"] == pytest.approx(0.15)
    assert on(date(2017, 7, 1))["gst_or_service_tax"] == pytest.approx(0.18)
    assert on(date(2015, 11, 14))["gst_or_service_tax"] == pytest.approx(0.14)
    assert on(date(2015, 11, 15))["gst_or_service_tax"] == pytest.approx(0.145)
    # stamp duty: Maharashtra 0.01% both sides, then 0.015% on the buy side only
    assert on(date(2020, 6, 30))["stamp_duty"] == pytest.approx(1e-4)
    assert on(date(2020, 6, 30), "sell")["stamp_duty"] == pytest.approx(1e-4)
    assert on(date(2020, 7, 1))["stamp_duty"] == pytest.approx(1.5e-4)
    assert on(date(2020, 7, 1), "sell")["stamp_duty"] == 0.0


# ---------------------------------------------------------------- charges_on

def _two_rows() -> pd.DataFrame:
    rows = [
        {**_ROW, "effective_from": date(2020, 1, 1), "effective_to": date(2020, 12, 31), "rate": 0.00003},
        {**_ROW, "effective_from": date(2021, 1, 1), "effective_to": None, "rate": 0.00004},
    ]
    return pd.DataFrame(rows, columns=list(ch.COLUMNS)).astype(
        {"effective_from": object, "effective_to": object}
    )


def test_charges_on_is_exact_at_boundaries_and_none_outside():
    h = _two_rows()
    key = "exchange_txn_charge"
    assert ch.charges_on(date(2019, 12, 31), h)[key] is None
    assert ch.charges_on(date(2020, 1, 1), h)[key] == 0.00003
    assert ch.charges_on(date(2020, 12, 31), h)[key] == 0.00003
    assert ch.charges_on(date(2021, 1, 1), h)[key] == 0.00004
    assert ch.charges_on(date(2099, 1, 1), h)[key] == 0.00004
    # the other three charges have no rows at all
    others = {k: v for k, v in ch.charges_on(date(2020, 6, 1), h).items() if k != key}
    assert set(others) == set(ch.CHARGES) - {key}
    assert all(v is None for v in others.values())


def test_charges_on_ignores_rows_that_start_later(history):
    """No lookahead: dropping every row that starts after d cannot change the answer on d."""
    for d in (date(2006, 3, 1), date(2012, 5, 5), date(2017, 7, 1), date(2020, 7, 1), date(2025, 1, 1)):
        known = history[history["effective_from"] <= d]
        for side in ("buy", "sell"):
            assert ch.charges_on(d, known, side=side) == ch.charges_on(d, history, side=side)


def test_charges_on_rejects_bad_side(history):
    with pytest.raises(ValueError):
        ch.charges_on(date(2025, 1, 1), history, side="both")


def test_charges_on_flags_overlapping_unvalidated_rows():
    h = pd.concat([_two_rows(), _two_rows().iloc[[1]].assign(rate=0.5)], ignore_index=True)
    with pytest.raises(ValueError):
        ch.charges_on(date(2022, 1, 1), h)


# ---------------------------------------------------------------- with_dated_charges

_FOUR = ("stamp_duty_rate", "exchange_txn_charge_rate", "sebi_turnover_fee_rate", "gst_rate")


def test_with_dated_charges_changes_only_the_four_fields(history):
    d = date(2014, 6, 2)
    stored = regime_for("equity", d)
    dated, unsourced = ch.with_dated_charges(stored, d, history)
    assert unsourced == []
    before, after = dataclasses.asdict(stored), dataclasses.asdict(dated)
    changed = {k for k in before if before[k] != after[k]}
    assert changed <= set(_FOUR)
    # 2014 exchange charge and SEBI fee equal the stored copies; stamp duty and service tax differ
    assert changed == {"stamp_duty_rate", "gst_rate"}
    assert dated.stt_buy_rate == stored.stt_buy_rate
    assert dated.stt_sell_rate == stored.stt_sell_rate
    assert dated.stcg_rate == stored.stcg_rate
    assert dated.regime_name == stored.regime_name
    assert dated.gst_rate == pytest.approx(0.1236)
    assert dated.stamp_duty_rate == pytest.approx(1e-4)
    assert dated.exchange_txn_charge_rate == pytest.approx(3.25e-5)
    assert dated.sebi_turnover_fee_rate == pytest.approx(1e-6)


def test_with_dated_charges_reports_unsourced_fields_and_keeps_the_copy(history):
    d = date(2005, 3, 1)  # only service tax is sourced this early
    stored = regime_for("equity", d)
    dated, unsourced = ch.with_dated_charges(stored, d, history)
    assert unsourced == ["stamp_duty_rate", "exchange_txn_charge_rate", "sebi_turnover_fee_rate"]
    assert dated.stamp_duty_rate == stored.stamp_duty_rate
    assert dated.exchange_txn_charge_rate == stored.exchange_txn_charge_rate
    assert dated.sebi_turnover_fee_rate == stored.sebi_turnover_fee_rate
    assert dated.gst_rate == pytest.approx(0.102)
    assert dated.stt_buy_rate == stored.stt_buy_rate == 0.00075


def test_with_dated_charges_before_any_coverage_returns_the_regime_unchanged(history):
    d = date(2004, 9, 1)
    stored = regime_params("IN_EQUITY_LISTED")
    dated, unsourced = ch.with_dated_charges(stored, d, history)
    assert dated == stored
    assert unsourced == list(_FOUR)


def test_with_dated_charges_rejects_non_equity(history):
    with pytest.raises(ValueError):
        ch.with_dated_charges(regime_params("IN_CRYPTO_VDA"), date(2025, 1, 1), history)


# ---------------------------------------------------------------- the loader

def test_loader_accepts_a_valid_file(tmp_path):
    h = ch.load_charge_history(_write(tmp_path, [{}]))
    assert len(h) == 1 and h.iloc[0]["effective_to"] is None


@pytest.mark.parametrize(
    "rows, fragment",
    [
        ([{"effective_to": "2020-12-31"}, {"effective_from": "2020-12-31"}], "overlapping"),
        ([{}, {"effective_from": "2022-01-01"}], "overlapping"),  # open row, then another
        ([{"side": "both"}, {"side": "buy", "effective_from": "2020-06-01"}], "overlapping"),
        ([{"source_url": ""}], "source_url"),
        ([{"source_url": "   "}], "source_url"),
        ([{"effective_from": "01/01/2020"}], "ISO date"),
        ([{"effective_from": "2020-02-30"}], "ISO date"),
        ([{"effective_to": "soon"}], "ISO date"),
        ([{"effective_from": ""}], "effective_from is empty"),
        ([{"effective_from": "2021-01-01", "effective_to": "2020-12-31"}], "after effective_to"),
        ([{"rate": "1"}], "outside"),
        ([{"rate": "-0.1"}], "outside"),
        ([{"rate": "abc"}], "not a number"),
        ([{"charge": "brokerage"}], "unknown charge"),
        ([{"side": "maker"}], "side"),
        ([{"basis": ""}], "basis"),
    ],
)
def test_loader_rejects_malformed_files(tmp_path, rows, fragment):
    with pytest.raises(ValueError, match=fragment):
        ch.load_charge_history(_write(tmp_path, rows))


def test_loader_allows_adjacent_rows_and_separate_sides(tmp_path):
    h = ch.load_charge_history(
        _write(
            tmp_path,
            [
                {"effective_to": "2020-12-31"},
                {"effective_from": "2021-01-01"},
                {"charge": "stamp_duty", "side": "buy"},
                {"charge": "stamp_duty", "side": "sell", "rate": "0"},
            ],
        )
    )
    assert len(h) == 4


def test_loader_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("charge,rate\nstamp_duty,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        ch.load_charge_history(path)


# ---------------------------------------------------------------- docs table comes from code

def test_decisions_pending_table_is_the_scripts_output():
    from scripts.charges_cost_table import render_table

    text = (DOCS / "DECISIONS_PENDING.md").read_text(encoding="utf-8")
    for line in render_table().splitlines():
        assert line in text, line
