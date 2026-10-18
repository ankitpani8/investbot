from dataclasses import fields
from datetime import date

import pytest

from investbot.config.tax_regimes import TAX_REGIMES
from investbot.trading import tax_rules
from investbot.trading.tax_rules import (
    TaxRegimeParams,
    days_to_long_term,
    first_long_term_date,
    fiscal_year,
    is_long_term,
    regime_params,
)

EQ = regime_params("IN_EQUITY_LISTED")
CRYPTO = regime_params("IN_CRYPTO_VDA")


def test_regime_params_mirror_config():
    for entry in TAX_REGIMES:
        p = regime_params(entry["regime_name"])
        assert {f.name for f in fields(TaxRegimeParams)} == set(entry)
        for key, val in entry.items():
            if key not in ("effective_from", "effective_to"):
                assert getattr(p, key) == val
    assert EQ.effective_from == date(2024, 7, 23)
    assert EQ.effective_to is None
    assert CRYPTO.effective_from == date(2022, 4, 1)


def test_regime_params_frozen_and_unknown():
    with pytest.raises(Exception):
        EQ.stcg_rate = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        regime_params("NOPE")


@pytest.mark.parametrize(
    "d,label",
    [
        (date(2026, 3, 31), "2025-26"),
        (date(2026, 4, 1), "2026-27"),
        (date(2026, 10, 18), "2026-27"),
        (date(2026, 1, 1), "2025-26"),
        (date(1999, 12, 31), "1999-00"),
        (date(2099, 3, 1), "2098-99"),
    ],
)
def test_fiscal_year(d, label):
    assert fiscal_year(d) == label


def test_anniversary_boundaries():
    buy = date(2025, 3, 15)
    assert first_long_term_date(buy, EQ) == date(2026, 3, 16)
    assert not is_long_term(buy, date(2026, 3, 14), EQ)  # day before anniversary
    assert not is_long_term(buy, date(2026, 3, 15), EQ)  # the anniversary itself
    assert is_long_term(buy, date(2026, 3, 16), EQ)  # the day after
    assert is_long_term(buy, date(2030, 1, 1), EQ)
    assert not is_long_term(buy, buy, EQ)


def test_calendar_months_not_365_days_across_leap_day():
    # 2023-03-15 -> 2024-03-15 spans 29 Feb 2024: 366 days, anniversary is still 15 Mar.
    buy = date(2023, 3, 15)
    assert (date(2024, 3, 15) - buy).days == 366
    assert not is_long_term(buy, date(2024, 3, 15), EQ)
    assert is_long_term(buy, date(2024, 3, 16), EQ)
    # 2024-03-15 -> 2025-03-15 is 365 days; a plain 365-day count would flip a day early
    buy = date(2024, 3, 15)
    assert (date(2025, 3, 15) - buy).days == 365
    assert not is_long_term(buy, date(2025, 3, 15), EQ)
    # a buy before 29 Feb: 2023-02-28 -> anniversary 2024-02-28 (a 365-day count would say 2024-02-28 too)
    assert first_long_term_date(date(2023, 2, 28), EQ) == date(2024, 2, 29)


def test_feb_29_buy_anniversary_is_feb_28():
    buy = date(2024, 2, 29)
    assert first_long_term_date(buy, EQ) == date(2025, 3, 1)
    assert not is_long_term(buy, date(2025, 2, 28), EQ)  # the anniversary
    assert not is_long_term(buy, date(2025, 2, 27), EQ)
    assert is_long_term(buy, date(2025, 3, 1), EQ)


def test_sale_on_anniversary_switch(monkeypatch):
    monkeypatch.setattr(tax_rules, "LTCG_STRICTLY_AFTER_ANNIVERSARY", False)
    buy = date(2025, 3, 15)
    assert first_long_term_date(buy, EQ) == date(2026, 3, 15)
    assert is_long_term(buy, date(2026, 3, 15), EQ)
    assert not is_long_term(buy, date(2026, 3, 14), EQ)


def test_month_end_clamp():
    # 31 Jan has an anniversary every year; 30 Apr likewise
    assert first_long_term_date(date(2025, 1, 31), EQ) == date(2026, 2, 1)
    assert first_long_term_date(date(2025, 12, 31), EQ) == date(2027, 1, 1)


def test_multi_year_holding_days_use_whole_years():
    p = TaxRegimeParams(**{**EQ.__dict__, "ltcg_holding_days": 730})
    assert first_long_term_date(date(2025, 3, 15), p) == date(2027, 3, 16)


def test_crypto_never_long_term():
    buy = date(2020, 1, 1)
    assert first_long_term_date(buy, CRYPTO) is None
    assert not is_long_term(buy, date(2030, 1, 1), CRYPTO)
    assert days_to_long_term(buy, date(2026, 1, 1), CRYPTO) == 0


def test_days_to_long_term():
    buy = date(2025, 3, 15)
    assert days_to_long_term(buy, date(2026, 3, 14), EQ) == 2
    assert days_to_long_term(buy, date(2026, 3, 15), EQ) == 1
    assert days_to_long_term(buy, date(2026, 3, 16), EQ) == 0
    assert days_to_long_term(buy, date(2027, 1, 1), EQ) == 0
    assert days_to_long_term(buy, buy, EQ) == 366


def test_regime_for_picks_regime_by_date():
    from investbot.trading.tax_rules import regime_for

    old = regime_for("equity", date(2024, 7, 22))
    new = regime_for("equity", date(2024, 7, 23))
    assert (old.regime_name, old.stcg_rate, old.ltcg_rate, old.ltcg_annual_exemption) == (
        "IN_EQUITY_LISTED_PRE_2024_07_23", 0.15, 0.10, 100_000)
    assert (new.regime_name, new.stcg_rate, new.ltcg_rate, new.ltcg_annual_exemption) == (
        "IN_EQUITY_LISTED", 0.20, 0.125, 125_000)
    assert regime_for("equity", date(2018, 4, 1)).regime_name == "IN_EQUITY_LISTED_PRE_2024_07_23"
    assert regime_for("equity", date(2030, 1, 1)).regime_name == "IN_EQUITY_LISTED"
    assert regime_for("crypto", date(2022, 4, 1)).regime_name == "IN_CRYPTO_VDA"


def test_regime_for_raises_when_no_row_covers_date():
    from investbot.trading.tax_rules import regime_for

    with pytest.raises(ValueError):
        regime_for("equity", date(2018, 3, 31))
    with pytest.raises(ValueError):
        regime_for("crypto", date(2022, 3, 31))
    with pytest.raises(ValueError):
        regime_for("bonds", date(2025, 1, 1))


def test_pre_2024_regime_matches_current_except_rates_exemption_dates():
    pre = regime_params("IN_EQUITY_LISTED_PRE_2024_07_23")
    diff = {f.name for f in fields(TaxRegimeParams) if getattr(pre, f.name) != getattr(EQ, f.name)}
    assert diff == {"regime_name", "stcg_rate", "ltcg_rate", "ltcg_annual_exemption",
                    "effective_from", "effective_to"}
    assert pre.effective_to == date(2024, 7, 22)
