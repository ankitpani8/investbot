"""
Dated dividend-income tax (investbot/trading/dividend_tax.py).
"""

import math

import pytest

from investbot.trading.dividend_tax import (
    DIVIDEND_SLAB_RATE,
    S115BBDA_RATE,
    S115BBDA_THRESHOLD,
    dividend_tax_for_fy,
)
from investbot.trading.tax_ledger import cess_rate_for


def test_constants():
    assert DIVIDEND_SLAB_RATE == 0.30
    assert S115BBDA_THRESHOLD == 1_000_000.0
    assert S115BBDA_RATE == 0.10


def test_before_fy_2016_17_exempt():
    assert dividend_tax_for_fy("2015-16", 5_000_000.0) == 0.0
    assert dividend_tax_for_fy("2004-05", 5_000_000.0) == 0.0


@pytest.mark.parametrize("fy", ["2016-17", "2019-20"])
def test_s115bbda_years(fy):
    # 3% cess in FY 2016-17 / 2017-18, 4% from FY 2018-19
    cess = 0.03 if fy == "2016-17" else 0.04
    assert dividend_tax_for_fy(fy, 999_999.0) == 0.0
    assert dividend_tax_for_fy(fy, S115BBDA_THRESHOLD) == 0.0
    assert dividend_tax_for_fy(fy, 1_500_000.0) == pytest.approx(0.10 * 500_000.0 * (1 + cess))
    # the slab rate is irrelevant in these years
    assert dividend_tax_for_fy(fy, 1_500_000.0, slab_rate=0.0) == pytest.approx(
        0.10 * 500_000.0 * (1 + cess)
    )


def test_slab_from_fy_2020_21():
    assert dividend_tax_for_fy("2020-21", 100.0) == pytest.approx(100.0 * 0.30 * 1.04)
    assert dividend_tax_for_fy("2025-26", 2_000_000.0, slab_rate=0.20) == pytest.approx(
        2_000_000.0 * 0.20 * 1.04
    )
    assert dividend_tax_for_fy("2021-22", 0.0) == 0.0


def test_cess_default_and_override():
    for fy in ("2016-17", "2018-19", "2022-23"):
        gross = 2_000_000.0
        base = dividend_tax_for_fy(fy, gross, cess_rate=0.0)
        assert dividend_tax_for_fy(fy, gross) == pytest.approx(base * (1 + cess_rate_for(fy)))
        assert dividend_tax_for_fy(fy, gross, cess_rate=0.10) == pytest.approx(base * 1.10)


def test_surcharge_then_cess():
    assert dividend_tax_for_fy("2023-24", 1_000.0, surcharge_rate=0.15) == pytest.approx(
        1_000.0 * 0.30 * 1.15 * 1.04
    )
    assert dividend_tax_for_fy("2017-18", 2_000_000.0, surcharge_rate=0.10, cess_rate=0.03) == (
        pytest.approx(0.10 * 1_000_000.0 * 1.10 * 1.03)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gross_dividends": -1.0},
        {"gross_dividends": math.nan},
        {"gross_dividends": 1.0, "slab_rate": 1.5},
        {"gross_dividends": 1.0, "slab_rate": -0.1},
        {"gross_dividends": 1.0, "surcharge_rate": -0.01},
        {"gross_dividends": 1.0, "cess_rate": 2.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        dividend_tax_for_fy("2021-22", **kwargs)


def test_bad_label_raises():
    with pytest.raises(ValueError):
        dividend_tax_for_fy("2021-23", 1.0)
