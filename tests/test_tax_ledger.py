import json
from datetime import date

import pytest

from investbot.data.models import UserTaxLedger
from investbot.trading.lot_tracker import Realisation
from investbot.trading.tax_ledger import (
    CESS_RATE,
    CarryForward,
    compute_fiscal_year_tax,
    incremental_tax,
)

FY = "2025-26"
D = date(2025, 8, 1)  # inside FY 2025-26, post-2024-07-23 rates (STCG 20%, LTCG 12.5%)
NO_CESS = {"cess_rate": 0.0}


def R(gain, term, sell_date=D):
    return Realisation(
        lot_id="L", asset_id=1, buy_date=date(2020, 1, 1), sell_date=sell_date,
        quantity=1, cost_basis=0.0, sale_value=gain, sell_costs=0.0, gain=gain, term=term,
    )


def tax(rs, fy=FY, cf=None, **kw):
    return compute_fiscal_year_tax(rs, fy, cf, **{**NO_CESS, **kw})


def test_default_cess_constant():
    assert CESS_RATE == 0.04
    assert compute_fiscal_year_tax([R(100_000, "STCG")], FY).estimated_tax_liability == pytest.approx(20_800)


def test_empty_year():
    t = compute_fiscal_year_tax([], FY)
    assert t.estimated_tax_liability == 0
    assert t.ltcg_exemption_remaining == 125_000 and t.ltcg_exemption_used == 0
    assert t.carry_forward_out == CarryForward()


def test_pure_stcg():
    t = tax([R(100_000, "STCG"), R(50_000, "STCG")])
    assert t.realized_stcg == 150_000
    assert t.estimated_tax_liability == pytest.approx(30_000)


def test_ltcg_below_exemption():
    t = tax([R(100_000, "LTCG")])
    assert t.estimated_tax_liability == 0
    assert (t.ltcg_exemption_used, t.ltcg_exemption_remaining) == (100_000, 25_000)


def test_ltcg_above_exemption():
    t = tax([R(225_000, "LTCG")])
    assert t.realized_ltcg == 225_000
    assert t.ltcg_exemption_used == 125_000 and t.ltcg_exemption_remaining == 0
    assert t.estimated_tax_liability == pytest.approx(12_500)  # 100k x 12.5%


def test_stcl_offsets_stcg_before_ltcg():
    # STCG 50k absorbs 50k of the 80k STCL, the other 30k goes against LTCG 200k.
    t = tax([R(50_000, "STCG"), R(200_000, "LTCG"), R(-80_000, "STCG")])
    assert t.realized_stcl == 80_000
    assert t.ltcg_exemption_used == 125_000
    assert t.estimated_tax_liability == pytest.approx(45_000 * 0.125)
    assert t.carry_forward_out == CarryForward()


def test_ltcl_never_offsets_stcg():
    t = tax([R(100_000, "STCG"), R(-50_000, "LTCG")])
    assert t.estimated_tax_liability == pytest.approx(20_000)
    assert t.carry_forward_out.ltcl_by_fy == {FY: 50_000}
    assert t.carried_forward_ltcl == 50_000


def test_ltcl_offsets_ltcg_before_exemption():
    t = tax([R(200_000, "LTCG"), R(-100_000, "LTCG")])
    assert t.ltcg_exemption_used == 100_000 and t.ltcg_exemption_remaining == 25_000
    assert t.estimated_tax_liability == 0


def test_brought_forward_oldest_first():
    cf = CarryForward(stcl_by_fy={"2022-23": 30_000, "2023-24": 50_000})
    t = tax([R(40_000, "STCG")], cf=cf)
    assert t.estimated_tax_liability == 0
    assert t.carry_forward_out.stcl_by_fy == {"2023-24": 40_000}
    assert t.cf_stcl_by_year == {"2023-24": 40_000}


def test_brought_forward_ltcl_only_against_ltcg():
    cf = CarryForward(ltcl_by_fy={"2023-24": 100_000})
    t = tax([R(50_000, "STCG"), R(200_000, "LTCG")], cf=cf)
    # STCG untouched (10k tax); LTCG 200k - 100k BF = 100k, inside the exemption.
    assert t.estimated_tax_liability == pytest.approx(10_000)
    assert t.carry_forward_out == CarryForward()
    t = tax([R(50_000, "STCG")], cf=cf)
    assert t.estimated_tax_liability == pytest.approx(10_000)
    assert t.carry_forward_out.ltcl_by_fy == {"2023-24": 100_000}


def test_brought_forward_stcl_offsets_ltcg():
    cf = CarryForward(stcl_by_fy={"2023-24": 200_000})
    t = tax([R(300_000, "LTCG")], cf=cf)
    assert t.estimated_tax_liability == 0  # 100k left, inside the exemption
    assert t.ltcg_exemption_used == 100_000


def test_current_year_losses_before_brought_forward():
    cf = CarryForward(stcl_by_fy={"2023-24": 100_000})
    t = tax([R(60_000, "STCG"), R(-60_000, "STCG")], cf=cf)
    assert t.carry_forward_out.stcl_by_fy == {"2023-24": 100_000}


def test_expiry_after_eight_years():
    cf = CarryForward(stcl_by_fy={"2016-17": 100_000}, ltcl_by_fy={"2016-17": 100_000})
    # FY 2024-25 is the 8th year after 2016-17: still usable.
    t = tax([R(100_000, "STCG", date(2025, 1, 10))], "2024-25", cf)
    assert t.estimated_tax_liability == 0
    assert t.carry_forward_out.stcl_by_fy == {}
    assert t.carry_forward_out.ltcl_by_fy == {"2016-17": 100_000}
    # FY 2025-26 is the 9th: expired, dropped from the carry-forward too.
    t = tax([R(100_000, "STCG")], FY, cf)
    assert t.estimated_tax_liability == pytest.approx(20_000)
    assert t.carry_forward_out == CarryForward()
    # 2017-18 is still alive in 2025-26.
    t = tax([R(100_000, "STCG")], FY, CarryForward(stcl_by_fy={"2017-18": 100_000}))
    assert t.estimated_tax_liability == 0


def test_carry_forward_out_and_ledger_row():
    cf = CarryForward(stcl_by_fy={"2023-24": 10_000}, ltcl_by_fy={"2024-25": 5_000})
    t = tax([R(-70_000, "STCG"), R(-20_000, "LTCG")], cf=cf)
    assert t.carry_forward_out.stcl_by_fy == {"2023-24": 10_000, FY: 70_000}
    assert t.carry_forward_out.ltcl_by_fy == {"2024-25": 5_000, FY: 20_000}
    assert (t.carried_forward_stcl, t.carried_forward_ltcl) == (80_000, 25_000)
    row = t.to_ledger_row()
    assert json.loads(row["cf_stcl_by_year"]) == {"2023-24": 10_000, FY: 70_000}
    assert json.loads(row["cf_ltcl_by_year"]) == {"2024-25": 5_000, FY: 20_000}
    assert set(row) - {"fiscal_year"} <= {c.name for c in UserTaxLedger.__table__.columns}
    assert row["fiscal_year"] == FY


def test_crypto_loss_does_not_reduce_crypto_gain():
    t = tax([R(100_000, "FLAT"), R(-40_000, "FLAT")], crypto_tds_paid=1_000)
    assert t.crypto_realized_gains == 100_000
    assert t.estimated_tax_liability == pytest.approx(30_000)
    assert t.crypto_tds_paid == 1_000  # reported, not netted
    assert t.carry_forward_out == CarryForward()


def test_equity_loss_does_not_reduce_crypto_gain():
    t = tax([R(100_000, "FLAT"), R(-50_000, "STCG"), R(-50_000, "LTCG")])
    assert t.estimated_tax_liability == pytest.approx(30_000)
    assert t.carry_forward_out == CarryForward(
        stcl_by_fy={FY: 50_000}, ltcl_by_fy={FY: 50_000}
    )


def test_crypto_gain_does_not_absorb_equity_losses_or_carry_forward():
    cf = CarryForward(stcl_by_fy={"2023-24": 10_000})
    t = tax([R(100_000, "FLAT")], cf=cf)
    assert t.carry_forward_out.stcl_by_fy == {"2023-24": 10_000}


def test_surcharge_and_cess():
    t = compute_fiscal_year_tax([R(100_000, "STCG")], FY, surcharge_rate=0.10)
    assert t.estimated_tax_liability == pytest.approx(20_000 * 1.10 * 1.04)
    t = compute_fiscal_year_tax([R(100_000, "FLAT")], FY)
    assert t.estimated_tax_liability == pytest.approx(30_000 * 1.04)


def test_fy_2024_25_split_regimes_stcg():
    t = tax([R(100_000, "STCG", date(2024, 7, 22)), R(100_000, "STCG", date(2024, 7, 23))], "2024-25")
    assert t.estimated_tax_liability == pytest.approx(100_000 * 0.15 + 100_000 * 0.20)


def test_fy_2024_25_split_regimes_ltcg_exemption_higher_rate_first():
    rs = [R(200_000, "LTCG", date(2024, 7, 22)), R(100_000, "LTCG", date(2024, 7, 23))]
    t = tax(rs, "2024-25")
    # Exemption is the 31 March 2025 value (1.25L), spent on the 12.5% bucket first:
    # 100k at 12.5% gone, 25k of the 10% bucket gone, 175k left at 10%.
    assert t.ltcg_exemption_used == 125_000
    assert t.estimated_tax_liability == pytest.approx(17_500)


def test_fy_2024_25_losses_go_to_higher_rate_bucket_first():
    rs = [
        R(100_000, "STCG", date(2024, 7, 22)),
        R(100_000, "STCG", date(2024, 7, 23)),
        R(-50_000, "STCG", date(2024, 4, 5)),
    ]
    t = tax(rs, "2024-25")
    assert t.estimated_tax_liability == pytest.approx(100_000 * 0.15 + 50_000 * 0.20)


def test_fy_2023_24_uses_one_lakh_exemption_and_old_rates():
    d = date(2023, 9, 1)
    t = tax([R(200_000, "LTCG", d), R(100_000, "STCG", d)], "2023-24")
    assert t.ltcg_exemption_used == 100_000 and t.ltcg_exemption_remaining == 0
    assert t.estimated_tax_liability == pytest.approx(100_000 * 0.10 + 100_000 * 0.15)


def test_sale_with_no_regime_raises():
    with pytest.raises(ValueError):
        tax([R(1_000, "STCG", date(2004, 9, 30))], "2004-05")  # before STT: no regime


def test_only_realisations_in_the_fiscal_year_count():
    inside = [R(100_000, "STCG", date(2025, 4, 1)), R(50_000, "STCG", date(2026, 3, 31))]
    outside = [R(900_000, "LTCG", date(2026, 4, 1)), R(-900_000, "STCG", date(2025, 3, 31))]
    assert tax(inside + outside) == tax(inside)
    assert tax(inside).estimated_tax_liability == pytest.approx(30_000)


def test_no_lookahead_future_realisation_never_affects_fy():
    base = [R(100_000, "STCG"), R(200_000, "LTCG")]
    future = [R(-1_000_000, "STCG", date(2027, 1, 1)), R(1_000_000, "FLAT", date(2026, 5, 1))]
    assert tax(base + future) == tax(base)
    assert incremental_tax(base, future, FY) == 0


def test_inputs_not_mutated():
    cf = CarryForward(stcl_by_fy={"2023-24": 10_000})
    tax([R(50_000, "STCG")], cf=cf)
    assert cf == CarryForward(stcl_by_fy={"2023-24": 10_000})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fiscal_year": "2025-27"},
        {"fiscal_year": "2025"},
        {"fiscal_year": FY, "carry_forward_in": CarryForward(stcl_by_fy={FY: 1.0})},
    ],
)
def test_bad_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        compute_fiscal_year_tax([], **kwargs)


def test_unknown_term_raises():
    with pytest.raises(ValueError):
        tax([R(1.0, "XYZ")])


def test_incremental_tax_zero_inside_remaining_exemption():
    assert incremental_tax([], [R(125_000, "LTCG")], FY, **NO_CESS) == 0
    assert incremental_tax([R(25_000, "LTCG")], [R(100_000, "LTCG")], FY) == 0
    # Beyond the remaining 25k: 75k taxed at 12.5%, plus cess.
    assert incremental_tax([R(100_000, "LTCG")], [R(100_000, "LTCG")], FY) == pytest.approx(
        75_000 * 0.125 * 1.04
    )


def test_incremental_tax_negative_for_loss_harvest():
    inc = incremental_tax([R(100_000, "STCG")], [R(-40_000, "STCG")], FY)
    assert inc == pytest.approx(-40_000 * 0.20 * 1.04)
    assert incremental_tax([R(100_000, "STCG")], [R(-40_000, "LTCG")], FY) == 0


def test_incremental_tax_uses_carry_forward_and_kwargs():
    cf = CarryForward(stcl_by_fy={"2024-25": 30_000})
    inc = incremental_tax([], [R(50_000, "STCG")], FY, cf, cess_rate=0.0)
    assert inc == pytest.approx(20_000 * 0.20)


# --- exempt LTCG (s.10(38)) and dated cess, FY 2004-05 to 2017-18 ---

def test_cess_rate_for_history():
    from investbot.trading.tax_ledger import cess_rate_for

    assert cess_rate_for("2003-04") == 0.0
    for fy in ("2004-05", "2005-06", "2006-07"):
        assert cess_rate_for(fy) == 0.02
    for fy in ("2007-08", "2010-11", "2017-18"):
        assert cess_rate_for(fy) == 0.03
    for fy in ("2018-19", "2025-26"):
        assert cess_rate_for(fy) == 0.04 == CESS_RATE
    with pytest.raises(ValueError):
        cess_rate_for("2010-2011")


def test_exempt_ltcg_not_taxed_stcg_taxed_with_dated_cess():
    from investbot.trading.tax_ledger import cess_rate_for

    d = date(2010, 8, 1)
    t = compute_fiscal_year_tax([R(500_000, "LTCG", d), R(100_000, "STCG", d)], "2010-11")
    assert (t.realized_ltcg, t.realized_stcg) == (500_000, 100_000)
    assert cess_rate_for("2010-11") == 0.03
    assert t.estimated_tax_liability == pytest.approx(100_000 * 0.15 * 1.03)
    assert (t.ltcg_exemption_used, t.ltcg_exemption_remaining) == (0, 0)
    assert t.carry_forward_out == CarryForward()


def test_explicit_cess_overrides_dated_cess():
    d = date(2010, 8, 1)
    t = compute_fiscal_year_tax([R(100_000, "STCG", d)], "2010-11", cess_rate=0.0)
    assert t.estimated_tax_liability == pytest.approx(15_000)


def test_exempt_ltcl_is_neither_offset_nor_carried_forward():
    d = date(2015, 9, 1)
    t = tax([R(-300_000, "LTCG", d), R(80_000, "STCG", d)], "2015-16")
    assert t.realized_ltcl == 300_000
    assert t.estimated_tax_liability == pytest.approx(80_000 * 0.15)  # LTCL did not reduce STCG
    assert t.carry_forward_out == CarryForward() and t.carried_forward_ltcl == 0


def test_stcl_not_set_off_against_exempt_ltcg_and_carried_forward():
    d = date(2012, 10, 1)
    t = tax([R(400_000, "LTCG", d), R(-60_000, "STCG", d)], "2012-13")
    assert (t.realized_ltcg, t.realized_stcl) == (400_000, 60_000)
    assert t.estimated_tax_liability == 0
    assert t.carry_forward_out.stcl_by_fy == {"2012-13": 60_000}  # nothing absorbed it


def test_stcl_carried_from_2016_17_usable_against_ltcg_in_2018_19():
    loss = tax([R(-90_000, "STCG", date(2016, 12, 1))], "2016-17")
    assert loss.carry_forward_out.stcl_by_fy == {"2016-17": 90_000}
    t = tax([R(300_000, "LTCG", date(2018, 8, 1))], "2018-19", loss.carry_forward_out)
    # LTCG 3L less BF STCL 90k = 2.1L; exemption 1L; 1.1L taxed at 10%.
    assert t.ltcg_exemption_used == 100_000
    assert t.estimated_tax_liability == pytest.approx(110_000 * 0.10)
    assert t.carry_forward_out == CarryForward()


def test_stcg_rate_changes_on_2008_04_01():
    before = tax([R(100_000, "STCG", date(2008, 3, 31))], "2007-08")
    after = tax([R(100_000, "STCG", date(2008, 4, 1))], "2008-09")
    assert before.estimated_tax_liability == pytest.approx(10_000)
    assert after.estimated_tax_liability == pytest.approx(15_000)


def test_exempt_loss_in_pre_2018_fy_uses_sell_date_regime_only():
    # FY 2017-18 sells are all exempt-era; the same gross in FY 2018-19 is taxable.
    old = tax([R(200_000, "LTCG", date(2018, 3, 31))], "2017-18")
    new = tax([R(200_000, "LTCG", date(2018, 4, 1))], "2018-19")
    assert old.estimated_tax_liability == 0
    assert new.estimated_tax_liability == pytest.approx(100_000 * 0.10)
