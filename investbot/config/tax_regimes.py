"""
Indian tax regime parameters for seeding the tax_regimes table.

Dated regimes: several rows may share an asset class as long as their
``effective_from``/``effective_to`` ranges (both inclusive) do not overlap;
``investbot.trading.tax_rules.regime_for`` picks the row in force on a date.
Sources: Finance (No. 2) Act 2024 (rates from 23 July 2024), Finance Act 2025; for the
2004-10-01 to 2018-03-31 rows see the comment on each row.
"""

TAX_REGIMES = [
    {
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED",
        "stcg_rate": 0.20,              # Section 111A — 20%
        "ltcg_rate": 0.125,             # Section 112A — 12.5%
        "ltcg_holding_days": 365,       # 12 months for listed equity
        "ltcg_annual_exemption": 125000, # ₹1.25 lakh
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "ltcg_only",
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,            # India has no wash sale rule
        "tds_rate": None,
        "stt_buy_rate": 0.001,          # 0.1% delivery
        "stt_sell_rate": 0.001,
        "stamp_duty_rate": 0.00015,     # 0.015% buy side
        "exchange_txn_charge_rate": 0.0000325,
        "sebi_turnover_fee_rate": 0.000001,
        "gst_rate": 0.18,
        "effective_from": "2024-07-23",  # Finance (No. 2) Act 2024
        "effective_to": None,
    },
    {
        # Rates before the Finance (No. 2) Act 2024 changed them on 23 July 2024.
        # All fields other than the rates, the exemption and the dates match the row above.
        # Before 2018-04-01 listed-equity LTCG was exempt (s.10(38)); see the IN_EQUITY_LISTED_2004_10_01
        # to IN_EQUITY_LISTED_2012_07_01 rows (2004-10-01 to 2018-03-31) below.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_PRE_2024_07_23",
        "stcg_rate": 0.15,              # Section 111A — 15%
        "ltcg_rate": 0.10,              # Section 112A — 10%
        "ltcg_holding_days": 365,
        "ltcg_annual_exemption": 100000, # ₹1 lakh
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "ltcg_only",
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.001,
        "stt_sell_rate": 0.001,
        "stamp_duty_rate": 0.00015,
        "exchange_txn_charge_rate": 0.0000325,
        "sebi_turnover_fee_rate": 0.000001,
        "gst_rate": 0.18,
        "effective_from": "2018-04-01",
        "effective_to": "2024-07-22",
    },
    {
        # STT 0.075% each side from 1 Oct 2004 (Finance (No. 2) Act 2004, Ch. VII s.98 table); STCG 10% s.111A (same Act).
        # LTCG on STT-paid listed equity exempt under s.10(38) (Finance (No. 2) Act 2004) until 31 Mar 2018.
        # Stamp duty, exchange charge, SEBI fee and GST rates are the current values
        # (not verified for this period); service tax before 2017-07-01 was not 18%.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_2004_10_01",
        "stcg_rate": 0.10,              # Section 111A
        "ltcg_rate": 0.0,               # s.10(38): exempt
        "ltcg_holding_days": 365,       # 12 months
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "none",         # exempt LTCG: a long-term loss cannot be set off or carried forward
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.00075,
        "stt_sell_rate": 0.00075,
        "stamp_duty_rate": 0.00015,     # not verified for this period
        "exchange_txn_charge_rate": 0.0000325,  # not verified for this period
        "sebi_turnover_fee_rate": 0.000001,     # not verified for this period
        "gst_rate": 0.18,               # not verified for this period
        "effective_from": "2004-10-01",
        "effective_to": "2005-05-31",
    },
    {
        # STT 0.1% each side from 1 Jun 2005 (Finance Act 2005); STCG 10% s.111A.
        # LTCG on STT-paid listed equity exempt under s.10(38) (Finance (No. 2) Act 2004) until 31 Mar 2018.
        # Stamp duty, exchange charge, SEBI fee and GST rates are the current values
        # (not verified for this period); service tax before 2017-07-01 was not 18%.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_2005_06_01",
        "stcg_rate": 0.10,              # Section 111A
        "ltcg_rate": 0.0,               # s.10(38): exempt
        "ltcg_holding_days": 365,       # 12 months
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "none",         # exempt LTCG: a long-term loss cannot be set off or carried forward
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.001,
        "stt_sell_rate": 0.001,
        "stamp_duty_rate": 0.00015,     # not verified for this period
        "exchange_txn_charge_rate": 0.0000325,  # not verified for this period
        "sebi_turnover_fee_rate": 0.000001,     # not verified for this period
        "gst_rate": 0.18,               # not verified for this period
        "effective_from": "2005-06-01",
        "effective_to": "2006-05-31",
    },
    {
        # STT 0.125% each side from 1 Jun 2006 (Finance Act 2006); STCG 10% s.111A.
        # LTCG on STT-paid listed equity exempt under s.10(38) (Finance (No. 2) Act 2004) until 31 Mar 2018.
        # Stamp duty, exchange charge, SEBI fee and GST rates are the current values
        # (not verified for this period); service tax before 2017-07-01 was not 18%.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_2006_06_01",
        "stcg_rate": 0.10,              # Section 111A
        "ltcg_rate": 0.0,               # s.10(38): exempt
        "ltcg_holding_days": 365,       # 12 months
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "none",         # exempt LTCG: a long-term loss cannot be set off or carried forward
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.00125,
        "stt_sell_rate": 0.00125,
        "stamp_duty_rate": 0.00015,     # not verified for this period
        "exchange_txn_charge_rate": 0.0000325,  # not verified for this period
        "sebi_turnover_fee_rate": 0.000001,     # not verified for this period
        "gst_rate": 0.18,               # not verified for this period
        "effective_from": "2006-06-01",
        "effective_to": "2008-03-31",
    },
    {
        # STCG 15% s.111A from 1 Apr 2008 (Finance Act 2008); STT unchanged at 0.125%.
        # LTCG on STT-paid listed equity exempt under s.10(38) (Finance (No. 2) Act 2004) until 31 Mar 2018.
        # Stamp duty, exchange charge, SEBI fee and GST rates are the current values
        # (not verified for this period); service tax before 2017-07-01 was not 18%.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_2008_04_01",
        "stcg_rate": 0.15,              # Section 111A
        "ltcg_rate": 0.0,               # s.10(38): exempt
        "ltcg_holding_days": 365,       # 12 months
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "none",         # exempt LTCG: a long-term loss cannot be set off or carried forward
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.00125,
        "stt_sell_rate": 0.00125,
        "stamp_duty_rate": 0.00015,     # not verified for this period
        "exchange_txn_charge_rate": 0.0000325,  # not verified for this period
        "sebi_turnover_fee_rate": 0.000001,     # not verified for this period
        "gst_rate": 0.18,               # not verified for this period
        "effective_from": "2008-04-01",
        "effective_to": "2012-06-30",
    },
    {
        # STT 0.1% each side from 1 Jul 2012 (Finance Act 2012 s.153, CBDT Circular 3/2012; SEBI Bulletin Mar 2012). The task text expected 2013-06-01; the sources say 2012-07-01.
        # LTCG on STT-paid listed equity exempt under s.10(38) (Finance (No. 2) Act 2004) until 31 Mar 2018.
        # Stamp duty, exchange charge, SEBI fee and GST rates are the current values
        # (not verified for this period); service tax before 2017-07-01 was not 18%.
        "country": "IN",
        "asset_class_name": "equity",
        "regime_name": "IN_EQUITY_LISTED_2012_07_01",
        "stcg_rate": 0.15,              # Section 111A
        "ltcg_rate": 0.0,               # s.10(38): exempt
        "ltcg_holding_days": 365,       # 12 months
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": None,
        "loss_offset_allowed": True,
        "stcl_offsets": "stcg_and_ltcg",
        "ltcl_offsets": "none",         # exempt LTCG: a long-term loss cannot be set off or carried forward
        "loss_carryforward_years": 8,
        "wash_sale_days": 0,
        "tds_rate": None,
        "stt_buy_rate": 0.001,
        "stt_sell_rate": 0.001,
        "stamp_duty_rate": 0.00015,     # not verified for this period
        "exchange_txn_charge_rate": 0.0000325,  # not verified for this period
        "sebi_turnover_fee_rate": 0.000001,     # not verified for this period
        "gst_rate": 0.18,               # not verified for this period
        "effective_from": "2012-07-01",
        "effective_to": "2018-03-31",
    },
    {
        "country": "IN",
        "asset_class_name": "crypto",
        "regime_name": "IN_CRYPTO_VDA",
        "stcg_rate": None,
        "ltcg_rate": None,
        "ltcg_holding_days": None,      # No STCG/LTCG distinction
        "ltcg_annual_exemption": 0,
        "flat_tax_rate": 0.30,          # Section 115BBH — 30% flat
        "loss_offset_allowed": False,   # No loss offset at all
        "stcl_offsets": "none",
        "ltcl_offsets": "none",
        "loss_carryforward_years": 0,
        "wash_sale_days": 0,
        "tds_rate": 0.01,              # 1% TDS — Section 194S
        "stt_buy_rate": None,
        "stt_sell_rate": None,
        "stamp_duty_rate": None,
        "exchange_txn_charge_rate": None,
        "sebi_turnover_fee_rate": None,
        "gst_rate": 0.18,
        "effective_from": "2022-04-01",
        "effective_to": None,
    },
]
