"""
Indian tax regime parameters for seeding the tax_regimes table.
Source: Finance Act 2025 (Budget 2025-26 rates).
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
        "effective_from": "2025-04-01",
        "effective_to": None,
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
