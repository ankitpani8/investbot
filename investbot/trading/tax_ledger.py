"""
Fiscal-year capital-gains tax ledger (plan §2.8 4f).

Pure compute: no DB, no network. Takes ``Realisation`` rows from ``lot_tracker`` and
returns one April-March year's gains, set-offs, exemption, tax and carry-forward.

Equity rules, applied in this order (Income-tax Act ss.70-74, 111A, 112A):

1. Current-year STCL is set off against STCG first, then LTCG. Current-year LTCL is set
   off against LTCG only.
2. Brought-forward losses are used oldest origin year first. BF STCL offsets STCG then
   LTCG; BF LTCL offsets LTCG only. A loss older than ``loss_carryforward_years`` (8)
   expires: a FY 2016-17 loss is usable up to and including FY 2024-25.
3. The annual LTCG exemption (the regime in force on the FY's last day, 31 March) applies
   to the LTCG left after set-off.
4. Each gain is taxed at the rate of the regime in force on its ``sell_date``.
5. tax = sum(bucket x rate), then x (1 + surcharge_rate) x (1 + cess_rate). The cess
   defaults to the rate of the FY (``cess_rate_for``: 2% / 3% / 4%).
6. Unabsorbed current-year losses join the carry-forward under this FY's key.

Two rate regimes in one FY (FY 2024-25, rates changed on 23 July 2024): gains form one
bucket per (term, rate). Losses and the exemption are applied to the **higher-rate
bucket first**, which minimises tax (taxpayer-optimal). See docs/DECISIONS_PENDING.md.

Exempt LTCG (s.10(38), sales from 2004-10-01 to 2018-03-31, ``tax_rules.ltcg_exempt`` of
the sell-date regime): an LTCG or LTCL realisation is counted in ``realized_ltcg`` /
``realized_ltcl`` but forms no bucket, absorbs no loss (current-year or brought-forward
STCL is never set off against it), uses no exemption, and an exempt LTCL is not carried
forward. STCG and STCL in those years behave as usual (STCL offsets STCG only).

Crypto (``term == "FLAT"``, Section 115BBH): the flat rate applies to the sum of the
*positive* gains only. Losses are ignored: never offset (not even against other crypto
gains) and never carried forward. TDS is reported, not netted against the tax.

Not modelled (BACKLOG): sales before 2004-10-01 (they raise ValueError: STCG at slab
rates and taxable LTCG with indexation), 31-Jan-2018 grandfathering,
basic-exemption-limit adjustment of resident individuals, the s.87A rebate, slab-rate
assets, the pre-2018 STT-not-paid case (LTCG taxable under s.112).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from investbot.trading import tax_rules
from investbot.trading.lot_tracker import Realisation

CESS_RATE = 0.04  # health and education cess on the tax (plus surcharge), from FY 2018-19

# First FY (start year) of each cess rate, oldest first. Sources: Finance (No. 2) Act 2004
# (education cess 2%, FY 2004-05), Finance Act 2007 (+1% secondary and higher education
# cess, 3% from FY 2007-08), Finance Act 2018 (4% health and education cess from FY 2018-19).
_CESS_HISTORY = ((2004, 0.02), (2007, 0.03), (2018, CESS_RATE))

_EPS = 1e-9


@dataclass
class CarryForward:
    """Unabsorbed losses as positive amounts keyed by the FY of origin ('2023-24')."""

    stcl_by_fy: dict[str, float] = field(default_factory=dict)
    ltcl_by_fy: dict[str, float] = field(default_factory=dict)


@dataclass
class FiscalYearTax:
    """One FY's result. Fields map 1:1 onto ``UserTaxLedger`` columns (see to_ledger_row).

    ``realized_*`` are gross, before any set-off or carry-forward (losses as positive
    amounts). ``estimated_tax_liability`` includes crypto, surcharge and cess.
    """

    fiscal_year: str
    realized_stcg: float
    realized_ltcg: float
    realized_stcl: float
    realized_ltcl: float
    ltcg_exemption_used: float
    ltcg_exemption_remaining: float
    estimated_tax_liability: float
    carried_forward_stcl: float
    carried_forward_ltcl: float
    cf_stcl_by_year: dict[str, float]
    cf_ltcl_by_year: dict[str, float]
    crypto_realized_gains: float
    crypto_tds_paid: float
    carry_forward_out: CarryForward

    def to_ledger_row(self) -> dict:
        """``UserTaxLedger`` column values; the two JSON columns are strings."""
        return {
            "fiscal_year": self.fiscal_year,
            "realized_stcg": self.realized_stcg,
            "realized_ltcg": self.realized_ltcg,
            "realized_stcl": self.realized_stcl,
            "realized_ltcl": self.realized_ltcl,
            "ltcg_exemption_used": self.ltcg_exemption_used,
            "ltcg_exemption_remaining": self.ltcg_exemption_remaining,
            "estimated_tax_liability": self.estimated_tax_liability,
            "carried_forward_stcl": self.carried_forward_stcl,
            "carried_forward_ltcl": self.carried_forward_ltcl,
            "cf_stcl_by_year": json.dumps(self.cf_stcl_by_year, sort_keys=True),
            "cf_ltcl_by_year": json.dumps(self.cf_ltcl_by_year, sort_keys=True),
            "crypto_realized_gains": self.crypto_realized_gains,
            "crypto_tds_paid": self.crypto_tds_paid,
        }


def _fy_start_year(label: str) -> int:
    try:
        start, end = label.split("-")
        if len(end) != 2 or (int(start) + 1) % 100 != int(end):
            raise ValueError
        return int(start)
    except ValueError:
        raise ValueError(f"bad fiscal year label {label!r}, expected e.g. '2024-25'") from None


def cess_rate_for(fiscal_year: str) -> float:
    """Education / health and education cess on income tax for ``fiscal_year`` ('2010-11').

    0.0 before FY 2004-05, when no cess existed.
    """
    start = _fy_start_year(fiscal_year)
    rate = 0.0
    for first_year, r in _CESS_HISTORY:
        if start >= first_year:
            rate = r
    return rate


def _fy_end(label: str) -> date:
    return date(_fy_start_year(label) + 1, 3, 31)


class _Bucket:
    """Gain taxed at one rate. ``amount`` shrinks as losses and the exemption absorb it."""

    __slots__ = ("rate", "amount")

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.amount = 0.0


def _absorb(loss: float, *bucket_groups: list[_Bucket]) -> float:
    """Set ``loss`` off against the groups in order, higher rate first within a group.

    Returns the unabsorbed remainder.
    """
    for group in bucket_groups:
        for bucket in sorted(group, key=lambda b: -b.rate):
            if loss <= _EPS:
                return 0.0
            used = min(loss, bucket.amount)
            bucket.amount -= used
            loss -= used
    return max(loss, 0.0)


def _clean(amount: float) -> float:
    return 0.0 if abs(amount) < _EPS else amount


def compute_fiscal_year_tax(
    realisations: Iterable[Realisation],
    fiscal_year: str,
    carry_forward_in: CarryForward | None = None,
    *,
    cess_rate: float | None = None,
    surcharge_rate: float = 0.0,
    crypto_tds_paid: float = 0.0,
) -> FiscalYearTax:
    """Tax for ``fiscal_year`` ('2024-25') from the realisations sold in it.

    ``cess_rate=None`` uses ``cess_rate_for(fiscal_year)``; an explicit float overrides it.
    Realisations with a ``sell_date`` outside the FY are ignored. ``carry_forward_in`` is
    what ``compute_fiscal_year_tax`` of the previous FY returned as ``carry_forward_out``;
    keys must be earlier FYs. Raises ValueError for a bad label, an unknown term, a
    carry-forward key that is not before ``fiscal_year``, or a sell date with no regime.
    """
    fy_start = _fy_start_year(fiscal_year)
    if cess_rate is None:
        cess_rate = cess_rate_for(fiscal_year)
    cf_in = carry_forward_in or CarryForward()
    for key in (*cf_in.stcl_by_fy, *cf_in.ltcl_by_fy):
        if _fy_start_year(key) >= fy_start:
            raise ValueError(f"carry-forward from {key} cannot be used in {fiscal_year}")

    stcg: dict[float, _Bucket] = {}
    ltcg: dict[float, _Bucket] = {}
    gross = {"STCG": 0.0, "LTCG": 0.0, "STCL": 0.0, "LTCL": 0.0}
    exempt = {"LTCG": 0.0, "LTCL": 0.0}  # exempt-era long-term amounts: reported, never used
    crypto_gain = 0.0
    crypto_tax = 0.0
    regime_cache: dict[tuple[str, date], tax_rules.TaxRegimeParams] = {}

    def regime(asset_class: str, on: date) -> tax_rules.TaxRegimeParams:
        if (asset_class, on) not in regime_cache:
            regime_cache[(asset_class, on)] = tax_rules.regime_for(asset_class, on)
        return regime_cache[(asset_class, on)]

    for r in realisations:
        if tax_rules.fiscal_year(r.sell_date) != fiscal_year:
            continue
        if r.term == "FLAT":
            if r.gain > 0:  # s.115BBH: losses are ignored entirely
                crypto_gain += r.gain
                crypto_tax += r.gain * regime("crypto", r.sell_date).flat_tax_rate
        elif r.term in ("STCG", "LTCG"):
            reg = regime("equity", r.sell_date)
            if r.term == "LTCG" and tax_rules.ltcg_exempt(reg):
                # s.10(38): counted gross, never taxed, never offset, never carried forward.
                exempt["LTCG" if r.gain > 0 else "LTCL"] += abs(r.gain)
            elif r.gain < 0:
                gross["STCL" if r.term == "STCG" else "LTCL"] += -r.gain
            elif r.gain > 0:
                rate = reg.stcg_rate if r.term == "STCG" else reg.ltcg_rate
                buckets = stcg if r.term == "STCG" else ltcg
                buckets.setdefault(rate, _Bucket(rate)).amount += r.gain
                gross[r.term] += r.gain
        else:
            raise ValueError(f"unknown realisation term {r.term!r}")

    st_buckets, lt_buckets = list(stcg.values()), list(ltcg.values())

    # 1. Current-year losses.
    stcl_left = _absorb(gross["STCL"], st_buckets, lt_buckets)
    ltcl_left = _absorb(gross["LTCL"], lt_buckets)

    # 2. Brought-forward losses, oldest origin FY first; expired ones are dropped.
    fy_end_regime = regime("equity", _fy_end(fiscal_year))
    max_age = fy_end_regime.loss_carryforward_years
    out = CarryForward()
    for origin in sorted(set(cf_in.stcl_by_fy) | set(cf_in.ltcl_by_fy), key=_fy_start_year):
        if fy_start - _fy_start_year(origin) > max_age:
            continue
        left = _absorb(cf_in.stcl_by_fy.get(origin, 0.0), st_buckets, lt_buckets)
        if left > _EPS:
            out.stcl_by_fy[origin] = left
        left = _absorb(cf_in.ltcl_by_fy.get(origin, 0.0), lt_buckets)
        if left > _EPS:
            out.ltcl_by_fy[origin] = left

    # 6. Unabsorbed current-year losses carry forward under this FY.
    if stcl_left > _EPS:
        out.stcl_by_fy[fiscal_year] = stcl_left
    if ltcl_left > _EPS:
        out.ltcl_by_fy[fiscal_year] = ltcl_left

    # 3. Exemption on the net LTCG, using the regime in force on 31 March.
    net_ltcg = sum(b.amount for b in lt_buckets)
    exemption = fy_end_regime.ltcg_annual_exemption
    exempt_used = min(exemption, net_ltcg)
    _absorb(exempt_used, lt_buckets)

    # 4-5. Tax at each bucket's own rate, then surcharge and cess.
    base_tax = sum(b.amount * b.rate for b in st_buckets + lt_buckets) + crypto_tax
    total = base_tax * (1 + surcharge_rate) * (1 + cess_rate)

    cf_stcl = {k: _clean(v) for k, v in sorted(out.stcl_by_fy.items(), key=lambda kv: _fy_start_year(kv[0]))}
    cf_ltcl = {k: _clean(v) for k, v in sorted(out.ltcl_by_fy.items(), key=lambda kv: _fy_start_year(kv[0]))}
    return FiscalYearTax(
        fiscal_year=fiscal_year,
        realized_stcg=gross["STCG"],
        realized_ltcg=gross["LTCG"] + exempt["LTCG"],
        realized_stcl=gross["STCL"],
        realized_ltcl=gross["LTCL"] + exempt["LTCL"],
        ltcg_exemption_used=exempt_used,
        ltcg_exemption_remaining=exemption - exempt_used,
        estimated_tax_liability=total,
        carried_forward_stcl=sum(cf_stcl.values()),
        carried_forward_ltcl=sum(cf_ltcl.values()),
        cf_stcl_by_year=dict(cf_stcl),
        cf_ltcl_by_year=dict(cf_ltcl),
        crypto_realized_gains=crypto_gain,
        crypto_tds_paid=crypto_tds_paid,
        carry_forward_out=CarryForward(cf_stcl, cf_ltcl),
    )


def incremental_tax(
    ytd_realisations: Iterable[Realisation],
    candidate_realisations: Iterable[Realisation],
    fiscal_year: str,
    carry_forward_in: CarryForward | None = None,
    **kw,
) -> float:
    """Tax cost of the candidate sale(s) given the year so far.

    ``tax(ytd + candidate) - tax(ytd)``. Negative when the candidate harvests a loss
    against gains already realised. A candidate sold outside ``fiscal_year`` has no
    effect on it. ``kw`` goes to ``compute_fiscal_year_tax``.
    """
    ytd = list(ytd_realisations)
    with_candidate = compute_fiscal_year_tax(
        ytd + list(candidate_realisations), fiscal_year, carry_forward_in, **kw
    )
    base = compute_fiscal_year_tax(ytd, fiscal_year, carry_forward_in, **kw)
    return with_candidate.estimated_tax_liability - base.estimated_tax_liability
