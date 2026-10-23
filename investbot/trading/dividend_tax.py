"""
Dated tax on dividend income from domestic companies, for a resident individual (pure).

One fiscal year's gross dividends -> that FY's dividend tax. No DB, no network.

The rule by FY (April-March)
    * Up to FY 2019-20 the company paid dividend distribution tax (s.115-O) and the dividend
      was exempt in the shareholder's hands under s.10(34).
    * FY 2016-17 to FY 2019-20: s.115BBDA, inserted by the Finance Act 2016 with effect from
      AY 2017-18 (i.e. FY 2016-17), taxes a resident individual, HUF or firm at 10% on the
      aggregate dividends from domestic companies **above ₹10 lakh** in the FY, with no
      deduction of expenses. The Finance Act 2020 limited it to dividends declared,
      distributed or paid on or before 31 March 2020.
      Sources: incometaxindia.gov.in, "Section 115BBDA"; ICAI branch notes on the Finance
      Bill 2020; dsij.in "Taxability of dividend above Rs10 lakh".
    * From FY 2020-21: the Finance Act 2020 abolished DDT on dividends distributed on or
      after 1 April 2020 and withdrew the s.10(34) exemption from AY 2021-22. The whole
      dividend is taxed with the shareholder's other income at the slab rate.
      Sources: as above; conventuslaw.com "India: dividend distribution tax abolished"
      (18 February 2020).

``dividend_tax_for_fy`` returns ``base x (1 + surcharge) x (1 + cess)``, where the base is 0
(before FY 2016-17), ``S115BBDA_RATE x max(0, gross - S115BBDA_THRESHOLD)`` (FY 2016-17 to
2019-20) or ``slab_rate x gross`` (FY 2020-21 on). The cess defaults to
``tax_ledger.cess_rate_for(fiscal_year)``, as for capital gains.

``DIVIDEND_SLAB_RATE = 0.30`` is the top slab of the old regime, before surcharge and cess:
the conservative default for an investor who files at the highest rate
(docs/DECISIONS_PENDING.md, "Dividend tax: slab default and FY rule").

Not modelled: the s.57 cap on the interest deduction against dividend income (20% of the
dividend, from FY 2020-21); TDS under s.194 (only a prepayment of this same tax, not an
extra cost); the basic exemption limit, the slab structure below the chosen rate and the
s.87A rebate; non-resident rates (s.115A / treaties); the DDT the company paid before
FY 2020-21 (the gross dividend received is already net of it).
"""

from __future__ import annotations

from investbot.trading.tax_ledger import _fy_start_year, cess_rate_for

DIVIDEND_SLAB_RATE: float = 0.30  # top slab, old regime, before surcharge and cess

S115BBDA_THRESHOLD = 1_000_000.0  # ₹10 lakh of aggregate domestic-company dividends per FY
S115BBDA_RATE = 0.10
S115BBDA_FIRST_FY = 2016  # FY 2016-17 (AY 2017-18), Finance Act 2016
SLAB_FIRST_FY = 2020  # FY 2020-21, Finance Act 2020: DDT gone, s.10(34) withdrawn

# Scope of s.115BBDA: resident individuals, HUFs and firms from AY 2017-18. The Finance
# Act 2017 is reported to have widened it to most residents from AY 2018-19: not verified,
# and irrelevant here (the simulated investor is an individual).
# A cap on the surcharge on dividend income (reported as 15%): not verified. The caller's
# ``surcharge_rate`` is applied as given.


def _check_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def dividend_tax_for_fy(
    fiscal_year: str,
    gross_dividends: float,
    *,
    slab_rate: float = DIVIDEND_SLAB_RATE,
    surcharge_rate: float = 0.0,
    cess_rate: float | None = None,
) -> float:
    """Tax on ``gross_dividends`` (₹, domestic companies) received in ``fiscal_year`` ('2021-22').

    See the module docstring for the rule. ``cess_rate=None`` uses ``cess_rate_for``.
    Raises ValueError for a bad FY label, negative dividends or a rate outside [0, 1].
    """
    start = _fy_start_year(fiscal_year)
    if not gross_dividends >= 0.0:  # also rejects NaN
        raise ValueError(f"gross_dividends must be >= 0, got {gross_dividends}")
    if cess_rate is None:
        cess_rate = cess_rate_for(fiscal_year)
    for name, value in (("slab_rate", slab_rate), ("surcharge_rate", surcharge_rate),
                        ("cess_rate", cess_rate)):
        _check_rate(name, value)
    if start < S115BBDA_FIRST_FY:
        base = 0.0
    elif start < SLAB_FIRST_FY:
        base = S115BBDA_RATE * max(0.0, gross_dividends - S115BBDA_THRESHOLD)
    else:
        base = slab_rate * gross_dividends
    return base * (1.0 + surcharge_rate) * (1.0 + cess_rate)
