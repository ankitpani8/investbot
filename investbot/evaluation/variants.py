"""
Pre-registered confluence polarity variants, scored next to production TECH_SCORE.

Ankit decided (DECISIONS_PENDING → Decided, 2026-10-18, "Confluence polarity")
that the polarity alternatives are tested in the harness, not chosen by hand.
These are the alternatives from that decision, fixed before any results were
seen. **Do not add, tune or search over more variants here**: a new one is a
proposal for docs/DECISIONS_PENDING.md first (every extra column also enters
the BH family and raises every q-value).

    VAR_OSC_REVERT        alternative (a), full mean reversion: every signal in
                          ``OSCILLATOR_SIGNALS`` with ``osc_extreme = -1``, so a
                          pinned oscillator (RSI 100) is a full sell
    VAR_OSC_MOMENTUM      alternative (a), pure momentum: ``osc_extreme = +1``,
                          no fade (RSI 100 reads +1)
    VAR_REGIME_POLARITY   alternative (b), regime-dependent polarity: production
                          signals, but every signal in ``OSCILLATOR_SIGNALS`` is
                          negated on bars where ``REGIME == 0`` (range: revert);
                          trend, chaotic and NaN-regime bars keep the production
                          sign (follow momentum)

Each comes with ``<name>_MOMENTUM``, its momentum-family score (the family the
oscillators mostly sit in; BB_POS/KC_POS are volatility, MFI volume). Variants
are aggregated by ``confluence.aggregate_signals`` with the production weights,
regime blend and multiplier, so only the polarity differs.

Pure: takes the joined daily frame from ``scorer.compute_daily_frame`` (upstream
columns + confluence) and returns arrays. Only the signal and aggregation steps
re-run; no indicator is recomputed. Every signal is a function of bars <= t and
REGIME at t is lookahead-free, so every column is too (truncation-invariance
test in ``tests/test_evaluation_variants.py``). No dropna: warmup stays NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from investbot.scoring.technical.confluence import (
    OSCILLATOR_SIGNALS,
    aggregate_signals,
    compute_signals,
)
from investbot.scoring.technical.regime import REGIME_RANGE

VARIANT_PREFIX = "VAR_"
#: Variant name → osc_extreme for alternative (a).
OSC_EXTREME_VARIANTS: dict[str, float] = {"OSC_REVERT": -1.0, "OSC_MOMENTUM": 1.0}
REGIME_POLARITY = "REGIME_POLARITY"
VARIANT_NAMES: tuple[str, ...] = (*OSC_EXTREME_VARIANTS, REGIME_POLARITY)
#: Production columns each variant column is compared against (report).
VARIANT_COLUMNS: dict[str, str] = {
    **{f"{VARIANT_PREFIX}{v}": "TECH_SCORE" for v in VARIANT_NAMES},
    **{f"{VARIANT_PREFIX}{v}_MOMENTUM": "TECH_SCORE_MOMENTUM" for v in VARIANT_NAMES},
}


def list_variant_columns() -> list[str]:
    """Keys of ``compute_variants``, in output order."""
    return [c for v in VARIANT_NAMES for c in (f"{VARIANT_PREFIX}{v}", f"{VARIANT_PREFIX}{v}_MOMENTUM")]


def regime_polarity_signals(frame: pd.DataFrame,
                            signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``signals`` with every ``OSCILLATOR_SIGNALS`` entry negated on range bars
    (REGIME == REGIME_RANGE). NaN regime keeps the sign. Order kept, input not mutated."""
    n = len(frame)
    regime = (frame["REGIME"].to_numpy(dtype=np.float64) if "REGIME" in frame
              else np.full(n, np.nan))
    in_range = regime == REGIME_RANGE           # NaN compares False
    return {k: (np.where(in_range, -v, v) if k in OSCILLATOR_SIGNALS else v)
            for k, v in signals.items()}


def compute_variants(frame: pd.DataFrame, *,
                     signals: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """
    Every ``VAR_*`` column for one asset (keys as ``list_variant_columns()``).

    ``signals``: production ``compute_signals(frame)`` if the caller already has
    them (saves one signal pass); computed here otherwise.
    """
    out: dict[str, np.ndarray] = {}
    for name, extreme in OSC_EXTREME_VARIANTS.items():
        agg = aggregate_signals(frame, compute_signals(frame, osc_extreme=extreme))
        out[f"{VARIANT_PREFIX}{name}"] = agg["TECH_SCORE"]
        out[f"{VARIANT_PREFIX}{name}_MOMENTUM"] = agg["TECH_SCORE_MOMENTUM"]
    base = compute_signals(frame) if signals is None else signals
    agg = aggregate_signals(frame, regime_polarity_signals(frame, base))
    out[f"{VARIANT_PREFIX}{REGIME_POLARITY}"] = agg["TECH_SCORE"]
    out[f"{VARIANT_PREFIX}{REGIME_POLARITY}_MOMENTUM"] = agg["TECH_SCORE_MOMENTUM"]
    return out
