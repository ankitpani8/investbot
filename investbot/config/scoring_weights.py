"""
Family weights for the rule-based technical score (``confluence.py``).

All values here are HAND-SET starting points, not fitted. The ML layer
replaces them with weights learned from backtests (per regime, possibly per
asset class) once the backtesting layer exists. Until then, change them here,
never as literals inside the scoring code.

How they are used
    For each bar, with regime label r (from ``regime.py``) and confidence c:
        w = c * FAMILY_WEIGHTS[r] + (1 - c) * FAMILY_WEIGHTS["neutral"]
    then renormalised over the families that have a value on that bar (e.g.
    MTF is missing on daily-only scoring). Before the regime has warmed up
    (REGIME is NaN) the neutral weights are used. The final score is
    multiplied by the same blend of ``REGIME_SCORE_MULTIPLIER``.
"""
from __future__ import annotations

#: Indicator families, in output order (TECH_SCORE_<FAMILY.upper()>).
FAMILIES: tuple[str, ...] = ("trend", "momentum", "volatility", "volume", "pattern", "sr", "mtf")

#: Per-regime family weights. Each row sums to 1. Keys of the outer dict are
#: "neutral" plus the regime names in ``REGIME_NAMES``.
FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    # No regime information (warmup) or zero confidence: a balanced view.
    "neutral": {
        "trend": 0.25, "momentum": 0.20, "volatility": 0.05, "volume": 0.15,
        "pattern": 0.05, "sr": 0.15, "mtf": 0.15,
    },
    # Trending: follow the trend and the higher timeframes; S/R levels get
    # broken in trends, so they count for little.
    "trend": {
        "trend": 0.35, "momentum": 0.15, "volatility": 0.05, "volume": 0.10,
        "pattern": 0.05, "sr": 0.05, "mtf": 0.25,
    },
    # Range-bound: oscillator exhaustion and position between support and
    # resistance carry the signal; trend-following signals whipsaw.
    "range": {
        "trend": 0.10, "momentum": 0.30, "volatility": 0.10, "volume": 0.10,
        "pattern": 0.10, "sr": 0.25, "mtf": 0.05,
    },
    # Chaotic / high volatility: lean on volume (participation) and S/R, and
    # shrink the whole score via REGIME_SCORE_MULTIPLIER.
    "chaotic": {
        "trend": 0.20, "momentum": 0.15, "volatility": 0.15, "volume": 0.20,
        "pattern": 0.05, "sr": 0.15, "mtf": 0.10,
    },
}

#: Regime code (``regime.REGIME_RANGE/TREND/CHAOS``) → key in FAMILY_WEIGHTS.
REGIME_NAMES: dict[int, str] = {0: "range", 1: "trend", 2: "chaotic"}

#: Multiplier on TECH_SCORE per regime, blended like the weights. Values <= 1
#: keep the score in [-1, 1]. Chaotic markets halve conviction (conservative,
#: see docs/DECISIONS_PENDING.md).
REGIME_SCORE_MULTIPLIER: dict[str, float] = {
    "neutral": 1.0,
    "trend": 1.0,
    "range": 1.0,
    "chaotic": 0.5,
}
