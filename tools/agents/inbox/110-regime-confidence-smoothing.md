---
id: 110-regime-confidence-smoothing
depends: []
model: opus
---
# Smooth REGIME_CONFIDENCE with a causal 5-bar EMA

Decided 2026-10-18 (docs/DECISIONS_PENDING.md). The top-two-gap definition stays; smoothing cuts flip churn before it reaches the Black-Litterman ↔ risk-parity blend.

## Requirements
- In `investbot/scoring/technical/regime.py`: rename the current gap output to `REGIME_CONFIDENCE_RAW`. `REGIME_CONFIDENCE` = EMA(span=`CONFIDENCE_EMA_SPAN = 5`) of the raw value, causal (`adjust=False`), NaN until the raw series is valid. At a NaN in the middle of the raw series, restart the EMA after the gap rather than carrying stale state across it.
- Update `list_regime_columns`, the confluence column→family contract (RAW is ignored by confluence) and `multi_timeframe` (MTF_REGIME_CONFIDENCE_1D uses the smoothed value).
- Bump `SCORER_VERSION` in `scorer.py` so cached rows recompute.

## Acceptance criteria
- Tests: truncation invariance on the new columns; the EMA matches a hand-computed sequence; NaN-gap restart behaviour; on a synthetic series that flips regime every few bars, the smoothed confidence has fewer zero-crossings of (confidence − 0.1) than the raw one.
- The suite is green.
