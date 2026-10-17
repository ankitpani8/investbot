---
id: 050-confluence
depends: [010-candle-patterns, 020-support-resistance, 030-regime, 040-multi-timeframe]
model: opus
---
# confluence.py: group-normalised rule-based technical score

## Requirements
- New module `investbot/scoring/technical/confluence.py`, pure compute.
- Map every indicator column to a family: trend, momentum, volatility, volume, pattern, S/R or MTF. Keep the mapping as an explicit dict in the module, with a test that every column produced by the upstream modules is mapped or explicitly ignored.
- Per family, convert each indicator to a directional signal in [-1, +1] with documented rules (e.g. RSI → piecewise around 30/50/70; price vs EMA; MACD histogram sign and slope; SUPERTd; S/R distance; candle net score). Average **within** the family first. This is the group normalisation that handles correlated indicators.
- Family weights depend on regime: a trending regime upweights trend and MTF, a range regime upweights momentum mean-reversion and S/R. Weights are a dict per regime, blended by REGIME_CONFIDENCE.
- Outputs: `TECH_SCORE` in [-1, 1]; `TECH_SCORE_<FAMILY>` per family; and `TECH_REASONS`, a short list of the top contributing signals (for the dashboard and explanations).
- Weights are config, not magic numbers: put them in `investbot/config/scoring_weights.py`. Initial values are hand-set, with a comment that the ML layer replaces them after backtesting.

## Acceptance criteria
- Tests: a monotonic uptrend fixture gives a clearly positive score, a downtrend a negative one, and flat noise a value near 0. The score is bounded; truncation invariance holds; the family mapping is complete; the suite is green.
