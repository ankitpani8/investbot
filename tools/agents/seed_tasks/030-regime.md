---
id: 030-regime
depends: []
---
# regime.py: Hurst, Lyapunov, regime label and confidence

## Requirements
- New module `investbot/scoring/technical/regime.py`, pure compute, dict-return pattern. Every metric is a **rolling** estimate over past data only.
- `HURST_<W>`: rolling Hurst exponent on log returns, using R/S analysis or DFA (pick one and justify it in the docstring). Default windows are 100 and 250 bars.
- `LYAP_<W>`: rolling largest Lyapunov exponent (Rosenstein method). Embedding dimension and lag are parameters; document the defaults.
- Combine with ADX_14 and realised volatility to produce:
  - `REGIME`: categorical 0 = mean-reverting/range, 1 = trending, 2 = chaotic/high-volatility. Document the thresholds as named constants.
  - `REGIME_CONFIDENCE` in [0, 1]. This feeds the Black-Litterman ↔ risk-parity blend later, so it should be smooth, not a step function.
- Sanity fixtures: a random walk should give Hurst ≈ 0.5 ± tolerance; a strongly trending synthetic series > 0.6; an AR(1) mean-reverting series < 0.45; the logistic map at r = 4 should have a positive Lyapunov exponent; a sine wave ≈ 0.
- Performance: it must run on 1250 daily bars for 50 assets in under about a minute on a laptop. Vectorise or use numba only if needed. If you add numba, pin it in requirements.

## Acceptance criteria
- The sanity-fixture tests pass, and truncation invariance holds on all columns.
- No dropna() and the suite is green.
