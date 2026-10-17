---
id: 020-support-resistance
depends: []
---
# support_resistance.py: pivots, volume profile, Fibonacci

## Requirements
- New module `investbot/scoring/technical/support_resistance.py`, pure compute, dict-return pattern.
- **Swing pivots without lookahead.** A pivot high at bar t is confirmed only at bar t+k (k = `PIVOT_RIGHT_BARS`, default 3). The value is emitted at the confirmation bar, never back-filled to t. Document this explicitly, because it's the classic lookahead trap.
- Rolling S/R levels from the last N confirmed pivots, clustered by price tolerance (`SR_CLUSTER_PCT`, default 0.5%) and weighted by touches and recency. Output: `SR_SUPPORT_1..3`, `SR_RESIST_1..3`, `SR_DIST_SUPPORT_PCT`, `SR_DIST_RESIST_PCT`, `SR_STRENGTH_SUPPORT` and `SR_STRENGTH_RESIST`.
- **Rolling volume profile** over `VP_WINDOW` bars (default 120 daily or 500 intraday, as a parameter), computed only from past bars. Output: `VP_POC`, `VP_VAH` and `VP_VAL` (70% value area), plus `VP_POS` (where the close sits within the value area, -1..+1).
- **Fibonacci retracements** of the most recent confirmed swing (low→high or high→low). Output: `FIB_236`, `FIB_382`, `FIB_500`, `FIB_618`, `FIB_786` and `FIB_NEAREST_DIST_PCT`.
- Use numpy loops or rolling windows that stay inside 8 GB of RAM for 1500 rows × 50 assets. Benchmark and note the timing in the docstring.

## Acceptance criteria
- A truncation-invariance test over **all** output columns at several cut points. This is the key test for this module.
- A synthetic zig-zag fixture with known pivots gives the expected levels.
- No dropna(), the input is not mutated, and the suite is green.
