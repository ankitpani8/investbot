---
id: 010-candle-patterns
depends: []
---
# candle_patterns.py: TA-Lib candlestick patterns

## Requirements
- New module `investbot/scoring/technical/candle_patterns.py`, pure compute (no DB), using the dict-return pattern.
- `compute_candle_patterns(df) -> dict[str, np.ndarray]` runs **all** TA-Lib `CDL*` functions (`talib.get_function_groups()["Pattern Recognition"]`).
- Output columns:
  - `CDL_<NAME>` holds the raw TA-Lib value (-200..200).
  - `CDLN_<NAME>` holds it normalised to -1/0/+1.
  - Aggregates: `CDL_BULL_COUNT`, `CDL_BEAR_COUNT` and `CDL_NET_SCORE`. The net score is a weighted sum where confirmed patterns (±200) weigh 1.0 and plain signals (±100) weigh 0.5, clipped to [-1, 1] after dividing by a documented constant.
- `add_candle_patterns(df) -> pd.DataFrame` does a single `pd.concat`.
- Module docstring: TA-Lib patterns only look at the current and past bars. State which patterns need the most lookback.
- Don't wire it into `indicators.add_all_indicators` yet. `scorer.py` will compose the modules later, and the indicator column contract stays unchanged.

## Acceptance criteria
- Tests on synthetic OHLCV: column count equals 2 × number of CDL functions + 3; values are in their allowed sets; truncation invariance (no lookahead) at several cut points; input is not mutated; row count is preserved.
- A hand-built engulfing-pattern fixture triggers `CDLN_ENGULFING == +1`.
- The suite is green.
