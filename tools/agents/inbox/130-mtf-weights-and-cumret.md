---
id: 130-mtf-weights-and-cumret
depends: []
---
# MTF timeframe weights 1/2/3, and NaN-restart cumulative returns

Two small decided changes (2026-10-18, docs/DECISIONS_PENDING.md) in separate files.

## Requirements
- `investbot/scoring/technical/multi_timeframe.py`: `TIMEFRAME_WEIGHTS = (1.0, 2.0, 3.0)` for (15m, 1h, 1d). Update the docstring and any test that hard-codes equal weights.
- `investbot/scoring/technical/_extensions.py`: `CUMLOGRET_1` and `CUMPCTRET_1` are NaN on any bar whose return is NaN, and the cumulation restarts from 0 on the next valid bar (one segment per gap-free run). Gap-free data must be unchanged; add a test proving it.
- Bump `SCORER_VERSION` in `scorer.py`.

## Acceptance criteria
- Tests: weighted MTF alignment on a fixture where only the daily trend disagrees gives the sign of the daily trend; cumulative-return gap behaviour; unchanged output on gap-free data; truncation invariance still holds.
- The suite is green.
