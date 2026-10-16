# Backlog

Status: `[ ]` open · `[~]` in progress (put the agent branch name next to it) · `[x]` done

## Housekeeping
- [ ] Replace TATAMOTORS.NS (delisted on Yahoo after the 2025 demerger) in `investbot/config/tickers.py`. Check the current NIFTY50 constituents and update the list.
- [ ] Remove the duplicate `investbot/scoring/technical/smoke_indicators.py` (the canonical copy is in `scripts/`).
- [ ] Update `docs/investbot_master_plan_v3.md` to v4: TA-Lib instead of pandas-ta, institutional flow signals, the cross-asset-class allocation layer, the 21-day LODR lag.

## Phase 2: technical scoring
- [x] Indicator layer (TA-Lib core + extensions): 112 columns, smoke-tested
- [ ] `candle_patterns.py`
- [ ] `support_resistance.py` (swing pivots, volume profile, Fibonacci)
- [ ] `regime.py` (Hurst, Lyapunov, regime_confidence)
- [ ] `multi_timeframe.py`
- [ ] `confluence.py`
- [ ] `scorer.py`: `score(ticker)`

## Later phases
See `CLAUDE.md` → Build order and the master plan.
