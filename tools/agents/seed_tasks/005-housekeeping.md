---
id: 005-housekeeping
depends: []
---
# Housekeeping: tickers, duplicate smoke script, pandas deprecation

## Requirements
- TATAMOTORS.NS is delisted on Yahoo after the 2025 Tata Motors demerger. In `investbot/config/tickers.py`, replace it with the correct current Yahoo symbols for the resulting listed entities **that are NIFTY50 constituents today**. Verify with one or two yfinance lookups; no loops. If you can't verify constituency, add both successor symbols, mark them `is_active=True`, and record the uncertainty in docs/DECISIONS_PENDING.md. Update the seed so existing DBs get the new assets and the old one is set `is_active=False`; don't delete history.
- Delete the duplicate `investbot/scoring/technical/smoke_indicators.py`. `scripts/smoke_indicators.py` is the canonical copy.
- Fix the pandas FutureWarning in `_extensions.py` (`pct_change()` default `fill_method`). Use `fill_method=None`, and make sure no column values change on gap-free data (add a test).
- Add a `filterwarnings = error::FutureWarning` line to pytest.ini so new deprecations fail CI.

## Acceptance criteria
- Tests are green with FutureWarnings promoted to errors.
- `python main.py status` shows the new tickers on the scratch DB after `python main.py setup` (the seed is idempotent).
