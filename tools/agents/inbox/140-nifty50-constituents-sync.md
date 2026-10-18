---
id: 140-nifty50-constituents-sync
depends: []
---
# Sync NIFTY50_TICKERS with the current index

## Requirements
- WIPRO.NS appears twice in `investbot/config/tickers.py`, and NSE's 2026-09-15 reconstitution removed WIPRO from NIFTY50. Fetch the current constituent list once (niftyindices.com "ind_nifty50list.csv", or NSE's index page; at most a handful of requests, no loops) and reconcile: remove the duplicate, move names no longer in the index to `RETIRED_TICKERS` (deactivated, history kept), and add any missing constituents with their Yahoo `.NS` symbol, name, sector and industry.
- If the source can't be reached, make only the change you can verify (the duplicate WIPRO row), and put the rest in docs/DECISIONS_PENDING.md with the exact list to check.
- Record the as-of date of the constituent list in a comment at the top of the ticker list.
- Add a BACKLOG item: "historical NIFTY50 constituents for survivorship-free evaluation" (needed by the evaluation harness).

## Acceptance criteria
- A test that the ticker list has no duplicate symbols and that every retired ticker is absent from the active list.
- The seed stays idempotent (run `python main.py setup` twice on the scratch DB). The suite is green.
