# InvestBot (Medallion 2.0)

An automated portfolio allocation and rebalancing system for Indian retail investors.
It answers one continuous question: *given the capital available right now, where should it be?*
It starts with NIFTY50 on NSE, paper trading only. The data model is asset-class agnostic from day one.

Ankit is the project manager and domain expert. He decides when there are real options.
You are the lead developer. When a design choice is open and the plan doesn't settle it,
write the options in `docs/DECISIONS_PENDING.md` and move to other work. Don't guess on
anything that changes the architecture.

## Source of truth

- `docs/investbot_master_plan_v3.md` is the canonical plan for schema, scoring, the control
  loop, tax handling and backtesting KPIs.
- **This file wins over the plan where they conflict.** The plan predates these decisions:
  - **TA-Lib replaced pandas-ta** (pandas-ta is abandoned and has lookahead bias in DPO and
    Ichimoku). Wherever the plan says pandas-ta, read TA-Lib plus our `_extensions.py`.
    The future ML pipeline also uses TA-Lib features, not pandas-ta.
  - Institutional flow: FII and DII net flow are **always combined, never used alone**.
    The combined signal activates only after 3 consecutive days with z-score > 1.5.
    Bulk/block deals are confirmation-only. Shareholding change is a slow independent signal.
    SEBI LODR shareholding filings lag **21 days**, not 45.
  - A **top-level cross-asset-class allocation layer** sits above the per-asset-class
    control loops. It runs daily or when the regime changes.

## Layout

```
main.py                         CLI: setup | ingest | run | status | gaps | dq | score | evaluate | constituents
investbot/config/               settings (.env), tickers, tax regimes
investbot/data/                 SQLAlchemy models, db session, seed, ingestion (yfinance)
investbot/scheduler/            APScheduler jobs
investbot/scoring/technical/    indicators.py (public API), _talib_core.py, _extensions.py
investbot/scoring/fundamental/  stub, returns 100% pass-through for MVP
investbot/scoring/sentiment/    stub (FinBERT later)
investbot/evaluation/           walk-forward forward-return harness (python main.py evaluate)
scripts/smoke_indicators.py     end-to-end indicator smoke test against the real DB
tests/                          pytest, no network or DB needed
docs/                           master plan and decision logs
```

The database is at `~/investbot_data/investbot.db`, outside the repo. Never commit `*.db` or `.env`.
**The GitHub repo is public.** Treat every commit as published.

## Commands (run from repo root with `.venv` active)

```bash
source .venv/bin/activate
python -m pytest -q                  # must pass before every commit
python -m scripts.smoke_indicators   # needs an ingested DB
python main.py status                # DB row counts
python main.py ingest                # incremental ingest, hits yfinance (--full: full history; --members: past index members)
```

## Build order (Phase 2 and on)

Each step goes in its own module under `investbot/scoring/technical/` with tests:

1. `candle_patterns.py`: TA-Lib CDL* functions, normalised to -1/0/+1 plus a strength score
2. `support_resistance.py`: swing pivots, volume profile (deferred here from indicators), Fibonacci
3. `regime.py`: Hurst exponent, largest Lyapunov exponent, trend/range/chaotic labels with a regime_confidence
4. `multi_timeframe.py`: 15m / 1h / daily alignment
5. `confluence.py`: group-normalised within indicator families (correlation is handled here, not at the indicator layer)
6. `scorer.py`: `score(ticker)` returns technical score, regime confidence and reasons, and owns caching and DB I/O

Then: fundamental scoring, sentiment (FinBERT, `ProsusAI/finbert`, never bag-of-words),
the allocation layer (Black-Litterman ↔ risk parity blended by regime_confidence, PID execution),
tax layer, backtesting, Streamlit dashboard.
Open tasks are tracked in `docs/BACKLOG.md`. If it doesn't exist yet, create it from this list.

## Current priorities (updated 2026-10-20; the planner must follow these)

The harness works, and on the data we have, nothing beats a trivial baseline after costs (eval_2026101x/20,
regime_study_*). The bottleneck is now **data, not methodology**: 5 years of history, 20 past index members
with no prices, and an empty `corporate_actions` table (so total-return labels equal price-only labels).
Only a human ingest can fix that, since agents must not bulk-download.

1. **Human step (Ankit):** `python main.py ingest --full && python main.py ingest --members --full && python main.py dq`
   (DAILY_PERIOD is now "max"). Until it has run, don't schedule tasks that need more history.
2. **Task 400:** re-run the harness and the regime study on the full history; write what changed vs the 5-year runs.
3. **No more inference-method work on the current dataset.** Robust IC, bootstrap and BH are done.
   More statistics on the same 5 years won't change the answer.
4. **Allocation work is frozen** (Black-Litterman, PID, 370/380 parked). With no signal that survives costs,
   there are no views to allocate on. It unfreezes when a signal passes the harness out of sample, net of costs.
5. **Direction decision pending (Ankit):** move the decision horizon to monthly/weekly and research slow factors
   (momentum, low-vol, point-in-time quality/value) instead of intraday technical scoring. See docs/DECISIONS_PENDING.md.
   Don't schedule work that presumes either answer.
6. **Fundamentals are paused** until they are **point-in-time**: a filing's numbers count only from its publication
   date, or conservatively from period end + 45 days (quarterly) / + 60 days (annual), per SEBI LODR Reg. 33.
   yfinance statements carry period-end dates, so using them as of period end is lookahead. Balance-sheet and
   cash-flow rules also wait for Ankit's notes from *How to Read Financial Statements*. Don't invent them.

## Non-negotiable engineering rules

- **No lookahead bias.** A value at bar *t* may only use data up to *t*. Every new
  indicator, pattern or feature gets a truncation-invariance test (see `tests/test_indicators.py`).
  Ichimoku's Chikou Span stays out.
- **Never call `dropna()` inside compute functions.** Warmup NaNs are kept. Each consumer picks its own cutoff.
- **Compute modules are pure.** They take a DataFrame and return dicts or DataFrames. No DB I/O. Only `scorer.py` touches the DB.
- **Dict-return pattern.** Collect columns in `dict[str, np.ndarray]` and assemble with one `pd.concat`. No repeated single-column assignment.
- **Column names are a contract.** Don't rename existing indicator columns. Downstream code depends on them.
- **Schema-first.** Add complete columns and tables up front even when the logic is stubbed. Exposing them in the UI is a separate decision.
- **Tax is a first-class constraint.** STCG 20%, LTCG 12.5% above ₹1.25L per year, crypto 30% flat with no loss offset.
  The 365-day cliff is modelled. India has no wash-sale rule.
- One correct long-term choice beats optionality. Don't add fallback libraries "just in case".
- Paper trading only. Nothing in this repo places real orders or touches a broker account.

## Working as an autonomous agent

- Work on a branch named `agent/<short-task>`, never directly on `main`. Keep commits small with clear messages.
- Before committing, run `python -m pytest -q`. A red test suite is never committed.
- Don't run `python main.py ingest` or anything else that hammers yfinance in a loop. Use existing DB data or synthetic fixtures.
- Only add dependencies to `requirements.txt` when a module needs them, pinned with a lower bound.
- When you finish a task, update `docs/BACKLOG.md` (mark it done, add follow-ups) and append a
  one-line entry to `docs/CHANGELOG.md`.
- The machine has 8 GB of RAM. Avoid loading all 54 assets' intraday history into memory at once.
