---
id: 400-eval-full-history
depends: []
model: opus
---
# Re-run the harness and regime study on full history

Read "Current priorities" in CLAUDE.md first.

## Requirements
- Check the scratch DB first: `SELECT MIN(date) FROM price_daily`, the number of index members with no price rows, and the `corporate_actions` row count. If the earliest date is after 2012-01-01 **or** `corporate_actions` is empty, the human ingest hasn't run. Stop with `RESULT: BLOCKED needs human ingest (python main.py ingest --full && python main.py ingest --members --full)`. Don't evaluate the 5-year data again.
- Otherwise run `python main.py evaluate` and the regime study with their default pre-registered settings. Change no signals, thresholds or polarities.
- Commit both reports. In docs/BACKLOG.md under "Evaluation: results", add a short **5-year vs full-history comparison table**: TECH_SCORE, the family scores, the 4 baselines and SIG_RVOL_CONFIRM at h = 1/5/20. Columns: mean IC, NW t, BH q, held-portfolio net CAGR vs universe. Also record the OOS period length, and note in one line each the survivorship (membership before 2019-03-29 falls back to today's list) and the dividend coverage.
- One plain-language paragraph in docs/CHANGELOG.md: did anything become significant, or stop being significant, with 4–8× more data?

## Acceptance criteria
- Both reports are committed and the comparison table is in BACKLOG.
- No code changes beyond what's needed to run on the larger dataset. Memory stays within 8 GB: process assets one at a time, as the panel builder already does.
- The suite is green.
