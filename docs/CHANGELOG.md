# Changelog

- 2026-10-17: Migrated Phase 1 and the Phase 2 indicator layer from Google Drive to WSL and git. Added CLAUDE.md, contract tests and the backlog. Fixed the DB path (`.env` was pointing SQLite inside Google Drive).
- 2026-10-17: Fixed the daily smoke-test failure. 50 NULL-close bars on 2026-10-16 (plus 108 zero-volume placeholder bars) poisoned the indicators. Ingestion now validates every price row (`investbot/data/quality.py`), added `python main.py dq [--purge]`, purged 158 bad rows, and added NaN-propagation and ingestion tests.
