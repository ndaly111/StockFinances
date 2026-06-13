# Phase 1 — DB Persistence via `data` Branch: Verified Runbook
**Date:** 2026-06-13 · **Status:** Foundation shipped; irreversible flip staged & verified, pending multi-day proof.
Design adversarially verified this session (verdict: **SAFE_WITH_CHANGES**; corrections folded in below).

## Goal
Get the 22.7 MB `Stock Data.db` out of `main`'s daily commits (the cause of the 9 GiB repo / 3m43s checkouts) by making an orphan `data` branch the DB's single source of truth. This also defuses the actions.yml landmine: `actions.yml` is currently the *only* thing persisting the pipeline's DB writes (the daily `Implied_Growth_History`/`Forward_EPS_FY_History` dots + weekly `TTM_Data`/`Annual_Data`) to `main`; the refresh workflows push only to gh-pages. Kill it without a replacement and the ticker history charts freeze.

## DONE (this session — safe, additive, on the live repo)
- **`data` branch created** (orphan, contains only `Stock Data.db`; seed blob `38f0ceec`).
- **Trial persist steps** added to `site-refresh-daily.yml` (evening cron + dispatch) and `site-refresh.yml` (weekly): clone `-b data`, copy the freshly-built DB, push `HEAD:data`. `continue-on-error: true` + a **`<10MB` size guard** (rejects an empty/truncated DB — sqlite auto-creates empty files). **Purely additive:** actions.yml still commits the DB to main and every consumer still reads main, so this cannot affect the live site. The `data` branch just starts accumulating real snapshots.

## Verification findings that corrected the original design
1. **The "hydrate everywhere in Stage A" idea was a landmine.** Five **dual consumer-writers** (`daily_index_data`, `update-assumptions`, `monthly_spy_backfill`, `Spyhistorical`, `run_index_history`) still push the DB to *main*. If you add hydrate-from-`data` to them while they still push to main, they round-trip a stale `data` blob onto main and **drop the daily dots** → the exact chart freeze. Fix: for those five, hydrate **and** convert push-to-data in the **same atomic commit** (Stage C), never split.
2. Use **`git restore --source=origin/data --worktree -- "Stock Data.db"`**, NOT `git checkout origin/data -- …`. `restore` writes the file but **never stages it** (so a later `git add .` can't re-commit it to main) and works even when the DB is gitignored/untracked (it reads the blob from the ref, bypassing the index). Add a `test -s "Stock Data.db" || exit 1` guard after every hydrate (empty-DB → history-wipe protection).
3. **gh-pages re-bloat isn't fixed by one `git rm` alone.** The rsync blocks don't exclude the DB, and `render_earnings_to_dashboard.yml` (`cp -r main_repo/* gh-pages/` daily) + legacy `cp -r` publishers re-add the hydrated 22 MB DB every day. Land `--exclude "Stock Data.db"` in both rsyncs + an `rm -f gh-pages/'Stock Data.db'` guard on the `cp -r` publishers **together with** the one-time gh-pages `git rm`.
4. **Dead scripts to drop from the plan:** `QQQ_Pe.yml`→`QQQ_PE.py` and `QQQPE.yml`→`QQQPE.py` (scripts don't exist), `Update_DB.yml`→`DB_Update.py` (NameError before any DB access). `screen-microcaps` does **not** read the canonical DB (uses a throwaway temp DB) — no hydrate needed.
5. **Concurrency:** `daily_index_data` and the 4th daily refresh share the `30 21 UTC` cron and push to `data` concurrently. They write **disjoint tables**, so the safe mechanism is a **clone→modify→push, on non-fast-forward re-clone-and-re-run-the-idempotent-writer, retry ×5** loop (a lock-free compare-and-swap; correct because every writer is `INSERT OR REPLACE/IGNORE`). Optionally stagger the index cron to `35 21` (flag to Nick; not mandatory with the retry loop). Do **not** merge the two concurrency groups.

## Staged execution (each stage independently reversible)
- **STAGE A — hydrate pure consumers + gh-pages excludes.** One PR. Add the `git restore --source=origin/data` hydrate (with `test -s` guard) to the **pure-consumer** workflows only (refresh daily/weekly, `render_earnings`, `segment_charts`, `db_structure`, `cleanup`, `run_stock_split`, etc. — NOT the 5 dual writers, NOT actions.yml). Add `--exclude "Stock Data.db"` to both rsyncs + the `cp -r` guards. No-op while DB still on main → fully reversible by revert.
- **STAGE B — promote trial persist to real.** Remove `continue-on-error` + the daily conditional; wrap each persist in the retry loop (writer = `main_remote.py`). Run several cycles; confirm `git log origin/data` advances daily+weekly. Reversible.
- **STAGE C — THE IRREVERSIBLE FLIP (one atomic commit).** `git rm --cached "Stock Data.db"` + `.gitignore` entry + convert the 5 dual writers from push-to-main to persist-to-data (with hydrate + retry loop) + scope `QQQ_Pe` `git add`. Atomic so no still-`git add "Stock Data.db"` writer re-adds it. `git revert`-able (only loses the DB-history gap during the window, which doesn't matter once consumers read `data`).
- **STAGE D — retire `actions.yml` LAST.** Only after `origin/data` has advanced daily for several days **and** the live ticker-history charts are confirmed still updating. Remove its `schedule:` (keep `workflow_dispatch` as a week-long parachute). After Stage C its `git add .` already skips the ignored DB, so it stops bloating main regardless.

## Gate before Stage A
The trial persist must prove itself: confirm over several real evening/weekly runs that `origin/data` receives correct, full-size DB commits whose contents match what main_remote produced. Only then start Stage A.

## Out of scope (separate, deferred)
`git filter-repo` to purge the 22 MB blob from `main`/`gh-pages` **history** (the flip stops *new* DB commits but doesn't rewrite history) — destructive, force-push, quiet-weekend task. Full design with per-file yaml snippets + the C1–C19 consumer map is in this session's verification output.
