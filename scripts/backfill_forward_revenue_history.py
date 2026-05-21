"""One-shot backfill of forward_revenue snapshots into Forward_EPS_FY_History.

`ForwardFinancialData` has always carried `ForwardRevenue` alongside
`ForwardEPS`, but only the EPS value was snapshotted daily into
`Forward_EPS_FY_History`. That left the new forward-revenue chart with
only one data point per fiscal year on first launch, while the EPS chart
already had months of history.

This script walks the git history of `Stock Data.db`, opens each daily
snapshot, and uses its `ForwardFinancialData` rows to fill in the
`forward_revenue` and `revenue_analysts` columns of the matching
`Forward_EPS_FY_History` rows. Idempotent — only updates rows where
`forward_revenue` is still NULL.

Intended to run once via the `Backfill forward revenue history` workflow.
After it lands the updated DB, the next site refresh produces a revenue
chart with the same depth of history as the EPS chart.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "Stock Data.db"
DB_FILENAME_IN_GIT = "Stock Data.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("backfill")


def _git(*args: str) -> bytes:
    """Run `git -C <repo> <args>` and return stdout bytes."""
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args])


def find_commit_at_or_before(date_str: str) -> str | None:
    """Return the latest commit SHA that touched the DB on/before date_str.

    date_str: 'YYYY-MM-DD'. Treats the bound as end-of-day.
    """
    end = f"{date_str} 23:59:59"
    try:
        out = _git("log", "--format=%H", "-n", "1",
                   f"--before={end}", "--", DB_FILENAME_IN_GIT)
    except subprocess.CalledProcessError:
        return None
    sha = out.decode().strip()
    return sha or None


def extract_db_at_commit(commit: str, dest_path: str) -> None:
    """Stream the DB blob at commit into dest_path (raw binary)."""
    with open(dest_path, "wb") as f:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{DB_FILENAME_IN_GIT}"],
            stdout=f, check=True,
        )


def query_revenue_at(db_path: str) -> dict[tuple[str, str], tuple[float | None, int | None]]:
    """Read ForwardRevenue from a historical DB.

    Returns {(ticker, period_end): (forward_revenue, revenue_analysts)}.
    Skips rows where ForwardRevenue is NULL/missing.
    """
    out: dict[tuple[str, str], tuple[float | None, int | None]] = {}
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        # Older snapshots may not have ForwardRevenueAnalysts — guard with COALESCE.
        cur = con.execute(
            """
            SELECT Ticker, Date, ForwardRevenue, ForwardRevenueAnalysts
            FROM ForwardFinancialData
            WHERE ForwardRevenue IS NOT NULL
            """
        )
        for r in cur.fetchall():
            out[(r["Ticker"], r["Date"])] = (r["ForwardRevenue"], r["ForwardRevenueAnalysts"])
        con.close()
    except sqlite3.DatabaseError as e:
        log.warning(f"  historical DB unreadable: {e}")
    except sqlite3.OperationalError as e:
        # e.g. column missing on very old schemas — try minimal query
        log.warning(f"  schema mismatch, retrying without analysts: {e}")
        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT Ticker, Date, ForwardRevenue FROM ForwardFinancialData "
                "WHERE ForwardRevenue IS NOT NULL"
            )
            for r in cur.fetchall():
                out[(r["Ticker"], r["Date"])] = (r["ForwardRevenue"], None)
            con.close()
        except Exception as e2:
            log.warning(f"  fallback query failed: {e2}")
    return out


def main() -> int:
    if not DB_PATH.exists():
        log.error(f"DB not found at {DB_PATH}")
        return 1

    log.info("Collecting date_recorded values from Forward_EPS_FY_History needing backfill")
    main_con = sqlite3.connect(str(DB_PATH))
    main_con.execute("PRAGMA journal_mode=WAL")
    main_con.execute("PRAGMA busy_timeout=30000")

    # Ensure the columns exist (the chart code will also do this, but be safe).
    cur = main_con.execute("PRAGMA table_info(Forward_EPS_FY_History)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for col, decl in (("forward_revenue", "REAL"), ("revenue_analysts", "INTEGER")):
        if col not in existing_cols:
            try:
                main_con.execute(
                    f"ALTER TABLE Forward_EPS_FY_History ADD COLUMN {col} {decl}"
                )
                log.info(f"  added missing column {col}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

    cur = main_con.execute(
        """
        SELECT DISTINCT date_recorded
        FROM Forward_EPS_FY_History
        WHERE forward_revenue IS NULL
        ORDER BY date_recorded
        """
    )
    dates = [r[0] for r in cur.fetchall()]
    if not dates:
        log.info("Nothing to backfill — every row already has forward_revenue.")
        return 0
    log.info(f"  {len(dates)} dates need backfill ({dates[0]} -> {dates[-1]})")

    updates_total = 0
    dates_with_data = 0
    dates_no_commit = 0
    dates_unreadable = 0

    for date_str in dates:
        commit = find_commit_at_or_before(date_str)
        if not commit:
            dates_no_commit += 1
            log.info(f"  {date_str}: no DB commit at/before, skipping")
            continue

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="sf_hist_")
        os.close(tmp_fd)
        try:
            extract_db_at_commit(commit, tmp_path)
            rev_data = query_revenue_at(tmp_path)
            if not rev_data:
                dates_unreadable += 1
                log.info(f"  {date_str} ({commit[:7]}): no revenue rows in historical DB")
                continue

            updates_this_date = 0
            for (ticker, period_end), (rev, analysts) in rev_data.items():
                upd = main_con.execute(
                    """
                    UPDATE Forward_EPS_FY_History
                       SET forward_revenue  = ?,
                           revenue_analysts = ?
                     WHERE date_recorded = ?
                       AND ticker        = ?
                       AND period_end    = ?
                       AND forward_revenue IS NULL
                    """,
                    (rev, analysts, date_str, ticker, period_end),
                )
                updates_this_date += upd.rowcount

            main_con.commit()
            updates_total += updates_this_date
            dates_with_data += 1
            log.info(f"  {date_str} ({commit[:7]}): backfilled {updates_this_date} rows")
        except subprocess.CalledProcessError as e:
            log.error(f"  {date_str} ({commit[:7]}): git show failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    log.info(
        f"Backfill complete: {updates_total} rows updated "
        f"across {dates_with_data}/{len(dates)} dates "
        f"(no_commit={dates_no_commit}, no_data={dates_unreadable})"
    )
    main_con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
