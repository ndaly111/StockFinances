# inspect_db.py  ── quick one-off schema / sample dump
import sqlite3
from pathlib import Path

DB_FILE            = "Stock Data.db"
OUT_FILE           = "db_structure.txt"          # results go here
TABLES_OF_INTEREST = ["Implied_Growth_History"]  # ← pick the table(s) you care about
SAMPLE_ROWS        = 10                          # how many example rows to print

def main():
    if not Path(DB_FILE).is_file():
        raise FileNotFoundError(f"{DB_FILE} not found")

    report_lines = []

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()

        # decide which tables to inspect
        if TABLES_OF_INTEREST:
            tables = [(t,) for t in TABLES_OF_INTEREST]
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cur.fetchall()

        for (table,) in tables:
            report_lines.append(f"─────────────────────────────────────────────")
            report_lines.append(f"Table: {table}")

            # ── schema
            cur.execute(f"PRAGMA table_info('{table}');")
            cols = cur.fetchall()
            if not cols:
                report_lines.append("  (no columns?)\n")
                continue

            report_lines.append("  Columns:")
            for cid, name, ctype, notnull, dflt, pk in cols:
                nn = " NOT NULL" if notnull else ""
                pk = " PK"       if pk       else ""
                report_lines.append(f"    - {name}  ({ctype}{nn}{pk})")

            # ── quick stats
            cur.execute(f"SELECT COUNT(*) FROM '{table}';")
            total = cur.fetchone()[0]
            report_lines.append(f"  Total rows: {total:,}")

            # ── sample rows
            if total:
                cur.execute(f"SELECT * FROM '{table}' LIMIT {SAMPLE_ROWS};")
                sample = cur.fetchall()
                headers = [c[1] for c in cols]
                report_lines.append(f"\n  First {len(sample)} row(s):")
                report_lines.append("    " + " | ".join(headers))
                for row in sample:
                    report_lines.append("    " + " | ".join(map(str, row)))
            report_lines.append("")  # blank line separator

    Path(OUT_FILE).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Wrote database summary to {OUT_FILE}")

if __name__ == "__main__":
    main()
