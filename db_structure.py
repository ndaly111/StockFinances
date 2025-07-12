# inspect_db.py — lists all tables, columns, and sample rows
import sqlite3
from pathlib import Path

DB_FILE     = "Stock Data.db"
OUT_FILE    = "db_structure.txt"
SAMPLE_ROWS = 10  # How many example rows to show per table

def main():
    if not Path(DB_FILE).is_file():
        raise FileNotFoundError(f"{DB_FILE} not found")

    report_lines = []

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()

        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cur.fetchall()]
        if not tables:
            report_lines.append("No tables found in database.")
        else:
            for table in tables:
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
                    pk_str = " PK" if pk else ""
                    report_lines.append(f"    - {name}  ({ctype}{nn}{pk_str})")

                # ── row count
                try:
                    cur.execute(f"SELECT COUNT(*) FROM '{table}';")
                    total = cur.fetchone()[0]
                    report_lines.append(f"  Total rows: {total:,}")
                except Exception as e:
                    report_lines.append(f"  [Error counting rows: {e}]")
                    total = 0

                # ── sample rows
                if total:
                    try:
                        cur.execute(f"SELECT * FROM '{table}' LIMIT {SAMPLE_ROWS};")
                        sample = cur.fetchall()
                        headers = [c[1] for c in cols]
                        report_lines.append(f"\n  First {len(sample)} row(s):")
                        report_lines.append("    " + " | ".join(headers))
                        for row in sample:
                            report_lines.append("    " + " | ".join(map(str, row)))
                    except Exception as e:
                        report_lines.append(f"  [Error fetching sample rows: {e}]")
                report_lines.append("")  # Blank line between tables

    Path(OUT_FILE).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Wrote database summary to {OUT_FILE}")

if __name__ == "__main__":
    main()
