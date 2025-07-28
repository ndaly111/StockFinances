#!/usr/bin/env python3
"""
segment_etl.py – Pilot ETL for business-segment facts (Revenue & Operating Income)
---------------------------------------------------------------------------------
• Reads SEC email from env var `Email`
• Pulls companyfacts JSON for each CIK
• Keeps only USD facts on a …SegmentsAxis dimension (10-K / 10-K/A)
• Classifies concept → metric_class (REV / OPINC / UNKNOWN)
• Writes to two tables in Stock Data.db:
      concept_map    – remembers how each concept is classified
      segment_facts  – one row per segment fact
• Prints a pivot summary per ticker for human inspection
"""

import os, re, time, sqlite3, requests, pandas as pd
from datetime import datetime

# ───────── CONFIG ──────────────────────────────────────────────────────────────
EMAIL      = os.getenv("Email")           # SEC requires polite User-Agent
if not EMAIL:
    raise SystemExit("ERROR: export Email='your.sec.address@example.com' first")

DB_PATH    = "Stock Data.db"
TICKER2CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
}

HEADERS = {"User-Agent": f"{EMAIL} - segment-etl script"}
TARGET_FORMS = {"10-K", "10-K/A"}
SEG_AX_RE    = re.compile(r"SegmentsAxis$", re.IGNORECASE)
REV_RE       = re.compile(r"(Revenue|Sales|NetSales)", re.IGNORECASE)
OPINC_RE     = re.compile(r"(OperatingIncome|OperatingProfit|OperatingIncomeLoss)", re.IGNORECASE)
PAUSE_SEC    = 0.25                       # courtesy pause between company hits

# ───────── DB helpers ──────────────────────────────────────────────────────────
def ensure_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS concept_map (
        concept       TEXT PRIMARY KEY,
        metric_class  TEXT CHECK(metric_class IN ('REV','OPINC','IGNORE','UNKNOWN'))
    );

    CREATE TABLE IF NOT EXISTS segment_facts (
        cik            TEXT,
        ticker         TEXT,
        fiscal_year    INTEGER,
        axis_member    TEXT,
        concept        TEXT,
        metric_class   TEXT,
        value_usd      REAL,
        form           TEXT,
        end_date       TEXT,
        PRIMARY KEY(cik, axis_member, concept, end_date)
    );
    """)
    conn.commit()

def classify_concept(cur: sqlite3.Cursor, concept: str, label: str) -> str:
    """Return metric_class and update concept_map if new."""
    cur.execute("SELECT metric_class FROM concept_map WHERE concept=?", (concept,))
    row = cur.fetchone()
    if row:
        return row[0]

    # first-time seen – use regex heuristics
    if REV_RE.search(concept) or REV_RE.search(label):
        cls = "REV"
    elif OPINC_RE.search(concept) or OPINC_RE.search(label):
        cls = "OPINC"
    else:
        cls = "UNKNOWN"

    cur.execute("INSERT OR IGNORE INTO concept_map VALUES (?,?)", (concept, cls))
    return cls

# ───────── SEC fetch / parse ───────────────────────────────────────────────────
def fetch_companyfacts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def stream_segment_facts(cik: str, data: dict):
    """Yield dicts of candidate segment facts after dimension & form filters."""
    for ns_tags in data.get("facts", {}).values():           # loop namespaces
        for concept, tag_data in ns_tags.items():            # loop tags
            human_label = tag_data.get("label", concept)     # fallback label
            for unit, facts in tag_data.get("units", {}).items():
                if unit != "USD":
                    continue
                for f in facts:
                    if f.get("form") not in TARGET_FORMS or not f.get("segments"):
                        continue
                    dims = f["segments"][0].get("dimensions", {})
                    if not any(SEG_AX_RE.search(dim) for dim in dims):
                        continue
                    member = next(iter(dims.values()))
                    end   = f["end"]
                    fy    = datetime.fromisoformat(end).year
                    yield {
                        "concept":       concept,
                        "label":         human_label,
                        "member":        member,
                        "val":           f["val"],
                        "end":           end,
                        "fiscal_year":   fy,
                        "form":          f["form"],
                    }

# ───────── main ETL routine ────────────────────────────────────────────────────
def run_etl():
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)
    cur = conn.cursor()

    for tk, cik in TICKER2CIK.items():
        print(f"🔎 {tk} …")
        data = fetch_companyfacts(cik)

        for fact in stream_segment_facts(cik, data):
            cls = classify_concept(cur, fact["concept"], fact["label"])
            if cls == "IGNORE":
                continue

            cur.execute("""
            INSERT OR REPLACE INTO segment_facts
              (cik,ticker,fiscal_year,axis_member,concept,metric_class,
               value_usd,form,end_date)
            VALUES (?,?,?,?,?,?,?,?,?)
            """, (cik, tk, fact["fiscal_year"], fact["member"],
                  fact["concept"], cls, fact["val"], fact["form"], fact["end"]))

        conn.commit()
        time.sleep(PAUSE_SEC)

    # ─── quick pivot to verify ────────────────────────
    df = pd.read_sql_query("""
        SELECT ticker, axis_member AS segment,
               SUM(CASE WHEN metric_class='REV'  THEN value_usd END) AS revenue,
               SUM(CASE WHEN metric_class='OPINC' THEN value_usd END) AS op_income
        FROM segment_facts
        WHERE fiscal_year >= strftime('%Y','now')-1   -- latest year per company
        GROUP BY ticker, segment
        ORDER BY ticker, revenue DESC
    """, conn)

    print("\n=== Latest business-segment snapshot ===")
    print(df.to_string(index=False, float_format='{:,.0f}'.format))

    # Show any unknown concepts for manual follow-up
    unknown = pd.read_sql_query(
        "SELECT concept FROM concept_map WHERE metric_class='UNKNOWN'", conn)
    if not unknown.empty:
        print("\n⚠️  Review these concepts (metric_class='UNKNOWN'):\n",
              unknown['concept'].to_list())

    conn.close()

if __name__ == "__main__":
    run_etl()
