#!/usr/bin/env python3
"""
segment_etl.py – Business-segment ETL for MSFT, AAPL, TSLA
──────────────────────────────────────────────────────────
• Pulls latest 10-K + three 10-Qs per company
• Parses Inline-XBRL with Arelle (keeps dimensions)
• Filters rows whose axis ends with “SegmentsAxis”
• Classifies concepts → REV / OPINC / UNKNOWN (stored in concept_map)
• Writes annual & quarterly facts into segment_facts
• Prints annual + TTM snapshot for verification
"""
# ── compatibility shim for Py≥3.10 (Arelle 2.x expects old aliases) ───────────
import collections, collections.abc
for _alias in ("MutableSet", "MutableMapping", "MutableSequence"):
    if not hasattr(collections, _alias):
        setattr(collections, _alias, getattr(collections.abc, _alias))
# ──────────────────────────────────────────────────────────────────────────────
from arelle import Cntlr, ModelManager
import os, re, io, time, sqlite3, requests, pandas as pd
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
EMAIL        = os.getenv("Email")
if not EMAIL:
    raise SystemExit("ERROR: export Email='your.sec.address@example.com' first")

HEADERS      = {"User-Agent": f"{EMAIL} - segment ETL"}
DB_PATH      = "Stock Data.db"
TICKER2CIK   = {"MSFT": "0000789019", "AAPL": "0000320193", "TSLA": "0001318605"}
SEG_AX_RE    = re.compile(r"SegmentsAxis$", re.I)
REV_RE       = re.compile(r"(Revenue|Sales|NetSales)", re.I)
OPINC_RE     = re.compile(r"(OperatingIncome|OperatingProfit|OperatingIncomeLoss)", re.I)
PAUSE_SEC    = 0.30            # keep well <10 req/s

# ─── DB SETUP ─────────────────────────────────────────────────────────────────
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
        period_months  INTEGER,
        PRIMARY KEY (cik, axis_member, concept, end_date)
    );
    """)
    conn.commit()

def classify_concept(cur: sqlite3.Cursor, concept: str, label: str) -> str:
    cur.execute("SELECT metric_class FROM concept_map WHERE concept=?", (concept,))
    row = cur.fetchone()
    if row:
        return row[0]

    if REV_RE.search(concept) or REV_RE.search(label):
        cls = "REV"
    elif OPINC_RE.search(concept) or OPINC_RE.search(label):
        cls = "OPINC"
    else:
        cls = "UNKNOWN"

    cur.execute("INSERT OR IGNORE INTO concept_map VALUES (?,?)", (concept, cls))
    return cls

# ─── SEC HELPERS ──────────────────────────────────────────────────────────────
def latest_filings(cik: str):
    """Return up to 4 tuples: (accession, form, primary_document)."""
    subm = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        headers=HEADERS, timeout=30).json()

    picks = []
    for acc, form, doc in zip(subm["filings"]["recent"]["accessionNumber"],
                              subm["filings"]["recent"]["form"],
                              subm["filings"]["recent"]["primaryDocument"]):
        clean_acc = acc.replace("-", "")
        # keep 1×10-K + 3×10-Q at most
        if form.startswith("10-K") and not any(p[1].startswith("10-K") for p in picks):
            picks.append((clean_acc, form, doc))
        elif form.startswith("10-Q") and len([p for p in picks if p[1].startswith("10-Q")]) < 3:
            picks.append((clean_acc, form, doc))
        if len(picks) >= 4:
            break
    return picks

def download_instance(cik: str, accession: str, primary_doc: str) -> bytes:
    cik_num = str(int(cik))  # strip leading zeros
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession}/{primary_doc}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.content

# ─── XBRL PARSER ──────────────────────────────────────────────────────────────
def stream_segment_facts(inst_bytes: bytes):
    """Yield dicts for each business-segment fact in one Inline-XBRL instance."""
    cntlr = Cntlr.Cntlr(logFileName="logToPrint")
    model = ModelManager.initialize(cntlr).loadXbrl(io.BytesIO(inst_bytes))
    for fact in model.facts:
        dims = fact.context.segDimValues
        if not dims or not any(SEG_AX_RE.search(ax.localName) for ax, _ in dims):
            continue
        axis, member = next(iter(dims))
        end   = fact.context.endDatetime.date()
        start = fact.context.startDatetime.date()
        period_months = (end.year - start.year) * 12 + (end.month - start.month)
        yield {
            "concept":       fact.concept.qname.localName,
            "label":         fact.concept.label() or fact.concept.qname.localName,
            "member":        member.localName,
            "value":         float(fact.value or 0),
            "end":           end.isoformat(),
            "period_months": period_months,
        }
    model.close()

# ─── MAIN ETL ─────────────────────────────────────────────────────────────────
def run_etl():
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)
    cur = conn.cursor()

    for tk, cik in TICKER2CIK.items():
        print(f"▶ {tk} ({cik})")
        for accession, form, doc in latest_filings(cik):          # ← unpack 3
            try:
                inst = download_instance(cik, accession, doc)     # ← pass 3rd arg
            except Exception as e:
                print(f"  • skip {form} {accession[:10]} – {e}")
                continue

            for f in stream_segment_facts(inst):
                cls = classify_concept(cur, f["concept"], f["label"])
                if cls == "IGNORE":
                    continue
                fy = datetime.fromisoformat(f["end"]).year
                cur.execute("""
                INSERT OR REPLACE INTO segment_facts
                  (cik,ticker,fiscal_year,axis_member,concept,metric_class,
                   value_usd,form,end_date,period_months)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (cik, tk, fy, f["member"], f["concept"], cls,
                      f["value"], form, f["end"], f["period_months"]))
            conn.commit()
            time.sleep(PAUSE_SEC)

    # ─ verification pivot ───────────────────────────────────
    df = pd.read_sql_query("""
        WITH latest_year AS (
          SELECT ticker, MAX(fiscal_year) fy
          FROM segment_facts GROUP BY ticker)
        SELECT s.ticker,
               s.axis_member AS segment,
               SUM(CASE WHEN s.metric_class='REV'  AND s.period_months=12 THEN s.value_usd END) AS annual_rev,
               SUM(CASE WHEN s.metric_class='REV'  AND s.period_months=3  AND
                        DATE(s.end_date) >= DATE('now','-365 day')
                        THEN s.value_usd END) AS ttm_rev,
               SUM(CASE WHEN s.metric_class='OPINC' AND s.period_months=12 THEN s.value_usd END) AS annual_opinc,
               SUM(CASE WHEN s.metric_class='OPINC' AND s.period_months=3  AND
                        DATE(s.end_date) >= DATE('now','-365 day')
                        THEN s.value_usd END) AS ttm_opinc
        FROM segment_facts s
        JOIN latest_year y ON y.ticker=s.ticker AND y.fy=s.fiscal_year
        GROUP BY s.ticker, segment
        ORDER BY s.ticker, annual_rev DESC;
    """, conn)

    print("\n=== Annual + TTM snapshot ===")
    print(df.to_string(index=False, float_format='{:,.0f}'.format))

    unknown = pd.read_sql_query(
        "SELECT concept FROM concept_map WHERE metric_class='UNKNOWN'", conn)
    if not unknown.empty:
        print("\n⚠️  Review & label these concepts (metric_class='UNKNOWN'):\n",
              unknown['concept'].to_list())

    conn.close()

if __name__ == "__main__":
    run_etl()
