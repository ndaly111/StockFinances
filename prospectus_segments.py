#!/usr/bin/env python3
# prospectus_segments.py
# One-shot tool: parse annual segment P&L tables out of an S-1/424B4 prospectus
# (plain HTML, NOT XBRL-tagged) and freeze them into data/segment_seed_{TICKER}.json.
#
# The seed is consumed by generate_segment_charts.py, which uses it to replace
# (mode="replace") or backfill (mode="fill") the iXBRL-scraped segment data.
# Re-run this only if the seed needs regenerating; the prospectus is static.
#
# Usage:
#   SEC_EMAIL=you@example.com python prospectus_segments.py          # writes SPCX seed
#
# Why SPCX needs this: SpaceX IPO'd 2026-06-12 and has no 10-K yet, so the
# iXBRL path (sec_segment_data_arelle) only sees the 10-Q and mislabels
# double-counted 3mo+6mo sums as fiscal years. The 424B4 prospectus carries
# clean FY2023-2025 segment P&Ls (Note 19) with the full expense breakdown.

from __future__ import annotations

import html as html_mod
import json
import re
from pathlib import Path

from sec_segment_data_arelle import _sec_get

# 424B4 filed 2026-06-12, accession 0001628280-26-042639 (final IPO prospectus;
# same financials as the S-1/A, and the S-1 itself carries no XBRL for these tables).
SPCX_PROSPECTUS_URL = (
    "https://www.sec.gov/Archives/edgar/data/1181412/000162828026042639/"
    "spaceexplorationtechnologi.htm"
)

SEGMENTS = ["Space", "Connectivity", "AI"]

# Row label (as it appears in the prospectus) -> seed metric key.
# Order here is also the display order used by the expense table on the site.
METRIC_MAP = [
    ("Revenue", "Revenue"),
    ("Cost of revenue", "CostOfRevenue"),
    ("Research and development", "ResearchAndDevelopment"),
    ("Selling, general, and administrative", "SellingGeneralAndAdministrative"),
    ("Restructuring charges", "RestructuringCharges"),
    ("Impairment", "Impairment"),
    ("Total costs and expenses", "TotalCostsAndExpenses"),
    ("Income (loss) from operations", "OpIncome"),
]


def _table_to_rows(table_html: str) -> list[list[str]]:
    """Flatten one <table> into rows of cell strings."""
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", table_html, re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
        cleaned = []
        for c in cells:
            c = re.sub(r"<[^>]+>", "", c)
            c = html_mod.unescape(c)
            c = c.replace("\xa0", " ").strip()
            cleaned.append(c)
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _parse_money(cell: str):
    """'$(657)' -> -657.0, '4,423' -> 4423.0, em-dash/blank -> 0.0/None."""
    s = cell.replace("$", "").replace(",", "").strip()
    s = s.strip(".")
    if not s:
        return None
    if s in {"—", "–", "-", "—", "–"} or all(ch in "—–-—– " for ch in s):
        return 0.0  # accounting dash = zero
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _row_values(cells: list[str]) -> list[float]:
    """Numeric cells of a row, in order (label and filler cells skipped)."""
    vals = []
    for c in cells[1:]:
        v = _parse_money(c)
        if v is not None:
            vals.append(v)
    return vals


def parse_prospectus_segment_tables(page_html: str) -> dict:
    """Return {year: {segment: {metric: value_in_millions}}} for each annual
    segment P&L table (identified by full expense detail + FY header)."""
    out: dict[str, dict] = {}
    for table_html in re.findall(r"<table.*?</table>", page_html, re.S | re.I):
        if "Connectivity" not in table_html or "Cost of revenue" not in table_html:
            continue
        text = re.sub(r"<[^>]+>", " ", table_html)
        if "Year Ended December 31" not in html_mod.unescape(text).replace("\xa0", " "):
            continue  # skip quarterly (Three Months Ended) tables
        m = re.search(r"Year Ended December 31[^0-9]*(20\d\d)", html_mod.unescape(text))
        if not m:
            continue
        year = m.group(1)
        rows = _table_to_rows(table_html)
        year_data: dict[str, dict] = {s: {} for s in SEGMENTS}
        seen = set()
        for cells in rows:
            label = re.sub(r"\.{2,}.*$", "", cells[0]).strip().rstrip(".")
            for want, key in METRIC_MAP:
                if label == want and key not in seen:
                    vals = _row_values(cells)
                    if len(vals) >= len(SEGMENTS):
                        for seg, v in zip(SEGMENTS, vals):
                            year_data[seg][key] = v
                        seen.add(key)  # 1st occurrence only (Impairment repeats in supplemental)
                    break
        if len(seen) >= 6:  # a real P&L table, not the EBITDA reconciliation
            out[year] = year_data
    return out


def build_seed(parsed: dict) -> dict:
    rows = []
    for year in sorted(parsed):
        for seg in SEGMENTS:
            rows.append({"Segment": seg, "Year": year, **parsed[year][seg]})
    return {
        "ticker": "SPCX",
        "source": "424B4 prospectus (accession 0001628280-26-042639, filed 2026-06-12)",
        "source_url": SPCX_PROSPECTUS_URL,
        "units": "millions",
        "axis": "StatementBusinessSegmentsAxis",
        # replace: the iXBRL path mislabels double-counted 10-Q sums as fiscal
        # years until SpaceX files its first 10-K (~early 2027). Once a clean
        # 10-K exists, flip to "fill" (or delete the seed) so live data resumes.
        "mode": "replace",
        "rows": rows,
    }


def main():
    resp = _sec_get(SPCX_PROSPECTUS_URL)
    resp.raise_for_status()
    parsed = parse_prospectus_segment_tables(resp.text)
    if not parsed:
        raise SystemExit("No annual segment tables found — prospectus layout changed?")
    seed = build_seed(parsed)
    out = Path("data") / "segment_seed_SPCX.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    years = sorted(parsed)
    print(f"wrote {out} — years {years}, {len(seed['rows'])} rows")
    for year in years:
        total = sum(parsed[year][s].get("Revenue", 0) for s in SEGMENTS)
        print(f"  FY{year} revenue total: ${total:,.0f}M")


if __name__ == "__main__":
    main()
