#!/usr/bin/env python3
# s1_seed.py
# One-shot tool: turn a pre-IPO company's S-1 (plain HTML; the S-1 carries no
# XBRL for its financial statements) into two frozen files the site reads:
#
#   data/s1_seed_{TICKER}.json   companyfacts-shaped facts. build_aapl_mockup.py
#                                uses them only while EDGAR companyfacts is empty,
#                                so the first real 10-Q takes over automatically.
#   charts/{TICKER}/segments.json  revenue breakdowns for the Segment Performance
#                                section (same format generate_segment_charts.py writes).
#
# Every table is checked to foot (components sum to the reported total) before
# anything is written. Re-run only if a new S-1/A or 424B4 changes the numbers.
#
# Usage:
#   SEC_EMAIL=you@example.com python s1_seed.py ANT

from __future__ import annotations

import html as html_mod
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sec_segment_data_arelle import _sec_get

ROOT = Path(__file__).resolve().parent
PRE_IPO = json.loads((ROOT / "data" / "pre_ipo.json").read_text(encoding="utf-8"))

# Period columns of each S-1 table, in the order they appear.
ANNUAL_6M = [  # "Year Ended December 31, ... Six Months Ended June 30, ..."
    ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"),
    ("2025-01-01", "2025-06-30"), ("2026-01-01", "2026-06-30"),
]
QUARTERS = [
    ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
    ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
]
BALANCE = ["2024-12-31", "2025-12-31", "2026-06-30"]

_NUM = r"\(?-?[\d,]+(?:\.\d+)?\)?|[—–�-]"


def _text(url: str) -> str:
    raw = _sec_get(url).text
    t = html_mod.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", t)


def _num(tok: str) -> float:
    tok = tok.strip()
    if tok in ("—", "–", "�", "-"):
        return 0.0
    neg = tok.startswith("(")
    v = float(tok.strip("()").replace(",", ""))
    return -v if neg else v


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


def _row(section: str, label: str, n: int) -> list[float]:
    """The n numbers that follow `label` in a flattened table."""
    m = re.search(re.escape(label) + r"(?: \(\d\))?" + r"((?: \$?\s*(?:" + _NUM + r")){%d})" % n, section)
    if not m:
        raise ValueError(f"row not found: {label!r}")
    toks = re.findall(_NUM, m.group(1).replace("$", " "))
    return [_num(x) for x in toks[:n]]


def _foot(name: str, parts: list[list[float]], total: list[float], tol: float = 1.0) -> None:
    for i, tot in enumerate(total):
        s = sum(p[i] for p in parts)
        if abs(s - tot) > tol:
            raise ValueError(f"{name} does not foot in column {i}: parts {s} vs total {tot}")


def build(ticker: str) -> None:
    meta = PRE_IPO[ticker]
    text = _text(meta["s1_url"])

    # ---- audited statements of operations (FY2023-25 + H1 2025/26) ----------
    ops = _section(text, "Consolidated Statements of Operations (In millions",
                   "Consolidated Statements of Comprehensive Loss")
    rev = _row(ops, "Revenue $", 5)
    opex_labels = ["Compute and infrastructure", "Personnel", "Bartz settlement",
                   "Sales, marketing, and partnerships", "Other operating and administrative"]
    opex = [_row(ops, lab, 5) for lab in opex_labels]
    tot_opex = _row(ops, "Total operating expenses", 5)
    _foot("operating expenses", opex, tot_opex)
    op_inc = _row(ops, "Operating loss", 5)
    _foot("operating loss", [rev, [-x for x in tot_opex]], op_inc)
    pretax = _row(ops, "Loss before provision for income taxes", 5)
    tax = _row(ops, "Provision for income taxes", 5)
    ni = _row(ops, "Net loss $", 5)
    _foot("net loss", [pretax, [-x for x in tax]], ni)
    eps = _row(ops, "basic and diluted $", 5)

    # ---- unaudited quarterly statements (Q1 2025 - Q2 2026) -----------------
    qtr = _section(text, "Quarterly Consolidated Statements of Operations Three Months Ended",
                   "Quarterly Consolidated Statements of Operations, as a Percentage")
    q_rev = _row(qtr, "Revenue $", 6)
    q_op = _row(qtr, "Operating income (loss)", 6)
    q_ni = _row(qtr, "Net loss $", 6)
    # quarters must add up to the audited year / half-years
    _foot("2025 quarterly revenue", [[x] for x in q_rev[:4]], [rev[2]])
    _foot("H1 2026 quarterly revenue", [[x] for x in q_rev[4:]], [rev[4]])
    _foot("2025 quarterly net loss", [[x] for x in q_ni[:4]], [ni[2]])

    # ---- balance sheet --------------------------------------------------------
    bs = _section(text, "Consolidated Balance Sheets (In millions", "Consolidated Statements of Operations (In millions")
    cash = _row(bs, "Cash and cash equivalents $", 3)
    assets = _row(bs, "Total assets $", 3)
    liab = _row(bs, "Total liabilities", 3)
    equity = _row(bs, "Total stockholders’ deficit", 3) if "Total stockholders’ deficit" in bs \
        else _row(bs, "Total stockholders' deficit", 3)

    # ---- revenue disaggregation (note 2) ---------------------------------------
    geo = _section(text, "revenue by geographical region", "consumption-based and subscription-based revenue")
    us = _row(geo, "United States $", 5)
    intl = _row(geo, "International", 5)
    _foot("geography", [us, intl], rev)
    # the note's table, not the MD&A mention of the same phrase
    typ = _section(text[text.index("revenue by geographical region"):],
                   "consumption-based and subscription-based revenue", "Deferred")
    cons = _row(typ, "Consumption-based $", 5)
    subs = _row(typ, "Subscription-based", 5)
    _foot("revenue type", [cons, subs], rev)

    # ---- companyfacts-shaped seed --------------------------------------------
    filed = meta["filed"]
    M = 1e6

    def dur(vals, periods, scale=M):
        return [{"start": s, "end": e, "val": v * scale, "form": "S-1", "filed": filed,
                 "accn": meta["accession"]} for v, (s, e) in zip(vals, periods)]

    def inst(vals):
        return [{"end": e, "val": v * M, "form": "S-1", "filed": filed,
                 "accn": meta["accession"]} for v, e in zip(vals, BALANCE)]

    def usd(entries):
        return {"units": {"USD": entries}}

    facts = {"us-gaap": {
        "Revenues": usd(dur(rev, ANNUAL_6M) + dur(q_rev, QUARTERS)),
        "OperatingIncomeLoss": usd(dur(op_inc, ANNUAL_6M) + dur(q_op, QUARTERS)),
        "NetIncomeLoss": usd(dur(ni, ANNUAL_6M) + dur(q_ni, QUARTERS)),
        "EarningsPerShareDiluted": {"units": {"USD/shares": dur(eps, ANNUAL_6M, scale=1)}},
        "CashAndCashEquivalentsAtCarryingValue": usd(inst(cash)),
        "Assets": usd(inst(assets)),
        "Liabilities": usd(inst(liab)),
        "StockholdersEquity": usd(inst(equity)),
    }}
    seed = {
        "ticker": ticker,
        "name": meta["name"],
        "source": meta["s1_url"],
        "note": "Frozen from the S-1 (no XBRL financials). Used only while EDGAR companyfacts is empty.",
        "facts": facts,
    }
    seed_path = ROOT / "data" / f"s1_seed_{ticker}.json"
    seed_path.write_text(json.dumps(seed, indent=1), encoding="utf-8")
    print(f"wrote {seed_path}")

    # ---- segments.json: TTM = FY2025 + H1 2026 - H1 2025 -----------------------
    def axis(name, rows):
        segs = []
        for label, v in rows:
            r = {"2023": v[0] * M, "2024": v[1] * M, "2025": v[2] * M,
                 "TTM": (v[2] + v[4] - v[3]) * M}
            segs.append({"name": label, "revenue": r, "profit": {}})
        return {"axis": name, "profit_label": None,
                "years": ["2023", "2024", "2025", "TTM"], "segments": segs}

    seg = {
        "ticker": ticker,
        "version": "S-1 seed",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axes": [
            axis("RevenueTypeAxis", [("Consumption-based", cons), ("Subscription-based", subs)]),
            axis("StatementGeographicalAxis", [("United States", us), ("International", intl)]),
        ],
    }
    seg_dir = ROOT / "charts" / ticker
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segments.json").write_text(json.dumps(seg, indent=1), encoding="utf-8")
    print(f"wrote {seg_dir / 'segments.json'}")
    print(f"FY2025 revenue ${rev[2]:,.0f}M, TTM ${sum(q_rev[2:]):,.0f}M, H1-26 net loss ${ni[4]:,.0f}M")


if __name__ == "__main__":
    build((sys.argv[1] if len(sys.argv) > 1 else "ANT").upper())
