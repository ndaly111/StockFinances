#!/usr/bin/env python3
"""
sec_segment_data_arelle.py
Option 1: Pull segment revenue & operating income directly from SEC Inline XBRL.

What it does
------------
- Resolves ticker -> CIK (with small built-in fallback for common tickers)
- Finds latest 10-K and 10-Q from EDGAR "submissions" JSON
- Downloads the Inline XBRL (iXBRL) HTML files
- Parses contexts & facts to find segment-dimensioned:
    * Revenue (prefers us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax,
      falls back to SalesRevenueNet or Revenues)
    * Operating income (prefers SegmentOperatingIncomeLoss, falls back to OperatingIncomeLoss)
- Produces a DataFrame with Segment, Year/TTM, Revenue, OpIncome
- Records which concepts were used in df.attrs
- Requires: pandas, requests, beautifulsoup4, lxml

Install:
    pip install pandas requests beautifulsoup4 lxml
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
# SEC HTTP headers + tiny CIK fallback
# ──────────────────────────────────────────────────────────────────────────────

def _sec_headers() -> dict:
    """
    SEC requires a descriptive User-Agent with contact info.
    Set env SEC_USER_AGENT in CI:
      SEC_USER_AGENT: StockFinances/1.0 (Contact: you@example.com)
    """
    ua = os.getenv("SEC_USER_AGENT") or os.getenv("SEC_UA") or ""
    ua = ua.strip() or "StockFinancesBot/1.0 (contact: you@example.com)"
    return {"User-Agent": ua}

_FALLBACK_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOG": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}

# Preferred/fallback concepts
REV_TAGS = [
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:SalesRevenueNet",
    "us-gaap:Revenues",
]
OPINC_TAGS = [
    # First try segment-specific if provided; many filers just dimension OperatingIncomeLoss
    "us-gaap:SegmentOperatingIncomeLoss",
    "us-gaap:OperatingIncomeLoss",
]

# Regex helpers
_SEGMENT_DIM_RE = re.compile(r"segment", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Basic EDGAR helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_ticker_to_cik(ticker: str) -> str:
    t = ticker.upper().strip()
    if t in _FALLBACK_CIK:
        return _FALLBACK_CIK[t]

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to download ticker mapping: {e}")

    for rec in data.values():
        if rec.get("ticker", "").upper() == t:
            return str(rec["cik_str"]).zfill(10)

    raise ValueError(f"Ticker not found in SEC mapping: {ticker}")


def fetch_latest_filings(cik: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict with '10-K' and '10-Q' metadata:
      { '10-K': {'accession': '000...', 'document': 'xyz-YYYYMMDD.htm', 'filed': 'YYYY-MM-DD'},
        '10-Q': {...} }
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=_sec_headers(), timeout=30)
    resp.raise_for_status()
    j = resp.json()

    forms = j.get("filings", {}).get("recent", {})
    out = {}
    for ftype in ("10-K", "10-Q"):
        try:
            idxs = [i for i, f in enumerate(forms["form"]) if f == ftype]
            if not idxs:
                continue
            i0 = idxs[0]
            accession = forms["accessionNumber"][i0]
            primary = forms["primaryDocument"][i0]
            filed = forms["filingDate"][i0]
            out[ftype] = {
                "accession": accession.replace("-", ""),
                "document": primary,
                "filed": filed,
            }
        except Exception:
            pass
    return out


def build_filing_url(cik: str, accession_nodashes: str, primary_doc: str) -> str:
    # Example:
    # https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodashes}/{primary_doc}"
    return base


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=_sec_headers(), timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    time.sleep(0.5)  # be polite to SEC


# ──────────────────────────────────────────────────────────────────────────────
# Inline XBRL parsing (BeautifulSoup)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_contexts_ixbrl(soup: BeautifulSoup) -> Dict[str, dict]:
    """
    Returns contextId -> {
        'period_end': datetime,
        'dims': {dimension_qname: member_qname}
    }
    """
    ns_xbrli = ("xbrli", "http://www.xbrl.org/2003/instance")
    ns_xbrldi = ("xbrldi", "http://xbrl.org/2006/xbrldi")

    # The iXBRL HTML usually embeds a <xbrli:xbrl> section with contexts
    contexts = {}
    for ctx in soup.find_all(True, attrs={"id": True}):
        # look for xbrli:context tag
        if not (ctx.name.endswith("context") or ":context" in ctx.name):
            continue
        cid = ctx.get("id")
        if not cid:
            continue

        # period end
        period_end = None
        # <xbrli:period><xbrli:endDate>YYYY-MM-DD</xbrli:endDate>...
        end_tag = ctx.find(lambda t: t.name and t.name.endswith("endDate"))
        instant_tag = ctx.find(lambda t: t.name and t.name.endswith("instant"))
        date_text = None
        if end_tag and end_tag.text:
            date_text = end_tag.text.strip()
        elif instant_tag and instant_tag.text:
            date_text = instant_tag.text.strip()
        if date_text:
            try:
                period_end = datetime.strptime(date_text, "%Y-%m-%d")
            except Exception:
                period_end = None

        # explicit dims
        dims = {}
        seg = ctx.find(lambda t: t.name and t.name.endswith("segment"))
        if seg:
            for m in seg.find_all(lambda t: t.name and t.name.endswith("explicitMember")):
                dim = m.get("dimension") or ""
                val = m.text.strip() if m.text else ""
                if dim and val:
                    dims[dim] = val

        contexts[cid] = {"period_end": period_end, "dims": dims}
    return contexts


def _scale_and_sign(val_text: str, decimals: str | None, scale: str | None, sign: str | None) -> float:
    """
    Apply iXBRL numeric handling:
      - 'scale' indicates exponent-of-10 scaling (e.g., '6' => * 10^6)
      - 'decimals' may be present but can be ignored for this aggregation
      - 'sign' = '-' inverts the numeric value
    """
    if val_text is None:
        return None
    val_text = val_text.replace(",", "").strip()
    if val_text in ("", "—", "-", "NaN", "nan"):
        return None
    try:
        v = float(val_text)
    except Exception:
        return None

    try:
        sc = int(scale) if scale not in (None, "", "INF", "NaN") else 0
    except Exception:
        sc = 0
    if sc:
        v *= (10 ** sc)

    if (sign or "").strip() == "-":
        v = -v
    return v


def parse_ixbrl_segments(ix_path: Path) -> Tuple[pd.DataFrame, str, str]:
    """
    Parse a single iXBRL HTML file and return segmented revenue & op income.
    Returns (DataFrame, revenue_concept_used, op_income_concept_used).
    """
    html = ix_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "xml")

    contexts = _parse_contexts_ixbrl(soup)

    def collect(concepts: List[str]) -> Tuple[pd.DataFrame, str | None]:
        rows = []
        used = None
        for cname in concepts:
            # Facts are <ix:nonFraction name="us-gaap:Concept" contextRef="...">123</ix:nonFraction>
            facts = soup.find_all("nonFraction", attrs={"name": cname})
            if not facts:
                continue
            used = cname  # first concept that yields facts
            for fact in facts:
                ctx_id = fact.get("contextRef")
                if not ctx_id or ctx_id not in contexts:
                    continue
                ctx = contexts[ctx_id]
                dims = ctx["dims"] or {}
                # must have a segment-like dimension
                if not any(_SEGMENT_DIM_RE.search(d) for d in dims.keys()):
                    continue

                # choose the first member that looks like a business segment
                segment_label = None
                for dim_qn, mem_qn in dims.items():
                    if _SEGMENT_DIM_RE.search(dim_qn) and mem_qn:
                        segment_label = mem_qn.split(":")[-1]  # human-ish fallback
                        break
                if not segment_label:
                    continue

                val = _scale_and_sign(
                    fact.text,
                    fact.get("decimals"),
                    fact.get("scale"),
                    fact.get("sign"),
                )
                if val is None:
                    continue

                rows.append({
                    "Segment": segment_label,
                    "PeriodEnd": ctx["period_end"],
                    "Value": val,
                })

            if rows:
                break  # stop at first concept that produced usable facts

        df = pd.DataFrame(rows)
        # filter to non-null dates
        if not df.empty:
            df = df[df["PeriodEnd"].notna()]
        return df, used

    rev_df, rev_used = collect(REV_TAGS)
    op_df, op_used = collect(OPINC_TAGS)

    if rev_df.empty and op_df.empty:
        return pd.DataFrame(columns=["Segment", "PeriodEnd", "Revenue", "OpIncome"]), rev_used, op_used

    # sum by segment + period
    rev_g = rev_df.groupby(["Segment", "PeriodEnd"], as_index=False)["Value"].sum() if not rev_df.empty else pd.DataFrame(columns=["Segment", "PeriodEnd", "Value"])
    rev_g.rename(columns={"Value": "Revenue"}, inplace=True)

    op_g = op_df.groupby(["Segment", "PeriodEnd"], as_index=False)["Value"].sum() if not op_df.empty else pd.DataFrame(columns=["Segment", "PeriodEnd", "Value"])
    op_g.rename(columns={"Value": "OpIncome"}, inplace=True)

    df = pd.merge(rev_g, op_g, on=["Segment", "PeriodEnd"], how="outer")
    df = df.sort_values(["PeriodEnd", "Segment"]).reset_index(drop=True)
    return df, (rev_used or ""), (op_used or "")


# ──────────────────────────────────────────────────────────────────────────────
# TTM and public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_segment_ttm(fy_df: pd.DataFrame, q_df: pd.DataFrame) -> pd.DataFrame:
    """
    TTM ≈ latest FY + latest Q - same Q of prior FY, per segment.
    Requires PeriodEnd present (use before dropping it).
    """
    if fy_df.empty or q_df.empty:
        return pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome"])

    latest_q_date = q_df["PeriodEnd"].max()
    if pd.isna(latest_q_date):
        return pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome"])

    # derive “same quarter last year” by month/day offset of ~1 year
    prior_year_same_q_date = latest_q_date.replace(year=latest_q_date.year - 1)

    fy_last = fy_df[fy_df["PeriodEnd"] == fy_df["PeriodEnd"].max()]
    q_latest = q_df[q_df["PeriodEnd"] == latest_q_date]
    q_prev = q_df[q_df["PeriodEnd"] == prior_year_same_q_date]

    def _sum_cols(g: pd.DataFrame) -> pd.DataFrame:
        if g.empty:
            return pd.DataFrame(columns=["Segment", "Revenue", "OpIncome"])
        s = g.groupby("Segment", as_index=False)[["Revenue", "OpIncome"]].sum()
        return s

    ttm = _sum_cols(fy_last).merge(_sum_cols(q_latest), on="Segment", how="outer", suffixes=("_FY", "_Q"))
    ttm = ttm.merge(_sum_cols(q_prev), on="Segment", how="left")
    ttm.rename(columns={"Revenue": "Revenue_prevQ", "OpIncome": "OpIncome_prevQ"}, inplace=True)

    # Fill NaNs with 0 for arithmetic
    for col in ["Revenue_FY", "OpIncome_FY", "Revenue_Q", "OpIncome_Q", "Revenue_prevQ", "OpIncome_prevQ"]:
        if col in ttm.columns:
            ttm[col] = ttm[col].fillna(0.0)

    ttm["Revenue"] = ttm["Revenue_FY"] + ttm["Revenue_Q"] - ttm["Revenue_prevQ"]
    ttm["OpIncome"] = ttm["OpIncome_FY"] + ttm["OpIncome_Q"] - ttm["OpIncome_prevQ"]
    ttm["Year"] = "TTM"
    return ttm[["Segment", "Year", "Revenue", "OpIncome"]]


def get_segment_data(ticker: str) -> pd.DataFrame:
    """
    Public API:
      Input: ticker (e.g., 'AAPL')
      Output: DataFrame with columns: Segment, Year (yyyy or 'TTM'), Revenue, OpIncome
      df.attrs['revenue_concept'], df.attrs['op_income_concept'] record concept choices.
    """
    cik = resolve_ticker_to_cik(ticker)
    filings = fetch_latest_filings(cik)
    ten_k = filings.get("10-K")
    ten_q = filings.get("10-Q")

    rev_used = ""
    op_used = ""

    # Download iXBRL HTMLs and parse
    with pd.option_context("display.width", 200):
        k_df = pd.DataFrame(columns=["Segment", "PeriodEnd", "Revenue", "OpIncome"])
        q_df = pd.DataFrame(columns=["Segment", "PeriodEnd", "Revenue", "OpIncome"])

        if ten_k:
            url_k = build_filing_url(cik, ten_k["accession"], ten_k["document"])
            k_path = Path(".cache_ix") / f"{ticker}_10k.htm"
            download_file(url_k, k_path)
            k_raw, k_rev_used, k_op_used = parse_ixbrl_segments(k_path)
            if not k_raw.empty:
                rev_used = rev_used or k_rev_used
                op_used = op_used or k_op_used
                k_df = k_raw

        if ten_q:
            url_q = build_filing_url(cik, ten_q["accession"], ten_q["document"])
            q_path = Path(".cache_ix") / f"{ticker}_10q.htm"
            download_file(url_q, q_path)
            q_raw, q_rev_used, q_op_used = parse_ixbrl_segments(q_path)
            if not q_raw.empty:
                rev_used = rev_used or q_rev_used
                op_used = op_used or q_op_used
                q_df = q_raw

    # bail out gracefully if nothing
    if k_df.empty and q_df.empty:
        df = pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome"])
        df.attrs["revenue_concept"] = rev_used
        df.attrs["op_income_concept"] = op_used
        return df

    # Add "Year" from PeriodEnd (keep copies with PeriodEnd for TTM calc)
    def _with_year(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d["Year"] = d["PeriodEnd"].dt.year
        return d

    k_y = _with_year(k_df)
    q_y = _with_year(q_df)

    # Aggregate by Segment, PeriodEnd into totals (already aggregated, but ensure)
    def _roll(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        g = df.groupby(["Segment", "PeriodEnd"], as_index=False)[["Revenue", "OpIncome"]].sum()
        g["Year"] = g["PeriodEnd"].dt.year
        return g

    k_roll = _roll(k_y)
    q_roll = _roll(q_y)

    # Keep last 3 fiscal-year ends from 10-K side (if present), else from 10-Q
    source_for_years = k_roll if not k_roll.empty else q_roll
    years = sorted(source_for_years["Year"].unique())[-3:] if not source_for_years.empty else []
    fy = source_for_years[source_for_years["Year"].isin(years)][["Segment", "Year", "Revenue", "OpIncome"]].copy()

    # Compute TTM if we have both annual + quarterly dates
    ttm = compute_segment_ttm(k_df if not k_df.empty else q_df, q_df) if not q_df.empty else pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome"])

    out = pd.concat([fy, ttm], ignore_index=True)
    if out.empty:
        out = pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome"])
    else:
        # Sum duplicates (same Segment-Year) just in case
        out = out.groupby(["Segment", "Year"], as_index=False)[["Revenue", "OpIncome"]].sum()
        # Order nicely: latest first, TTM at bottom
        def _yrkey(y):
            return 9999 if y == "TTM" else int(y)
        out["__k"] = out["Year"].map(_yrkey)
        out = out.sort_values(["__k", "Segment"], ascending=[False, True]).drop(columns="__k").reset_index(drop=True)

    out.attrs["revenue_concept"] = rev_used
    out.attrs["op_income_concept"] = op_used
    return out
