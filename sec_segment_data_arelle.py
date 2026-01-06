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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from requests import exceptions as req_exc
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 20

# ──────────────────────────────────────────────────────────────────────────────
# SEC HTTP headers + tiny CIK fallback
# ──────────────────────────────────────────────────────────────────────────────

def _sec_headers() -> dict:
    """
    SEC requires a descriptive User-Agent with contact info.
    Set env SEC_USER_AGENT in CI:
      SEC_USER_AGENT: StockFinances/1.0 (Contact: you@example.com)

    If no explicit UA is provided we build one automatically, optionally using a
    contact email exposed via SEC_EMAIL/SEC_CONTACT/Email/EMAIL secrets.
    """
    ua = (os.getenv("SEC_USER_AGENT") or os.getenv("SEC_UA") or "").strip()
    if ua:
        return {"User-Agent": ua}

    email = (
        os.getenv("SEC_EMAIL")
        or os.getenv("SEC_CONTACT")
        or os.getenv("Email")
        or os.getenv("EMAIL")
        or "you@example.com"
    ).strip()
    ua = f"StockFinancesBot/1.0 (contact: {email})"
    return {"User-Agent": ua}


def _sec_get(url: str, *, timeout: int = REQUEST_TIMEOUT, stream: bool = False):
    """Wrapper around ``requests.get`` with friendlier SEC-specific errors."""

    headers = _sec_headers()
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=stream)
        resp.raise_for_status()
        return resp
    except req_exc.ProxyError as exc:
        raise RuntimeError(
            "SEC request blocked by proxy for "
            f"{url}. Ensure your network allows outbound HTTPS to data.sec.gov or configure the proxy "
            "credentials correctly."
        ) from exc
    except req_exc.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 403:
            raise RuntimeError(
                "SEC returned HTTP 403 for "
                f"{url}. Provide a descriptive contact email via SEC_USER_AGENT/SEC_UA or check that "
                "your account is permitted to access the endpoint."
            ) from exc
        raise RuntimeError(f"SEC request failed with HTTP {status} for {url}: {exc}") from exc
    except req_exc.RequestException as exc:
        raise RuntimeError(f"SEC request failed for {url}: {exc}") from exc

_FALLBACK_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOG": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}

# Simple in-memory CIK cache
_CIK_CACHE: Dict[str, int] | None = None

def _load_cik_map() -> Dict[str, int]:
    """Load (and cache) ticker→CIK mapping."""
    global _CIK_CACHE
    if _CIK_CACHE is None:
        _CIK_CACHE = {t: int(cik) for t, cik in _FALLBACK_CIK.items()}
    return _CIK_CACHE


def _resolve_ticker_to_cik_online(ticker: str) -> str:
    t = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    with _sec_get(url, timeout=30) as resp:
        data = resp.json()
    for rec in data.values():
        if rec.get("ticker", "").upper() == t:
            return str(rec["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found in SEC mapping: {ticker}")


def _cik(ticker: str) -> int:
    m = _load_cik_map()
    t = ticker.upper().strip()
    if t not in m:
        m[t] = int(_resolve_ticker_to_cik_online(t))
    return m[t]


def resolve_ticker_to_cik(ticker: str) -> str:
    """Public wrapper: return zero-padded 10-digit CIK string."""
    return f"{_cik(ticker):010d}"

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
# Previously we only captured dimensions whose qualified name contained
# “segment”.  Apple (and many other filers) report product level data using
# ``ProductOrServiceAxis`` which does not include the word “segment".  As a
# result, product/service facts were skipped and the generated tables showed
# the same axis twice.  Broaden the regex so that we recognise dimensions that
# mention "product" or "service" as well.
_SEGMENT_DIM_RE = re.compile(r"(segment|product|service)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Basic EDGAR helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_latest_filings(cik: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict with '10-K' and '10-Q' metadata:
      { '10-K': {'accession': '000...', 'document': 'xyz-YYYYMMDD.htm', 'filed': 'YYYY-MM-DD'},
        '10-Q': {...} }
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    with _sec_get(url, timeout=30) as resp:
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
    with _sec_get(url, timeout=60, stream=True) as r:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)


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
                if not any(_SEGMENT_DIM_RE.search(d) for d in dims.keys()):
                    continue

                segment_label = None
                axis_type = None
                for dim_qn, mem_qn in dims.items():
                    if _SEGMENT_DIM_RE.search(dim_qn) and mem_qn:
                        segment_label = mem_qn.split(":")[-1]
                        axis_type = dim_qn.split(":")[-1]
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
                    "AxisType": axis_type,
                    "PeriodEnd": ctx["period_end"],
                    "Value": val,
                })

            if rows:
                break  # stop at first concept that produced usable facts

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["PeriodEnd"].notna()]
        return df, used

    rev_df, rev_used = collect(REV_TAGS)
    op_df, op_used = collect(OPINC_TAGS)

    if rev_df.empty and op_df.empty:
        return pd.DataFrame(columns=["Segment", "AxisType", "PeriodEnd", "Revenue", "OpIncome"]), rev_used, op_used

    # sum by segment + axis + period
    rev_g = (
        rev_df.groupby(["Segment", "AxisType", "PeriodEnd"], as_index=False)["Value"].sum()
        if not rev_df.empty
        else pd.DataFrame(columns=["Segment", "AxisType", "PeriodEnd", "Value"])
    )
    rev_g.rename(columns={"Value": "Revenue"}, inplace=True)

    op_g = (
        op_df.groupby(["Segment", "AxisType", "PeriodEnd"], as_index=False)["Value"].sum()
        if not op_df.empty
        else pd.DataFrame(columns=["Segment", "AxisType", "PeriodEnd", "Value"])
    )
    op_g.rename(columns={"Value": "OpIncome"}, inplace=True)

    df = pd.merge(rev_g, op_g, on=["Segment", "AxisType", "PeriodEnd"], how="outer")
    df = df.sort_values(["PeriodEnd", "AxisType", "Segment"]).reset_index(drop=True)
    return df, (rev_used or ""), (op_used or "")


def collect_all_segment_facts(ix_path: Path) -> pd.DataFrame:
    """Return all revenue/operating income facts with any dimensions.

    This helper ignores the `_SEGMENT_DIM_RE` filter and returns a DataFrame
    with columns: Concept, PeriodEnd, Dims, Value. Dims is a semicolon-delimited
    list of `dimension:member` pairs. Missing or non-numeric facts are skipped.
    """

    html = ix_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "xml")

    contexts = _parse_contexts_ixbrl(soup)
    rows = []
    for cname in REV_TAGS + OPINC_TAGS:
        facts = soup.find_all("nonFraction", attrs={"name": cname})
        for fact in facts:
            ctx_id = fact.get("contextRef")
            if not ctx_id or ctx_id not in contexts:
                continue
            ctx = contexts[ctx_id]
            val = _scale_and_sign(
                fact.text,
                fact.get("decimals"),
                fact.get("scale"),
                fact.get("sign"),
            )
            if val is None:
                continue
            dims = ctx.get("dims", {})
            dim_str = "; ".join(f"{k}:{v}" for k, v in dims.items())
            rows.append(
                {
                    "Concept": cname,
                    "PeriodEnd": ctx["period_end"],
                    "Dims": dim_str,
                    "Value": val,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["PeriodEnd"].notna()]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# TTM and public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_segment_ttm(fy_df: pd.DataFrame, q_df: pd.DataFrame) -> pd.DataFrame:
    """
    TTM ≈ sum of the most recent four quarters per segment.

    We combine quarterly rows from both the latest 10-Q and the most recent
    10-K, then pick the trailing four periods per segment/axis. This ensures
    the latest fiscal Q4 (which comes from the 10-K) is included once a
    company has filed its annual report.
    """
    if fy_df.empty and q_df.empty:
        return pd.DataFrame(columns=["Segment", "AxisType", "Year", "Revenue", "OpIncome"])

    quarters = pd.concat([q_df, fy_df], ignore_index=True)
    if quarters.empty:
        return pd.DataFrame(columns=["Segment", "AxisType", "Year", "Revenue", "OpIncome"])

    quarters = quarters.dropna(subset=["PeriodEnd"])
    if quarters.empty:
        return pd.DataFrame(columns=["Segment", "AxisType", "Year", "Revenue", "OpIncome"])

    quarters = (
        quarters.groupby(["Segment", "AxisType", "PeriodEnd"], as_index=False)[["Revenue", "OpIncome"]]
        .sum()
    )

    rows = []
    for (seg, axis), g in quarters.groupby(["Segment", "AxisType"]):
        g = g.sort_values("PeriodEnd")
        if len(g) < 4:
            continue
        latest_four = g.tail(4)
        rows.append(
            {
                "Segment": seg,
                "AxisType": axis,
                "Year": "TTM",
                "Revenue": latest_four["Revenue"].sum(),
                "OpIncome": latest_four["OpIncome"].sum(),
            }
        )

    return pd.DataFrame(rows, columns=["Segment", "AxisType", "Year", "Revenue", "OpIncome"])


def get_segment_data(
    ticker: str, *, dump_raw: bool = False, raw_dir: Path | str | None = None
) -> pd.DataFrame:
    """Fetch segment-level revenue & operating income for ``ticker``.

    Parameters
    ----------
    ticker:
        Stock ticker symbol, e.g. ``"AAPL"``.
    dump_raw:
        If ``True``, write all unfiltered revenue and operating-income facts to
        ``{raw_dir}/{ticker}_segment_raw.txt``. Defaults to ``False``.
    raw_dir:
        Directory in which the raw text file should be written when
        ``dump_raw`` is ``True``. May be a ``Path`` or string. If ``None`` (the
        default), the file is written to ``charts/{ticker}`` relative to the
        current working directory.

    Returns
    -------
    pandas.DataFrame
        Columns: Segment, Year (yyyy or ``"TTM"``), Revenue, OpIncome.
        ``df.attrs['revenue_concept']`` and ``df.attrs['op_income_concept']``
        record which XBRL concepts were used.
    """
    cik = resolve_ticker_to_cik(ticker)
    filings = fetch_latest_filings(cik)
    ten_k = filings.get("10-K")
    ten_q = filings.get("10-Q")

    rev_used = ""
    op_used = ""
    raw_facts: List[pd.DataFrame] = []

    # Download iXBRL HTMLs and parse
    with pd.option_context("display.width", 200):
        k_df = pd.DataFrame(columns=["Segment", "AxisType", "PeriodEnd", "Revenue", "OpIncome"])
        q_df = pd.DataFrame(columns=["Segment", "AxisType", "PeriodEnd", "Revenue", "OpIncome"])

        if ten_k:
            url_k = build_filing_url(cik, ten_k["accession"], ten_k["document"])
            k_path = Path(".cache_ix") / f"{ticker}_10k.htm"
            download_file(url_k, k_path)
            k_raw, k_rev_used, k_op_used = parse_ixbrl_segments(k_path)
            if not k_raw.empty:
                rev_used = rev_used or k_rev_used
                op_used = op_used or k_op_used
                k_df = k_raw
            if dump_raw:
                raw_facts.append(collect_all_segment_facts(k_path))

        if ten_q:
            url_q = build_filing_url(cik, ten_q["accession"], ten_q["document"])
            q_path = Path(".cache_ix") / f"{ticker}_10q.htm"
            download_file(url_q, q_path)
            q_raw, q_rev_used, q_op_used = parse_ixbrl_segments(q_path)
            if not q_raw.empty:
                rev_used = rev_used or q_rev_used
                op_used = op_used or q_op_used
                q_df = q_raw
            if dump_raw:
                raw_facts.append(collect_all_segment_facts(q_path))

    if dump_raw:
        raw_df = (
            pd.concat(raw_facts, ignore_index=True)
            if raw_facts
            else pd.DataFrame(columns=["Concept", "PeriodEnd", "Dims", "Value"])
        )
        out_dir = (
            Path(raw_dir).resolve()
            if raw_dir
            else (Path("charts") / ticker.upper()).resolve()
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{ticker.upper()}_segment_raw.txt"
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        if raw_df.empty:
            raw_path.write_text(
                "No revenue or operating income facts found.",
                encoding="utf-8",
            )
        else:
            with pd.option_context("display.max_colwidth", None):
                raw_path.write_text(raw_df.to_string(index=False), encoding="utf-8")

    # bail out gracefully if nothing
    if k_df.empty and q_df.empty:
        df = pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome", "AxisType"])
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
        grp = df.groupby(["Segment", "AxisType", "PeriodEnd"], as_index=False)[["Revenue", "OpIncome"]]
        try:
            g = grp.sum(min_count=1)
        except TypeError:
            # Older pandas may not support min_count on GroupBy.sum
            g = grp.sum()
        g["Year"] = g["PeriodEnd"].dt.year
        return g

    k_roll = _roll(k_y)
    q_roll = _roll(q_y)

    # Keep last 3 fiscal-year ends from 10-K side (if present), else from 10-Q
    source_for_years = k_roll if not k_roll.empty else q_roll
    years = sorted(source_for_years["Year"].unique())[-3:] if not source_for_years.empty else []
    fy = source_for_years[source_for_years["Year"].isin(years)][["Segment", "AxisType", "Year", "Revenue", "OpIncome"]].copy()

    # Compute TTM if we have both annual + quarterly dates
    ttm = (
        compute_segment_ttm(k_df if not k_df.empty else q_df, q_df)
        if not q_df.empty
        else pd.DataFrame(columns=["Segment", "AxisType", "Year", "Revenue", "OpIncome"])
    )

    out = pd.concat([fy, ttm], ignore_index=True)
    if out.empty:
        out = pd.DataFrame(columns=["Segment", "Year", "Revenue", "OpIncome", "AxisType"])
    else:
        # Sum duplicates (same Segment-Year-Axis)
        grp = out.groupby(["Segment", "AxisType", "Year"], as_index=False)[["Revenue", "OpIncome"]]
        try:
            out = grp.sum(min_count=1)
        except TypeError:
            # Older pandas may not support min_count on GroupBy.sum
            out = grp.sum()
        def _yrkey(y):
            return 9999 if y == "TTM" else int(y)
        out["__k"] = out["Year"].map(_yrkey)
        out = (
            out.sort_values(["__k", "Segment", "AxisType"], ascending=[False, True, True])
            .drop(columns="__k")
            .reset_index(drop=True)
        )

    out = out[["Segment", "Year", "Revenue", "OpIncome", "AxisType"]]

    out.attrs["revenue_concept"] = rev_used
    out.attrs["op_income_concept"] = op_used

    from segment_overrides import load_overrides, apply_segment_overrides

    try:
        ov = load_overrides()
        out = apply_segment_overrides(out, ticker, ov)
    except Exception:
        pass  # fail-open: do not break runs due to overrides

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10‑Q Item 2 (MD&A) → Axis ranking diagnostics


def _get_text(url: str) -> Optional[str]:
    try:
        with _sec_get(url, timeout=REQUEST_TIMEOUT) as r:
            ct = (r.headers.get("content-type") or "").lower()
            if "html" not in ct and "text" not in ct:
                return None
            r.encoding = r.encoding or "utf-8"
            return r.text
    except Exception:
        return None


def _company_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    with _sec_get(url, timeout=REQUEST_TIMEOUT) as r:
        return r.json()


def _locate_item2_text(html: str) -> Optional[str]:
    """Extract plain text slice of Part I, Item 2 (MD&A) from 10‑Q HTML."""
    if not html:
        return None
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    raw = re.sub(r"(?is)<.*?>", " ", txt)
    raw = re.sub(r"\s+", " ", raw)
    p_item2 = re.compile(r"(?i)\bItem\s+2\b.*?(Management.?s?\s+Discussion\s+and\s+Analysis|MD&A)?")
    p_item3 = re.compile(r"(?i)\bItem\s+3\b")
    p_item4 = re.compile(r"(?i)\bItem\s+4\b")
    m2 = p_item2.search(raw)
    if not m2:
        return None
    start = m2.start()
    m3 = p_item3.search(raw, pos=start + 1)
    m4 = p_item4.search(raw, pos=start + 1)
    ends = [m.end() for m in [m3, m4] if m]
    end = min(ends) if ends else min(len(raw), start + 20000)
    return raw[start:end].strip()


def _axis_group_from_axistype(axis_type: str) -> str:
    a = (axis_type or "").lower()
    if "operatingsegment" in a or "segmentsaxis" in a or "reportablesegment" in a:
        return "OPERATING"
    if "product" in a or "service" in a or "productline" in a or "category" in a:
        return "PRODUCT"
    if "geograph" in a or "region" in a or "country" in a or "domestic" in a or "foreign" in a:
        return "GEOGRAPHY"
    if "customer" in a:
        return "CUSTOMER"
    if "channel" in a or "distribution" in a:
        return "CHANNEL"
    return "OTHER"


def _rank_axes_with_item2(df: pd.DataFrame, item2_text: str) -> pd.DataFrame:
    if df is None or df.empty or not item2_text:
        return pd.DataFrame(columns=["AxisType", "AxisGroup", "MembersMatched", "AxisScore"])
    text = item2_text.lower()
    KW = {
        "OPERATING": ["segment", "operating", "reportable"],
        "PRODUCT": ["product", "service", "model", "category", "line"],
        "GEOGRAPHY": [
            "geograph",
            "region",
            "country",
            "domestic",
            "foreign",
            "international",
            "americas",
            "europe",
            "emea",
            "china",
            "japan",
            "apac",
            "united states",
        ],
        "CUSTOMER": ["customer"],
        "CHANNEL": ["channel", "distribution"],
    }
    weights = {"group": 2.0, "member": 3.0}
    df2 = df.copy()
    if "AxisType" not in df2.columns:
        df2["AxisType"] = ""
    df2["AxisGroup"] = df2["AxisType"].map(_axis_group_from_axistype)
    axis_members = (
        df2.groupby(["AxisType", "AxisGroup"])["Segment"]
        .apply(lambda s: sorted(set([str(x) for x in s if str(x).strip()])))
        .reset_index(name="Members")
    )
    rows = []
    for _, r in axis_members.iterrows():
        at = r["AxisType"]
        ag = r["AxisGroup"]
        members = r["Members"]
        g_hits = sum(1 for kw in KW.get(ag, []) if kw in text)
        m_hits = 0
        matched_examples = []
        for m in members:
            m_norm = str(m).lower().strip()
            if not m_norm:
                continue
            if m_norm in text:
                m_hits += 1
                if len(matched_examples) < 3:
                    matched_examples.append(m)
        score = weights["group"] * g_hits + weights["member"] * m_hits
        rows.append(
            {
                "AxisType": at,
                "AxisGroup": ag,
                "MembersMatched": m_hits,
                "AxisScore": round(score, 3),
                "Examples": ", ".join(matched_examples),
            }
        )
    out = pd.DataFrame(rows).sort_values(["AxisScore", "MembersMatched"], ascending=[False, False])
    return out.reset_index(drop=True)


def dump_item2_axis_ranking(ticker: str, df: pd.DataFrame, out_root: str = "charts") -> None:
    """Fetch latest 10-Q Item 2 text and rank axes against it; writes diagnostics."""
    try:
        cik = resolve_ticker_to_cik(ticker)
        subs = _company_submissions(cik)
        recent = (subs.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        acc = recent.get("accessionNumber") or []
        prim = recent.get("primaryDocument") or []
        row = next((i for i, f in enumerate(forms) if (f or "").upper() == "10-Q"), None)
        if row is None:
            return
        acc_nd = (acc[row] or "").replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nd}/"
        url = urljoin(base, prim[row] or "")
        html = _get_text(url)
        if not html:
            return
        item2 = _locate_item2_text(html)
        if not item2:
            return
        rank = _rank_axes_with_item2(df, item2)
        if rank is None or rank.empty:
            return
        out_dir = Path(out_root) / ticker.upper() / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "item2_axis_rank.tsv").write_text(
            rank.to_csv(sep="\t", index=False), encoding="utf-8"
        )
        (out_dir / "item2_axis_rank.json").write_text(
            rank.to_json(orient="records"), encoding="utf-8"
        )
        print(f"[segments] wrote {out_dir/'item2_axis_rank.tsv'} and .json")
    except Exception as e:
        print(f"[segments] Item 2 axis ranking skipped for {ticker}: {e}")
