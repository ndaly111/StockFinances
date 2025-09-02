#!/usr/bin/env python3
"""
sec_segment_data_arelle.py — SEC CompanyFacts multi-axis segment extractor

Outputs a DataFrame with columns:
  Segment (clean human label)
  Year    (string, e.g., '2022' or 'TTM' not included here, TTM is handled by generator)
  Revenue (float, USD)
  OpIncome (float, USD; NaN if unavailable)
  AxisType (canonical axis name: ProductsAndServicesAxis, GeographicalAreasAxis, etc.)

- Combines multiple axes into a single composite axis so each fact contributes once.
- Filters to USD-like units.
- Polite SEC headers and small delay (PAUSE_SEC).
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd

SEC_CIK_CACHE = Path(".sec_cik_cache.json")
SEC_HEADERS_EMAIL_ENV = ("SEC_EMAIL", "EMAIL_FOR_SEC")
REQUEST_TIMEOUT = 20
PAUSE_SEC = 0.2

# Broad revenue / op-inc concept acceptance
REVENUE_BASE_TAGS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "Revenues",
    "NetSales",
    "ProductRevenue",
    "ServiceRevenue",
    "OperatingRevenue",
}

_REV_SEGMENT_RE = re.compile(
    r"(?:netsales|salesrevenue|revenue).*segment|segment.*(?:netsales|salesrevenue|revenue)",
    re.IGNORECASE,
)
_OPINC_LIKE_RE = re.compile(
    r"operatingincome|operatingprofit|segmentoperatingincome",
    re.IGNORECASE,
)

# Canonical axis names we keep
# Extended mapping to unify various SEC axis labels
AXIS_NORMALIZER = {
    # Geography → canonical "GeographicalAreasAxis"
    "StatementGeographicalAxis": "GeographicalAreasAxis",
    "GeographicalAreasAxis": "GeographicalAreasAxis",
    "GeographicalRegionsAxis": "GeographicalAreasAxis",
    "GeographicalRegionAxis": "GeographicalAreasAxis",
    "GeographicalSegmentsAxis": "GeographicalAreasAxis",
    "DomesticAndForeignAxis": "DomesticAndForeignAxis",
    "CountryAxis": "CountryAxis",

    # Product / Service → canonical "ProductsAndServicesAxis"
    "ProductOrServiceAxis": "ProductsAndServicesAxis",
    "ProductsAndServicesAxis": "ProductsAndServicesAxis",
    "ProductLineAxis": "ProductsAndServicesAxis",
    "ProductAxis": "ProductsAndServicesAxis",
    "ProductCategoryAxis": "ProductsAndServicesAxis",
    "ProductCategoriesAxis": "ProductsAndServicesAxis",
    "ProductSegmentsAxis": "ProductsAndServicesAxis",

    # Operating segments
    "OperatingSegmentsAxis": "OperatingSegmentsAxis",
    "BusinessSegmentsAxis": "OperatingSegmentsAxis",
    "ReportableSegmentsAxis": "OperatingSegmentsAxis",
    "SegmentsAxis": "OperatingSegmentsAxis",
    # Statement*SegmentsAxis variants from frames responses
    "StatementBusinessSegmentsAxis": "OperatingSegmentsAxis",

    # Customers / Channels
    "MajorCustomersAxis": "MajorCustomersAxis",
    "SignificantCustomersAxis": "MajorCustomersAxis",
    "SalesChannelsAxis": "SalesChannelsAxis",
    "DistributionChannelsAxis": "SalesChannelsAxis",
}

# Include "Statement*SegmentsAxis" variants automatically
AXIS_NORMALIZER.update({
    f"Statement{k}": v
    for k, v in list(AXIS_NORMALIZER.items())
    if k.endswith("SegmentsAxis") and not k.startswith("Statement") and f"Statement{k}" not in AXIS_NORMALIZER
})


_NEG_TOKENS = ("cost", "cogs", "expense", "gain", "loss", "grossprofit", "tax", "deferred", "impair", "interest")

def _user_agent_headers() -> Dict[str, str]:
    email = None
    for env_name in SEC_HEADERS_EMAIL_ENV:
        v = os.environ.get(env_name)
        if v:
            email = v.strip()
            break
    ua = f"ndaly-segments/1.3 ({email or 'email@domain.com'})"
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

def _load_cik_map() -> Dict[str, int]:
    if SEC_CIK_CACHE.is_file():
        try:
            return {k.upper(): int(v) for k, v in json.loads(SEC_CIK_CACHE.read_text()).items()}
        except Exception:
            pass
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=_user_agent_headers(), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    mapping = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    SEC_CIK_CACHE.write_text(json.dumps(mapping))
    return mapping

def _cik_from_ticker(ticker: str) -> int:
    m = _load_cik_map()
    t = ticker.upper().strip()
    if t not in m:
        raise ValueError(f"CIK not found for {ticker}")
    return int(m[t])

def _fetch_companyfacts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    r = requests.get(url, headers=_user_agent_headers(), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    time.sleep(PAUSE_SEC)
    return r.json()

def _strip_ns(name: Optional[str]) -> str:
    if not name: return ""
    return re.sub(r".*:", "", name)

def _clean_member_label(lbl: Optional[str], member: Optional[str]) -> str:
    cand = lbl or _strip_ns(member) or ""
    cand = re.sub(r"\s*(Member|Segment)$", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\b(?:[A-Z]\s+){1,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), cand)
    cand = re.sub(r"(?<!^)(?=[A-Z])", " ", cand).strip()
    return cand

def _axis_to_type(axis: str) -> Optional[str]:
    base = _strip_ns(axis)
    if not base:
        return None
    return AXIS_NORMALIZER.get(base, base)

def _iter_fact_items(fact: dict) -> List[dict]:
    out: List[dict] = []
    for unit, arr in (fact.get("units") or {}).items():
        if not str(unit).upper().startswith("USD"):
            continue
        for it in arr or []:
            out.append(it)
    return out

def _year_from_item(it: dict) -> Optional[str]:
    if it.get("fy"):
        return str(it.get("fy"))
    end = it.get("end")
    if isinstance(end, str) and len(end) >= 4 and end[:4].isdigit():
        return end[:4]
    return None

def _coerce_segments_list(segments) -> List[dict]:
    if not segments:
        return []
    if isinstance(segments, list):
        out = []
        for seg in segments:
            if isinstance(seg, dict) and ("dim" in seg or "axis" in seg or "member" in seg):
                out.append(seg)
        return out
    if isinstance(segments, dict):
        out = []
        for k, v in segments.items():
            if isinstance(v, dict):
                out.append({"dim": v.get("dim") or k, "member": v.get("member"), "memberLabel": v.get("memberLabel")})
            else:
                out.append({"dim": k, "member": v, "memberLabel": ""})
        return out
    return []

def _extract_axes_members(segments_raw) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for seg in _coerce_segments_list(segments_raw):
        axis_raw = seg.get("dim") or seg.get("axis") or ""
        axis = _axis_to_type(axis_raw)
        if not axis:
            continue
        label = _clean_member_label(seg.get("memberLabel"), seg.get("member"))
        if not label:
            continue
        out.append((axis, label))
    return out

def _is_revenue_like(base_tag: str) -> bool:
    t = base_tag.lower()
    if any(tok in t for tok in _NEG_TOKENS):
        return False
    return t in {s.lower() for s in REVENUE_BASE_TAGS} or bool(_REV_SEGMENT_RE.search(t))


def _is_opincome_like(base_tag: str) -> bool:
    return bool(_OPINC_LIKE_RE.search(base_tag))

def _collect_items(kind: str, all_facts: dict) -> List[dict]:
    items: List[dict] = []
    for tag, fact in (all_facts or {}).items():
        base = _strip_ns(tag)
        if kind == "rev":
            if not _is_revenue_like(base):
                continue
        elif kind == "op":
            if not _is_opincome_like(base):
                continue
        else:
            continue
        items.extend(_iter_fact_items(fact))
    return items

def _harvest_tag_multi(all_items: List[dict]) -> Dict[Tuple[str, str, str], float]:
    """Aggregate items by axis combination, avoiding double counting.

    If a fact has multiple axes, those axes are joined into a single
    composite key (e.g. "ProductsAndServicesAxis+GeographicalAreasAxis") and
    the member labels are joined with " | " so the fact contributes only once
    to the totals.
    """
    agg: Dict[Tuple[str, str, str], float] = {}
    for it in all_items:
        segs = it.get("segments") or it.get("segment")
        axes = _extract_axes_members(segs)
        if not axes:
            continue
        year = _year_from_item(it)
        if not year:
            continue
        try:
            val = float(it.get("val"))
        except Exception:
            continue
        axes_sorted = sorted(axes, key=lambda x: x[0])
        axis_key = "+".join(a for a, _ in axes_sorted)
        label_key = " | ".join(lbl for _, lbl in axes_sorted)
        key = (axis_key, label_key, str(year))
        agg[key] = agg.get(key, 0.0) + val
    return agg

def _total_company_revenue(all_items: List[dict]) -> Dict[str, float]:
    """Aggregate unsegmented revenue items by year."""
    totals: Dict[str, float] = {}
    for it in all_items:
        segs = it.get("segments") or it.get("segment")
        if _coerce_segments_list(segs):
            continue  # skip segmented facts
        year = _year_from_item(it)
        if not year:
            continue
        try:
            val = float(it.get("val"))
        except Exception:
            continue
        totals[str(year)] = totals.get(str(year), 0.0) + val
    return totals

def get_segment_data(ticker: str) -> pd.DataFrame:
    cik = _cik_from_ticker(ticker)
    facts = _fetch_companyfacts(cik)
    all_facts = facts.get("facts") or {}

    rev_items = _collect_items("rev", all_facts)
    op_items  = _collect_items("op",  all_facts)

    rev_agg = _harvest_tag_multi(rev_items)
    op_agg  = _harvest_tag_multi(op_items)
    totals_company = _total_company_revenue(rev_items)

    keys = set(rev_agg.keys()) | set(op_agg.keys())
    if not keys:
        return pd.DataFrame(columns=["Segment","Year","Revenue","OpIncome","AxisType"])

    rows = []
    for axis, seg, year in keys:
        rows.append({
            "AxisType": axis,
            "Segment": seg,
            "Year": str(year),
            "Revenue": rev_agg.get((axis, seg, year), float("nan")),
            "OpIncome": op_agg.get((axis, seg, year), float("nan")),
        })

    df = pd.DataFrame(rows)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["OpIncome"] = pd.to_numeric(df["OpIncome"], errors="coerce")
    df = df[df["Segment"].astype(str).str.strip() != ""].copy()

    # Validate that segment totals roughly match reported company revenue
    try:
        seg_totals = df.groupby("Year")["Revenue"].sum(min_count=1).to_dict()
        for yr, tot in totals_company.items():
            seg = seg_totals.get(yr)
            if seg is not None and abs(seg - tot) > 1.0:  # tolerance for rounding
                print(f"[segments] WARNING {ticker} {yr}: segments {seg} != reported {tot}")
    except Exception:
        pass

    return df
