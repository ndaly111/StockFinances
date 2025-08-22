# sec_segment_data_arelle.py — extractor returning AxisType (multi-axis; broad revenue tag coverage)
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd

# -------- configuration --------
SEC_CIK_CACHE = Path(".sec_cik_cache.json")
SEC_HEADERS_EMAIL_ENV = ("SEC_EMAIL", "EMAIL_FOR_SEC")  # pick one if set
REQUEST_TIMEOUT = 20
PAUSE_SEC = 0.2  # polite delay between requests

# Canonical revenue/operating-income tags seen across filers
REVENUE_BASE_TAGS = {
    # Very common
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "Revenues",
    "NetSales",
    # Occasional variants we want to pick up
    "ProductRevenue",
    "ServiceRevenue",
    "OperatingRevenue",
}
OPINC_BASE_TAGS = {"OperatingIncomeLoss"}

# Normalize common axis names to a canonical form
AXIS_NORMALIZER = {
    # Geography
    "StatementGeographicalAxis": "GeographicalAreasAxis",
    "GeographicalAreasAxis": "GeographicalAreasAxis",
    "GeographicalRegionsAxis": "GeographicalRegionsAxis",
    "GeographicalRegionAxis": "GeographicalRegionsAxis",
    "DomesticAndForeignAxis": "DomesticAndForeignAxis",
    "CountryAxis": "CountryAxis",
    # Product/Service
    "ProductOrServiceAxis": "ProductsAndServicesAxis",
    "ProductsAndServicesAxis": "ProductsAndServicesAxis",
    "ProductLineAxis": "ProductLineAxis",
    "ProductAxis": "ProductAxis",
    "ProductCategoryAxis": "ProductCategoryAxis",
    "ProductCategoriesAxis": "ProductCategoryAxis",
    # Operating/reportable segments
    "OperatingSegmentsAxis": "OperatingSegmentsAxis",
    "BusinessSegmentsAxis": "OperatingSegmentsAxis",
    "ReportableSegmentsAxis": "OperatingSegmentsAxis",
    "SegmentsAxis": "OperatingSegmentsAxis",
    # Customers / Channels
    "MajorCustomersAxis": "MajorCustomersAxis",
    "SignificantCustomersAxis": "MajorCustomersAxis",
    "SalesChannelsAxis": "SalesChannelsAxis",
    "DistributionChannelsAxis": "SalesChannelsAxis",
}
AXIS_WHITELIST = set(AXIS_NORMALIZER.values())

def _user_agent_headers() -> Dict[str, str]:
    email = None
    for env_name in SEC_HEADERS_EMAIL_ENV:
        v = os.environ.get(env_name)
        if v:
            email = v.strip()
            break
    ua = f"ndaly-segments/1.2 ({email or 'email@domain.com'})"
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
    # drop trailing Member/Segment, join spaced initials (e.g., "U S"), split camel case
    cand = re.sub(r"\s*(Member|Segment)$", "", cand, flags=re.IGNORECASE)
    cand = re.sub(r"\b(?:[A-Z]\s+){1,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), cand)
    cand = re.sub(r"(?<!^)(?=[A-Z])", " ", cand).strip()
    return cand

def _axis_to_type(axis: str) -> Optional[str]:
    base = AXIS_NORMALIZER.get(_strip_ns(axis)) or _strip_ns(axis)
    return AXIS_NORMALIZER.get(base, base) if base else None

def _iter_fact_items(fact: dict) -> List[dict]:
    out: List[dict] = []
    for unit, arr in (fact.get("units") or {}).items():
        if not str(unit).upper().startswith("USD"):  # USD/… variations
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
        if not axis or axis not in AXIS_WHITELIST:
            continue
        label = _clean_member_label(seg.get("memberLabel"), seg.get("member"))
        if not label:
            continue
        out.append((axis, label))
    return out

# ---------- revenue concept selection (broad but safe) ----------
_NEG_TOKENS = ("cost", "cogs", "expense", "gain", "loss", "grossprofit", "tax", "deferred", "impair", "interest")
def _is_revenue_like(base_tag: str) -> bool:
    """Accept common revenue/sales concepts; avoid CostOfSales, etc."""
    t = base_tag.lower()
    if any(tok in t for tok in _NEG_TOKENS):
        return False
    return (
        t in {s.lower() for s in REVENUE_BASE_TAGS}
        or t.endswith("revenuenet")
        or t.endswith("revenues")
        or t.endswith("revenue")
        or t == "netsales"
        or "salesrevenue" in t
    )

def _collect_items(kind: str, all_facts: dict) -> List[dict]:
    """
    kind: 'rev' or 'op'
    """
    items: List[dict] = []
    for tag, fact in (all_facts or {}).items():
        base = _strip_ns(tag)
        if kind == "rev":
            if not _is_revenue_like(base):
                continue
        else:  # op inc
            if base not in OPINC_BASE_TAGS:
                continue
        items.extend(_iter_fact_items(fact))
    return items

def _harvest_tag_multi(all_items: List[dict]) -> Dict[Tuple[str, str, str], float]:
    """
    Aggregate numeric values by **each** axis present, i.e., (AxisType, SegmentLabel, Year).
    If a single fact has multiple axes, it contributes to each axis section.
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
        for axis, label in axes:
            key = (axis, label, str(year))
            agg[key] = agg.get(key, 0.0) + val
    return agg

def get_segment_data(ticker: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Segment, Year, Revenue, OpIncome, AxisType.
    Aggregates by **all** AxisType & Segment for each fiscal year.
    """
    cik = _cik_from_ticker(ticker)
    facts = _fetch_companyfacts(cik)
    all_facts = facts.get("facts") or {}

    rev_items = _collect_items("rev", all_facts)
    op_items  = _collect_items("op",  all_facts)

    rev_agg = _harvest_tag_multi(rev_items)
    op_agg  = _harvest_tag_multi(op_items)

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
    return df
