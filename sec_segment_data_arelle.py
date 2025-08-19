# sec_segment_data_arelle.py — extractor returning AxisType
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

REVENUE_TAGS = [
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:SalesRevenueNet",
    "us-gaap:Revenues",
]
OPINC_TAGS = [
    "us-gaap:OperatingIncomeLoss",
]

AXIS_NORMALIZER = {
    "StatementGeographicalAxis": "GeographicalAreasAxis",
    "GeographicalAreasAxis": "GeographicalAreasAxis",
    "GeographicalRegionsAxis": "GeographicalRegionsAxis",
    "GeographicalRegionAxis": "GeographicalRegionsAxis",
    "DomesticAndForeignAxis": "DomesticAndForeignAxis",
    "CountryAxis": "CountryAxis",

    "ProductOrServiceAxis": "ProductsAndServicesAxis",
    "ProductsAndServicesAxis": "ProductsAndServicesAxis",
    "ProductLineAxis": "ProductLineAxis",
    "ProductAxis": "ProductAxis",
    "ProductCategoryAxis": "ProductCategoryAxis",
    "ProductCategoriesAxis": "ProductCategoryAxis",

    "OperatingSegmentsAxis": "OperatingSegmentsAxis",
    "BusinessSegmentsAxis": "OperatingSegmentsAxis",
    "ReportableSegmentsAxis": "OperatingSegmentsAxis",
    "SegmentsAxis": "OperatingSegmentsAxis",

    "MajorCustomersAxis": "MajorCustomersAxis",
    "SignificantCustomersAxis": "MajorCustomersAxis",

    "SalesChannelsAxis": "SalesChannelsAxis",
    "DistributionChannelsAxis": "SalesChannelsAxis",
}

def _user_agent_headers() -> Dict[str, str]:
    email = None
    for env_name in SEC_HEADERS_EMAIL_ENV:
        v = os.environ.get(env_name)
        if v:
            email = v.strip()
            break
    ua = f"ndaly-segments/1.0 ({email or 'email@domain.com'})"
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
    base = AXIS_NORMALIZER.get(_strip_ns(axis)) or _strip_ns(axis)
    return AXIS_NORMALIZER.get(base, base) if base else None

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

def _best_seg_axis(segments: List[dict]) -> Optional[Tuple[str, str, str]]:
    """
    Pick a 'best' axis/member from the item's segments.
    Prefer product/service > geography > operating > others.
    Returns (AxisType, memberLabel, member)
    """
    rank = {
        "ProductsAndServicesAxis": 0,
        "ProductLineAxis": 0,
        "ProductAxis": 0,
        "ProductCategoryAxis": 0,
        "GeographicalAreasAxis": 1,
        "GeographicalRegionsAxis": 1,
        "DomesticAndForeignAxis": 1,
        "CountryAxis": 1,
        "OperatingSegmentsAxis": 2,
        "MajorCustomersAxis": 3,
        "SalesChannelsAxis": 3,
    }
    best = None
    best_rank = 999
    for seg in segments or []:
        axis_raw = seg.get("dim") or seg.get("axis") or ""
        axis = _axis_to_type(axis_raw)
        if not axis:
            continue
        r = rank.get(axis, 9)
        if r < best_rank:
            best_rank = r
            best = (axis, seg.get("memberLabel") or "", seg.get("member") or "")
    return best

def _harvest_tag(all_items: List[dict]) -> Dict[Tuple[str, str, str], float]:
    """
    Aggregate numeric values by (AxisType, SegmentLabel, Year).
    """
    agg: Dict[Tuple[str, str, str], float] = {}
    for it in all_items:
        segments = it.get("segments") or it.get("segment") or []
        if isinstance(segments, dict):
            tmp = []
            for k, v in segments.items():
                if isinstance(v, dict):
                    tmp.append({"dim": v.get("dim") or k, "member": v.get("member"), "memberLabel": v.get("memberLabel")})
                else:
                    tmp.append({"dim": k, "member": v, "memberLabel": ""})
            segments = tmp
        choose = _best_seg_axis(segments)
        if not choose:
            continue
        axis, memberLabel, member = choose
        year = _year_from_item(it)
        if not year:
            continue
        seg_label = _clean_member_label(memberLabel, member)
        try:
            val = float(it.get("val"))
        except Exception:
            continue
        key = (axis, seg_label, str(year))
        agg[key] = agg.get(key, 0.0) + val
    return agg

def get_segment_data(ticker: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Segment, Year, Revenue, OpIncome, AxisType.
    Aggregates by AxisType & Segment for each fiscal year present in companyfacts.
    """
    cik = _cik_from_ticker(ticker)
    facts = _fetch_companyfacts(cik)
    all_facts = facts.get("facts") or {}
    rev_items: List[dict] = []
    for tag in REVENUE_TAGS:
        f = all_facts.get(tag)
        if not f: 
            continue
        rev_items.extend(_iter_fact_items(f))
    op_items: List[dict] = []
    for tag in OPINC_TAGS:
        f = all_facts.get(tag)
        if not f:
            continue
        op_items.extend(_iter_fact_items(f))

    rev_agg = _harvest_tag(rev_items)
    op_agg  = _harvest_tag(op_items)

    keys = set(rev_agg.keys()) | set(op_agg.keys())
    rows = []
    for axis, seg, year in keys:
        rows.append({
            "AxisType": axis,
            "Segment": seg,
            "Year": str(year),
            "Revenue": rev_agg.get((axis, seg, year), float("nan")),
            "OpIncome": op_agg.get((axis, seg, year), float("nan")),
        })
    if not rows:
        return pd.DataFrame(columns=["Segment","Year","Revenue","OpIncome","AxisType"])
    df = pd.DataFrame(rows)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["OpIncome"] = pd.to_numeric(df["OpIncome"], errors="coerce")
    df = df[df["Segment"].astype(str).str.strip() != ""].copy()
    return df
