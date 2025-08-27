#!/usr/bin/env python3
"""
generate_segment_tables.py
Build ONE combined HTML file per ticker with a section per axis (Products/Services, Regions, etc.)
Path written: charts/{TICKER}_segments.html

Each section:
  <h3>{Axis Name}</h3>
  <div class="table-wrap"><table ...> ... </table></div>

This is layout-safe: it uses minimal inline CSS only for the table block,
so it won’t alter your site’s existing styles.

Requires: pandas, matplotlib (for earlier chart step), requests (used by sec fetcher)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import math, re
import pandas as pd

from sec_segment_data_arelle import get_segment_data

OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# hide elimination / recon / unallocated lines
HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

# ---------- helpers ----------
def _humanize_segment_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw:
        return str(raw)
    name = str(raw)
    name = re.sub(r"\s*(Member|Segment)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b([A-Z])\s+([A-Z])\b", r"\1\2", name)
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split())

def _norm_axis_label(axis: Optional[str]) -> str:
    s = (axis or "").strip()
    s = re.sub(r".*:", "", s)
    s = s.replace("Axis", "")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").strip()
    if not s:
        return "Unlabeled Axis"
    s = s.replace("Geographical Areas", "Regions")
    s = s.replace("Geographical Region", "Regions")
    s = s.replace("Domestic And Foreign", "Domestic vs Foreign")
    s = s.replace("Products And Services", "Products / Services")
    s = re.sub(r"\s+", " ", s)
    return s.title()

def _choose_scale(max_abs_value: float) -> Tuple[float, str]:
    if not isinstance(max_abs_value, (int, float)) or pd.isna(max_abs_value) or max_abs_value == 0:
        return (1.0, "$")
    v = abs(max_abs_value)
    if v >= 1e12: return (1e12, "$T")
    if v >= 1e9:  return (1e9,  "$B")
    if v >= 1e6:  return (1e6,  "$M")
    if v >= 1e3:  return (1e3,  "$K")
    return (1.0, "$")

def _fmt_scaled(x, div, unit) -> str:
    if pd.isna(x): return "–"
    try:
        val = float(x) / float(div)
    except Exception:
        return "–"
    if abs(val) >= 100: fmt = "{:,.0f}" if unit == "$" else "{:.0f}"
    elif abs(val) >= 10: fmt = "{:,.1f}" if unit == "$" else "{:.1f}"
    else:                fmt = "{:,.2f}" if unit == "$" else "{:.2f}"
    s = fmt.format(val)
    return f"${s}" if unit == "$" else f"{s}{unit[-1]}"

def _last3_plus_ttm(all_years: List[str]) -> List[str]:
    """Keep last 3 numeric years; include TTM if present at the end."""
    nums = sorted({int(y) for y in all_years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(all_years):
        out.append("TTM")
    return out

# ---------- HTML assembly ----------
STYLE = """
<style>
.seg-table { width:100%; border-collapse:collapse; font-family:Arial,sans-serif; font-size:14px; }
.seg-table th, .seg-table td { padding:6px 8px; border-bottom:1px solid #eee; white-space:nowrap; }
.seg-table thead th { position:sticky; top:0; background:#fff; z-index:1; border-bottom:1px solid #ddd; }
.table-wrap { overflow:auto; max-width:100%; }
.table-note { font-size:12px; color:#666; margin:6px 0 8px; }
</style>
""".strip()

def _pivot(df: pd.DataFrame, col: str, years: List[str]) -> pd.DataFrame:
    p = df[df["Year"].isin(years)].pivot_table(index="Segment", columns="Year", values=col, aggfunc="sum")
    return p.reindex(columns=[y for y in years if y in p.columns])

def _render_axis_section(axis_label: str, rev_p: pd.DataFrame, oi_p: pd.DataFrame) -> str:
    if rev_p.empty and oi_p.empty:
        return f"<h3>{axis_label}</h3><div class='table-wrap'><p>No data for this axis.</p></div>"

    # sort on latest available column (prefer TTM, otherwise last year)
    cols = list(rev_p.columns)
    last = "TTM" if "TTM" in cols else (cols[-1] if cols else None)
    if last:
        if last in rev_p.columns:
            rev_p = rev_p[rev_p[last].notna()]
        oi_p = oi_p.reindex(index=rev_p.index)

        # drop hidden/negative rows
        mask_hide = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
        rev_p = rev_p[~mask_hide]
        oi_p  = oi_p.reindex(index=rev_p.index)
        if last in rev_p.columns:
            mask_neg = rev_p[last] < 0
            rev_p = rev_p[~mask_neg]
            oi_p  = oi_p.reindex(index=rev_p.index)

        rev_p = rev_p.sort_values(by=last, ascending=False)
        oi_p  = oi_p.loc[rev_p.index]

    # scale/unit selection across Rev/OpInc
    max_val = pd.concat([rev_p, oi_p]).abs().max().max()
    div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

    # interleave columns: each year shows Rev (and OI if present)
    out = pd.DataFrame(index=rev_p.index)
    for y in rev_p.columns:
        out[f"{y} Rev ({unit})"] = rev_p.get(y)
        if y in oi_p.columns and not oi_p.empty and not oi_p[y].isna().all():
            out[f"{y} OI ({unit})"] = oi_p.get(y)

    # % of Total (TTM)
    if "TTM" in rev_p.columns:
        total_ttm = rev_p["TTM"].sum(skipna=True)
        if total_ttm:
            out["% of Total (TTM)"] = (rev_p["TTM"] / total_ttm) * 100.0

    # format
    for c in out.columns:
        if c == "% of Total (TTM)":
            out[c] = out[c].map(lambda x: f"{float(x):.1f}%" if pd.notnull(x) else "–")
        else:
            out[c] = out[c].map(lambda x, d=div, u=unit: _fmt_scaled(x, d, u))
    for c in [c for c in out.columns if c.startswith("TTM ")]:
        out[c] = out[c].map(lambda s: f"<strong>{s}</strong>" if s != "–" else s)

    out.index.name = "Segment"
    html_table = out.reset_index().to_html(index=False, escape=False, classes="seg-table", border=0)
    return f"<h3>{axis_label}</h3>\n<div class='table-wrap'>{html_table}</div>"

def _build_combined_html(ticker: str, df: pd.DataFrame) -> str:
    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if df is None or df.empty:
        return (
            STYLE + "\n" +
            f"<div class='table-note'>{ticker} — No segment data available. Updated {updated}</div>"
        )

    # clean + types
    df = df.copy()
    df["Segment"]  = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"]     = df["Year"].astype(str)
    df["Revenue"]  = pd.to_numeric(df["Revenue"], errors="coerce")
    df["OpIncome"] = pd.to_numeric(df["OpIncome"], errors="coerce")

    years = _last3_plus_ttm(df["Year"].tolist())
    sections = []

    if "AxisType" not in df.columns or df["AxisType"].isna().all():
        df["AxisType"] = "UnlabeledAxis"

    for axis_value, sub in df.groupby("AxisType", dropna=False):
        axis_label = _norm_axis_label(axis_value)
        rev_p = _pivot(sub, "Revenue", years)
        oi_p  = _pivot(sub, "OpIncome", years)
        sections.append(_render_axis_section(axis_label, rev_p, oi_p))

    head = (
        STYLE + "\n" +
        f"<div class='table-note'>{ticker} — Segment Revenue & Operating Income (Last 3 FY + TTM where present). "
        f"Updated {updated}. Source: SEC Inline XBRL</div>"
    )
    return head + "\n" + "\n<hr/>\n".join(sections)

# ---------- public API ----------
def generate_segment_table_for_ticker(ticker: str, charts_dir: Path = OUTPUT_DIR) -> Path:
    charts_dir.mkdir(parents=True, exist_ok=True)
    df = get_segment_data(ticker)
    html = _build_combined_html(ticker, df)
    out_file = charts_dir / f"{ticker}_segments.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out_file}")
    return out_file
