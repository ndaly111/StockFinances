#!/usr/bin/env python3
"""
generate_segment_tables.py
Build ONE combined HTML file per ticker with a section per axis (Products/Services, Regions, etc.)
Canonical path: charts/<TICKER>/<TICKER>_segments_table.html

Section layout:
  <h3>{Axis Name}</h3>
  <div class="table-wrap"><table class="seg-table"> ... </table></div>
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import re, pandas as pd

from segment_formatting_helpers import _humanize_segment_name, _to_float
from sec_segment_data_arelle import get_segment_data

OUTPUT_DIR = Path("charts")

HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

STYLE = """
<style>
.seg-table { width:100%; border-collapse:collapse; font-family:Arial,sans-serif; font-size:14px; }
.seg-table th, .seg-table td { padding:6px 8px; border-bottom:1px solid #eee; white-space:nowrap; }
.seg-table thead th { position:sticky; top:0; background:#fff; z-index:1; border-bottom:1px solid #ddd; }
.table-wrap { overflow:auto; max-width:100%; }
.table-note { font-size:12px; color:#666; margin:6px 0 8px; }
</style>
""".strip()

def _norm_axis_label(axis: Optional[str]) -> str:
    s = (axis or "").strip()
    parts = [p for p in s.split("+") if p]
    labels: List[str] = []
    for part in parts:
        p = re.sub(r".*:", "", part)
        p = p.replace("Axis", "")
        p = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
        p = p.replace("_", " ").strip()
        if not p:
            p = "Unlabeled Axis"
        p = p.replace("Geographical Areas", "Regions")
        p = p.replace("Geographical Region", "Regions")
        p = p.replace("Domestic And Foreign", "Domestic vs Foreign")
        p = p.replace("Products And Services", "Products / Services")
        p = re.sub(r"\s+", " ", p)
        labels.append(p.title())
    return " & ".join(labels) if labels else "Unlabeled Axis"

def _choose_scale(max_abs_value: float):
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
    try: val = float(x)/float(div)
    except: return "–"
    if abs(val) >= 100: fmt = "{:,.0f}" if unit == "$" else "{:.0f}"
    elif abs(val) >= 10: fmt = "{:,.1f}" if unit == "$" else "{:.1f}"
    else:                fmt = "{:,.2f}" if unit == "$" else "{:.2f}"
    s = fmt.format(val)
    return f"${s}" if unit == "$" else f"{s}{unit[-1]}"

def _last3_plus_ttm(all_years: List[str]) -> List[str]:
    nums = sorted({int(y) for y in all_years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(all_years):
        out.append("TTM")
    return out

def _pivot(df: pd.DataFrame, col: str, years: List[str]) -> pd.DataFrame:
    p = df[df["Year"].isin(years)].pivot_table(index="Segment", columns="Year", values=col, aggfunc="sum")
    return p.reindex(columns=[y for y in years if y in p.columns])

def _render_axis_section(axis_label: str, rev_p: pd.DataFrame, oi_p: pd.DataFrame) -> str:
    if rev_p.empty and oi_p.empty:
        return f"<h3>{axis_label}</h3><div class='table-wrap'><p>No data for this axis.</p></div>"
    cols = list(rev_p.columns)
    last = "TTM" if "TTM" in cols else (cols[-1] if cols else None)
    if last:
        if last in rev_p.columns:
            rev_p = rev_p[rev_p[last].notna()]
        oi_p = oi_p.reindex(index=rev_p.index)
        hide = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
        rev_p = rev_p[~hide]; oi_p = oi_p.reindex(index=rev_p.index)
        if last in rev_p.columns:
            neg = rev_p[last] < 0
            rev_p = rev_p[~neg]; oi_p = oi_p.reindex(index=rev_p.index)
        rev_p = rev_p.sort_values(by=last, ascending=False)
        oi_p  = oi_p.loc[rev_p.index]
    max_val = pd.concat([rev_p, oi_p]).abs().max().max()
    div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)
    out = pd.DataFrame(index=rev_p.index)
    for y in rev_p.columns:
        out[f"{y} Rev ({unit})"] = rev_p.get(y)
        if y in oi_p.columns and not oi_p[y].isna().all():
            out[f"{y} OI ({unit})"] = oi_p.get(y)
    if "TTM" in rev_p.columns:
        tot = rev_p["TTM"].sum(skipna=True)
        if tot:
            out["% of Total (TTM)"] = (rev_p["TTM"] / tot) * 100.0
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
        return STYLE + "\n" + f"<div class='table-note'>{ticker} — No segment data available. Updated {updated}</div>"
    df = df.copy()
    df["Segment"]  = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"]     = df["Year"].astype(str)
    df["Revenue"]  = df["Revenue"].map(_to_float).astype(float)
    df["OpIncome"] = df["OpIncome"].map(_to_float).astype(float)
    years = _last3_plus_ttm(df["Year"].tolist())
    if "AxisType" not in df.columns or df["AxisType"].isna().all():
        df["AxisType"] = "UnlabeledAxis"
    sections = []
    for axis_value, sub in df.groupby("AxisType", dropna=False, sort=False):
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

def generate_segment_table_for_ticker(ticker: str, charts_dir: Path = OUTPUT_DIR) -> Path:
    out_dir = charts_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_segment_data(ticker)
    html = _build_combined_html(ticker, df)
    out_file = out_dir / f"{ticker}_segments_table.html"  # canonical
    out_file.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out_file}")
    return out_file
