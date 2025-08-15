#!/usr/bin/env python3
"""
generate_segment_charts.py — SEGMENTS v2025-08-12c (multi-view: product + geo + more)

Key features:
• One shared y-axis across all segment charts (handles negatives).
• Compact, scaled HTML pivot tables (last 3 fiscal years + TTM).
• Picks a single unit ($, $K, $M, $B, $T) for each table.
• Bold TTM, add “% of Total (TTM)”.
• Writes a single HTML that can show ALL views present (Products, Geography, Operating, Channels, Customers, Other):
    charts/{TICKER}/{TICKER}_segments_table.html
  (and alias copies as before)
• Cleans up duplicate/legacy PNGs.

Generalization:
• Maps XBRL axes → SegType when available (product / geo / operating / channel / customer / other);
  falls back to name heuristics; then applies optional per-ticker overrides via segment_typing.csv.
• Renders every present segment type in a fixed order on one page.
"""

from __future__ import annotations
import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data

VERSION = "SEGMENTS v2025-08-12c"

# ─────────────────────────── XBRL axis → segment-type mapping ───────────────────────────
AXIS_TO_SEGTYPE = {
    # geography
    "GeographicalAreasAxis": "geo",
    "StatementGeographicalAxis": "geo",
    "GeographicalRegionsAxis": "geo",
    "GeographicalRegionAxis": "geo",
    "DomesticAndForeignAxis": "geo",
    "CountryAxis": "geo",

    # products/services
    "ProductOrServiceAxis": "product",
    "ProductsAndServicesAxis": "product",
    "ProductLineAxis": "product",
    "ProductAxis": "product",
    "ProductCategoryAxis": "product",
    "ProductCategoriesAxis": "product",

    # operating/reportable segments
    "OperatingSegmentsAxis": "operating",
    "BusinessSegmentsAxis": "operating",
    "ReportableSegmentsAxis": "operating",
    "SegmentsAxis": "operating",

    # channels/customers
    "SalesChannelsAxis": "channel",
    "DistributionChannelsAxis": "channel",
    "MajorCustomersAxis": "customer",
    "SignificantCustomersAxis": "customer",
}

# Which section types to render, and in what order/title
SECTION_ORDER  = ["product", "geo", "operating", "channel", "customer", "other"]
SECTION_TITLES = {
    "product":   "Products / Categories",
    "geo":       "Geography",
    "operating": "Operating Segments",
    "channel":   "Sales Channels",
    "customer":  "Customers",
    "other":     "Other / Unclassified",
}

# ─────────────────────────── utilities ───────────────────────────

def read_tickers(csv_path: Path) -> List[str]:
    """Read a CSV of tickers and return a list of uppercase ticker symbols."""
    if not csv_path.is_file():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    cols = [c for c in df.columns if c.lower() == "ticker"]
    if not cols:
        return []
    return [str(t).upper().strip() for t in df[cols[0]].dropna().tolist()]

def sort_years(years: List[str]) -> List[str]:
    """Sort years so that integers come before strings, TTM is last."""
    def key(y: str) -> Tuple[int, int | str]:
        if y == "TTM":
            return (2, 0)
        try:
            return (0, int(y))
        except Exception:
            return (1, y)
    return [y for _, y in sorted([(key(y), y) for y in years], key=lambda x: x[0])]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _humanize_segment_name(raw: str) -> str:
    """Initial cleanup of raw segment names: split CamelCase and apply title-case fixes."""
    if not isinstance(raw, str) or not raw:
        return raw
    name = raw.replace("SegmentMember", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    fixes = {
        "Greater China": "Greater China",
        "Rest Of Asia Pacific": "Rest of Asia Pacific",
        "North America": "North America",
        "Latin America": "Latin America",
        "United States": "United States",
        "Middle East": "Middle East",
        "Asia Pacific": "Asia Pacific",
        "Americas": "Americas",
        "Europe": "Europe",
        "Japan": "Japan",
        "China": "China",
    }
    title = " ".join(w if w.isupper() else w.capitalize() for w in name.split())
    return fixes.get(title, title)

def _normalize_segment_tokens(s: str) -> str:
    """Remove trailing Member/Segment, join spaced initials, collapse whitespace."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"\s*(Member|Segment)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:[A-Z]\s+){1,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -–—").strip()
    return s

def _to_float(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return pd.NA

def _choose_scale(max_abs_value: float) -> Tuple[float, str]:
    """Return a divisor and a unit label based on the magnitude of the value."""
    if not isinstance(max_abs_value, (int, float)) or math.isnan(max_abs_value) or max_abs_value == 0:
        return (1.0, "$")
    v = abs(max_abs_value)
    if v >= 1e12:
        return (1e12, "$T")
    if v >= 1e9:
        return (1e9, "$B")
    if v >= 1e6:
        return (1e6, "$M")
    if v >= 1e3:
        return (1e3, "$K")
    return (1.0, "$")

def _fmt_scaled(x, div, unit) -> str:
    """Format a number after dividing by div, append unit suffix."""
    if pd.isna(x):
        return "–"
    try:
        val = float(x) / float(div)
    except Exception:
        return "–"
    if abs(val) >= 100:
        fmt = "{:,.0f}" if unit == "$" else "{:.0f}"
    elif abs(val) >= 10:
        fmt = "{:,.1f}" if unit == "$" else "{:.1f}"
    else:
        fmt = "{:,.2f}" if unit == "$" else "{:.2f}"
    s = fmt.format(val)
    return f"${s}" if unit == "$" else f"{s}{unit[-1]}"

def _last3_plus_ttm(years: List[str]) -> List[str]:
    """Return the last 3 numeric years (if >3) plus TTM if present."""
    nums = sorted({int(y) for y in years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(years):
        out.append("TTM")
    return out

def _safe_seg_filename(seg: str) -> str:
    return seg.replace("/", "_").replace(" ", "_")

# ───────────────────── optional per-ticker overrides ─────────────────────

def _load_segment_typing(path: Path = Path("segment_typing.csv")) -> pd.DataFrame:
    """Load a per-ticker segment typing override file."""
    if not path.is_file():
        return pd.DataFrame(columns=["Ticker", "Segment", "SegType"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Segment", "SegType"])
    req = {"Ticker", "Segment", "SegType"}
    if not req.issubset({c.strip() for c in df.columns}):
        return pd.DataFrame(columns=["Ticker", "Segment", "SegType"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Segment"] = df["Segment"].astype(str).str.strip()
    df["SegType"] = df["SegType"].astype(str).str.lower().str.strip()
    valid = {"geo", "product", "operating", "channel", "customer", "other"}
    df = df[df["SegType"].isin(valid)]
    return df[["Ticker", "Segment", "SegType"]]

# ───────────────────── detection: geography vs product vs other ──────────

_GEO_WORDS = re.compile(
    r"(americas|north america|latin america|south america|europe|emea|middle east|"
    r"africa|apac|asia pacific|greater china|china|japan|india|australia|canada|uk|korea|taiwan|"
    r"rest of asia|rest of world|international|global)", re.IGNORECASE
)

_PRODUCT_WORDS = re.compile(
    r"(service|services|software|hardware|devices?|platforms?|subscriptions?|"
    r"advertising|payments?|wearables|accessories|gaming|cloud|data|media|content|"
    r"iphones?|ipads?|macs?)",
    re.IGNORECASE,
)

def _infer_segtype_by_label(seg: str) -> str:
    """Heuristic classification of segment names."""
    s = seg or ""
    if _GEO_WORDS.search(s):
        return "geo"
    if _PRODUCT_WORDS.search(s):
        return "product"
    # Unknown: treat as 'operating' if it looks like a business unit, else 'other'
    if re.search(
        r"(segment|group|division|solutions|industrial|upstream|downstream|energy|networks?)",
        s,
        re.IGNORECASE,
    ):
        return "operating"
    return "other"

def _infer_segtype_by_axis(row: pd.Series) -> Optional[str]:
    """
    Map XBRL axis/dimension names → SegType using AXIS_TO_SEGTYPE when available.
    Looks across common columns if present: Axis, AxisName, Dimension, DimensionName, XBRLAxis.
    """
    axis_cols = [c for c in row.index if str(c).lower() in {
        "axis", "axisname", "dimension", "dimensionname", "xbrlaxis", "xbrldimension"
    }]
    for c in axis_cols:
        val = str(row[c]).strip()
        # some datasets include full QName like us-gaap:GeographicalAreasAxis
        if ":" in val:
            val = val.split(":", 1)[1]
        if val in AXIS_TO_SEGTYPE:
            return AXIS_TO_SEGTYPE[val]
    return None

# ───────────────────── cleanup helper & filters ───────────────────────

def _cleanup_segment_pngs(out_dir: Path, ticker: str, keep_files: List[str]) -> None:
    """Remove legacy/duplicate PNGs after writing canonical set."""
    try:
        for generic in ("segment_performance.png", f"{ticker}_segment_performance.png"):
            p = out_dir / generic
            if p.exists():
                p.unlink()
        keep = set(keep_files)
        for p in out_dir.glob(f"{ticker}_*.png"):
            if p.name not in keep:
                p.unlink()
    except Exception as e:
        print(f"[{VERSION}] WARN: cleanup in {out_dir} hit an issue: {e}")

# hide rows like Eliminations / Intersegment / Unallocated / Corporate & Other, etc.
HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

# ───────────────────── main per-ticker routine ────────────────────

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """Generate charts and a combined HTML that can include multiple segment tables."""
    try:
        df = get_segment_data(ticker)
    except Exception as fetch_err:
        print(f"[{VERSION}] Error fetching segment data for {ticker}: {fetch_err}")
        ensure_dir(out_dir)
        (out_dir / f"{ticker}_segments_table.html").write_text(
            f"<p>Error fetching segment data for {ticker}: {fetch_err}</p>", encoding="utf-8"
        )
        return

    ensure_dir(out_dir)

    if df is None or df.empty:
        (out_dir / f"{ticker}_segments_table.html").write_text(
            f"<p>No segment data available for {ticker}.</p>", encoding="utf-8"
        )
        return

    # Clean + normalize
    df = df.copy()
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Segment"] = df["Segment"].map(_normalize_segment_tokens)
    df["Year"] = df["Year"].astype(str)
    df["Revenue"] = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    # If OpIncome is completely missing or all zeros across dataset, mark as missing
    _op_all_missing = df["OpIncome"].isna().all() or (df["OpIncome"].fillna(0) == 0).all()
    if _op_all_missing:
        df["OpIncome"] = pd.NA

    # Add SegType: prefer XBRL axis mapping if available, else label heuristic; then apply overrides
    # Prepare an axis-derived type if any axis columns exist
    has_axis_cols = any(str(c).lower() in {"axis","axisname","dimension","dimensionname","xbrlaxis","xbrldimension"} for c in df.columns)
    if has_axis_cols:
        df["_axis_segtype"] = df.apply(_infer_segtype_by_axis, axis=1)
    else:
        df["_axis_segtype"] = None

    df["SegType"] = df["_axis_segtype"].fillna(df["Segment"].apply(_infer_segtype_by_label))

    overrides = _load_segment_typing()
    if not overrides.empty:
        temp = df.copy()
        temp["Ticker"] = ticker
        temp = temp.merge(overrides, on=["Ticker", "Segment"], how="left", suffixes=("", "_ovr"))
        df["SegType"] = temp["SegType_ovr"].fillna(df["SegType"])

    # Shared y-range across ALL segments for charts (handles negatives)
    all_vals = pd.concat([df["Revenue"].dropna(), df["OpIncome"].dropna()], ignore_index=True)
    if all_vals.empty:
        min_y, max_y = 0.0, 0.0
    else:
        min_y, max_y = float(all_vals.min()), float(all_vals.max())
        if min_y > 0:
            min_y = 0.0
        if max_y < 0:
            max_y = 0.0
    spread = (max_y - min_y)
    margin = spread * 0.1 if spread else 1.0
    min_y_plot, max_y_plot = min_y - margin, max_y + margin

    years_all = sort_years(sorted(set(df["Year"].tolist())))
    years_tbl = _last3_plus_ttm(df["Year"].tolist())
    segments = sorted(set(df["Segment"].tolist()))

    # ── Charts per segment (y in $B) ──
    written_pngs: List[str] = []
    for seg in segments:
        seg_df = df[df["Segment"] == seg]
        revenues = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum() for y in years_all]
        op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum() for y in years_all]

        revenues_b = [0.0 if pd.isna(v) else v / 1e9 for v in revenues]
        op_incomes_b = [0.0 if pd.isna(v) else v / 1e9 for v in op_incomes]
        min_y_plot_b = min_y_plot / 1e9
        max_y_plot_b = max_y_plot / 1e9

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(years_all)))
        w = 0.35
        ax.bar([i - w / 2 for i in x], revenues_b, width=w, label="Revenue")
        ax.bar([i + w / 2 for i in x], op_incomes_b, width=w, label="Operating Income")
        ax.set_xticks(x)
        ax.set_xticklabels(years_all)
        ax.set_ylim(min_y_plot_b, max_y_plot_b)
        ax.set_ylabel("Value ($B)")
        ax.set_title(seg)
        ax.axhline(0, linewidth=0.8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
        plt.tight_layout()

        safe_seg = _safe_seg_filename(seg)
        out_name = f"{ticker}_{safe_seg}.png"
        plt.savefig(out_dir / out_name)
        plt.close(fig)
        written_pngs.append(out_name)

    _cleanup_segment_pngs(out_dir, ticker, written_pngs)

    # ── Generic pivot builder for a subset ──
    def pv(col, sub_df):
        p = sub_df[sub_df["Year"].isin(years_tbl)].pivot_table(
            index="Segment", columns="Year", values=col, aggfunc="sum"
        )
        return p.reindex(columns=[y for y in years_tbl if y in p.columns])

    def build_table(sub_df: pd.DataFrame, title_text: str) -> Optional[str]:
        if sub_df.empty:
            return None

        rev_p = pv("Revenue", sub_df)
        oi_p = pv("OpIncome", sub_df)

        sort_col = "TTM" if "TTM" in rev_p.columns else (rev_p.columns[-1] if len(rev_p.columns) else None)
        if sort_col:
            # keep rows with non-missing latest revenue
            if sort_col in rev_p.columns:
                rev_p = rev_p[rev_p[sort_col].notna()]
            oi_p = oi_p.reindex(index=rev_p.index)

            # drop elimination/reconciliation/intersegment/unallocated/corporate other
            hide_mask = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
            rev_p = rev_p[~hide_mask]
            oi_p = oi_p.reindex(index=rev_p.index)

            # drop negative latest revenue
            neg_mask = rev_p[sort_col] < 0
            rev_p = rev_p[~neg_mask]
            oi_p = oi_p.reindex(index=rev_p.index)

            # sort by latest revenue (largest → smallest)
            rev_p = rev_p.sort_values(by=sort_col, ascending=False)
            oi_p = oi_p.loc[rev_p.index]

        # % mix on filtered data
        pct_series = None
        if "TTM" in rev_p.columns:
            total_ttm = rev_p["TTM"].sum(skipna=True)
            if total_ttm:
                pct_series = (rev_p["TTM"] / total_ttm) * 100.0

        # scale
        max_val = pd.concat([rev_p, oi_p]).abs().max().max()
        div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

        # columns (stable)
        cols: List[Tuple[str, str]] = []
        for y in [c for c in years_tbl if c != "TTM"]:
            cols += [(y, "Rev"), (y, "OI")]
        if "TTM" in years_tbl:
            cols += [("TTM", "Rev"), ("TTM", "OI")]

        out = pd.DataFrame(index=rev_p.index)
        for (y, kind) in cols:
            series = rev_p.get(y) if kind == "Rev" else oi_p.get(y)
            out[f"{y} {'Rev' if kind=='Rev' else 'OI'} ({unit})"] = series

        if pct_series is not None:
            out["% of Total (TTM)"] = pct_series

        # format values
        for c in out.columns:
            if c == "% of Total (TTM)":
                out[c] = out[c].map(lambda x: f"{float(x):.1f}%" if pd.notnull(x) else "–")
            else:
                out[c] = out[c].map(lambda x, d=div, u=unit: _fmt_scaled(x, d, u))
        for ttm_col in [c for c in out.columns if c.startswith("TTM ")]:
            out[ttm_col] = out[ttm_col].map(lambda s: f"<strong>{s}</strong>" if s != "–" else s)

        out.index.name = "Segment"
        out_disp = out.reset_index()
        html = out_disp.to_html(index=False, escape=False, classes="segment-pivot", border=0)

        sec_title = f"<h3 style='margin:10px 0 6px'>{title_text}</h3>"
        return sec_title + f"\n<div class='table-wrap'>{html}</div>\n"

    # ── Build sections dynamically for any segment type present ──
    sections: List[str] = []
    for segtype in SECTION_ORDER:
        sub = df[df["SegType"] == segtype].copy()
        if sub.empty:
            continue
        title = SECTION_TITLES.get(segtype, segtype.title())
        html = build_table(sub, title)
        if html:
            sections.append(html)

    if not sections:
        sections.append("<p>No usable segment pivots after filtering.</p>")

    css = """
<style>
.table-wrap{overflow:auto; max-width:100%;}
.segment-pivot{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px}
.segment-pivot thead th{position:sticky;top:0;background:#fff;z-index:1;border-bottom:1px solid #ddd}
.segment-pivot th,.segment-pivot td{padding:6px 8px;border-bottom:1px solid #f0f0f0}
.segment-pivot tbody tr:nth-child(even){background:#fafafa}
.segment-pivot td,.segment-pivot th{white-space:nowrap}
.segment-pivot td{font-variant-numeric:tabular-nums;text-align:right}
.segment-pivot td:first-child,.segment-pivot th:first-child{text-align:left}
.table-note{font-size:12px;color:#666;margin:6px 0 8px}
</style>
""".strip()

    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    caption = (
        f'<div class="table-note">{VERSION} · {stamp} — Each section uses a single scale. '
        f'TTM is <b>bold</b>. “% of Total (TTM)” uses the visible rows in that section.</div>'
    )

    table_content = f"<!-- {VERSION} -->\n" + css + "\n" + caption + "\n" + "\n<hr/>\n".join(sections)

    canonical = out_dir / f"{ticker}_segments_table.html"}
    aliases = [
        out_dir / "segments_table.html",
        out_dir / "segment_performance.html",
        out_dir / f"{ticker}_segment_performance.html",
    ]

    def write_file(p: Path, content: str):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        p.write_text(content, encoding="utf-8")
        print(f"[{VERSION}] wrote {p} ({p.stat().st_size} bytes)")

    write_file(canonical, table_content)
    for a in aliases:
        write_file(a, table_content)

# ─────────────────────────── CLI wrapper ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate segment charts and tables for a list of tickers.")
    parser.add_argument("--tickers_csv", type=str, default="tickers.csv", help="CSV with a 'Ticker' column")
    parser.add_argument("--output_dir", type=str, default="charts", help="Output directory (default: charts/)")
    args = parser.parse_args()

    print(f"{VERSION} starting…")

    csv_path = Path(args.tickers_csv)
    tickers = read_tickers(csv_path)
    if not tickers:
        print(f"Error: No tickers found in '{csv_path}'. Provide a CSV with a 'Ticker' column.")
        return

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    for i, tk in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Processing {tk}…")
        tk_dir = output_dir / tk
        ensure_dir(tk_dir)
        generate_segment_charts_for_ticker(tk, tk_dir)

    print("Done.")

if __name__ == "__main__":
    main()
