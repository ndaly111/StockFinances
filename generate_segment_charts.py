#!/usr/bin/env python3
"""
generate_segment_charts.py — write charts and a single canonical table per ticker.

• Writes PNGs into charts/<T>/<T>_<axis-slug>_<segment>.png
• Writes one combined table into charts/<T>/<T>_segments_table.html
• Cleans up old alias files so only the canonical table remains.
"""
from __future__ import annotations
import argparse, math, re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data

VERSION = "SEGMENTS v2025-08-27"

# Regex to hide recon/elimination/unallocated lines
HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

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

def _to_float(x):
    if pd.isna(x): return pd.NA
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except: return pd.NA

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

def _last3_plus_ttm(cols: List[str]) -> List[str]:
    # Keep only the last 3 fiscal years; TTM handled separately upstream
    nums = sorted({int(y) for y in cols if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    return [str(y) for y in keep]

def _safe_seg_filename(seg: str) -> str:
    return seg.replace("/", "_").replace(" ", "_")

def _slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"(^-|-$)", "", s)

def _cleanup_legacy_tables(out_dir: Path, ticker: str) -> None:
    # Remove older alias files so only the canonical table remains
    for p in [
        out_dir / "segments_table.html",
        out_dir / "segment_performance.html",
        out_dir / f"{ticker}_segment_performance.html",
        Path("charts") / f"{ticker}_segments_table.html",  # old root copy
    ]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """
    Given a ticker and output directory, produce:
    • charts/<ticker>/<ticker>_<axis-slug>_<segment>.png images
    • charts/<ticker>/<ticker>_segments_table.html (combined table)
    Remove any old alias files.
    """
    charts_root = Path("charts")
    canonical_dir = charts_root / ticker
    try:
        out_dir = Path(out_dir)
    except Exception:
        out_dir = canonical_dir
    if out_dir.resolve().name.upper() != ticker.upper():
        out_dir = canonical_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / f"{ticker}_segments_table.html"

    df = get_segment_data(ticker)
    if df is None or df.empty:
        table_path.write_text(f"<p>No segment data available for {ticker}.</p>", encoding="utf-8")
        _cleanup_legacy_tables(out_dir, ticker)
        return

    df = df.copy()
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"] = df["Year"].astype(str)
    df["Revenue"] = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    # Compute global y-limits across all charts for consistent scaling
    all_vals = pd.concat([df["Revenue"].dropna(), df["OpIncome"].dropna()], ignore_index=True)
    if all_vals.empty:
        min_y, max_y = 0.0, 0.0
    else:
        min_y, max_y = float(all_vals.min()), float(all_vals.max())
        if min_y > 0: min_y = 0.0
        if max_y < 0: max_y = 0.0
    spread = max_y - min_y
    margin = spread * 0.1 if spread else 1.0
    min_y_plot, max_y_plot = min_y - margin, max_y + margin

    years_all = sorted(set(df["Year"].tolist()), key=lambda s: (not s.isdigit(), s))
    years_tbl = _last3_plus_ttm(list(df["Year"].unique()))

    # Generate charts
    written_pngs: List[str] = []
    for (axis, seg), seg_df in df.groupby(["AxisType", "Segment"], dropna=False):
        revenues   = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum() for y in years_all]
        op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum() for y in years_all]

        revenues_b   = [0.0 if pd.isna(v) else v / 1e9 for v in revenues]
        op_incomes_b = [0.0 if pd.isna(v) else v / 1e9 for v in op_incomes]

        fig, axp = plt.subplots(figsize=(8, 5))
        x = list(range(len(years_all)))
        w = 0.35
        axp.bar([i - w/2 for i in x], revenues_b,  width=w, label="Revenue")
        axp.bar([i + w/2 for i in x], op_incomes_b, width=w, label="Operating Income")
        axp.set_xticks(x)
        axp.set_xticklabels(years_all)
        axp.set_ylim(min_y_plot / 1e9, max_y_plot / 1e9)
        axis_label = _norm_axis_label(axis)
        axp.set_ylabel("Value ($B)")
        axp.set_title(f"{seg} — {axis_label}")
        axp.axhline(0, linewidth=0.8)
        axp.yaxis.grid(True, linestyle="--", alpha=0.5)
        axp.legend(loc="upper left")
        plt.tight_layout()

        safe_seg = _safe_seg_filename(seg)
        axis_slug = _slug(axis_label)
        out_name = f"{ticker}_{axis_slug}_{safe_seg}.png"
        plt.savefig(out_dir / out_name)
        plt.close(fig)
        written_pngs.append(out_name)

    # Helper for pivot & table assembly
    def pivot_agg(col: str, sub_df: pd.DataFrame) -> pd.DataFrame:
        p = sub_df[sub_df["Year"].isin(years_tbl)].pivot_table(
            index="Segment", columns="Year", values=col, aggfunc="sum"
        )
        return p.reindex(columns=[y for y in years_tbl if y in p.columns])

    if "AxisType" not in df.columns or df["AxisType"].isna().all():
        df["AxisType"] = "UnlabeledAxis"

    sections_html: List[str] = []
    axes_found: List[str] = []

    for axis_value, group in df.groupby("AxisType", dropna=False):
        label = _norm_axis_label(axis_value)
        axes_found.append(label)

        rev_p = pivot_agg("Revenue", group)
        oi_p  = pivot_agg("OpIncome", group)

        sort_col = rev_p.columns[-1] if len(rev_p.columns) else None
        if sort_col:
            if sort_col in rev_p.columns:
                rev_p = rev_p[rev_p[sort_col].notna()]
            oi_p = oi_p.reindex(index=rev_p.index)

            hide_mask = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
            rev_p = rev_p[~hide_mask]
            oi_p  = oi_p.reindex(index=rev_p.index)

            neg_mask = rev_p[sort_col] < 0
            rev_p = rev_p[~neg_mask]
            oi_p  = oi_p.reindex(index=rev_p.index)

            rev_p = rev_p.sort_values(by=sort_col, ascending=False)
            oi_p  = oi_p.loc[rev_p.index]

        if rev_p.empty and oi_p.empty:
            sections_html.append(f"<h3>{label}</h3><div class='table-wrap'><p>No data for this axis.</p></div>")
            continue

        pct_series = None
        if "TTM" in rev_p.columns:
            total_ttm = rev_p["TTM"].sum(skipna=True)
            if total_ttm:
                pct_series = (rev_p["TTM"] / total_ttm) * 100.0

        max_val = pd.concat([rev_p, oi_p]).abs().max().max()
        div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

        cols: List[Tuple[str, str]] = []
        for y in [c for c in rev_p.columns if c != "TTM"]:
            cols += [(y, "Rev"), (y, "OI")]
        if "TTM" in rev_p.columns:
            cols += [("TTM", "Rev"), ("TTM", "OI")]

        out = pd.DataFrame(index=rev_p.index)
        for (y, kind) in cols:
            series = rev_p.get(y) if kind == "Rev" else oi_p.get(y)
            out[f"{y} {'Rev' if kind=='Rev' else 'OI'} ({unit})"] = series

        hide_oi = (group["OpIncome"].isna().all()) or ((group["OpIncome"].fillna(0) == 0).all())
        if hide_oi:
            out = out[[c for c in out.columns if " OI " not in c]]

        if pct_series is not None:
            out["% of Total (TTM)"] = pct_series

        for c in out.columns:
            if c == "% of Total (TTM)":
                out[c] = out[c].map(lambda x: f"{float(x):.1f}%" if pd.notnull(x) else "–")
            else:
                out[c] = out[c].map(lambda x, d=div, u=unit: _fmt_scaled(x, d, u))
        for ttm_col in [c for c in out.columns if c.startswith("TTM ")]:
            out[ttm_col] = out[ttm_col].map(lambda s: f"<strong>{s}</strong>" if s != "–" else s)

        out.index.name = "Segment"
        html_table = out.reset_index().to_html(index=False, escape=False, classes="segment-pivot", border=0)
        sections_html.append(f"<h3>{label}</h3>\n<div class='table-wrap'>{html_table}</div>")

    css = """
<style>
.table-wrap{overflow:auto; max-width:100%;}
.segment-pivot{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px}
.segment-pivot th,.segment-pivot td{padding:6px 8px;border-bottom:1px solid #f0f0f0; white-space:nowrap}
.segment-pivot thead th{position:sticky;top:0;background:#fff;z-index:1;border-bottom:1px solid #ddd}
</style>
""".strip()

    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    caption = (
        f'<div style="font-size:12px;color:#666;margin:6px 0 8px;">'
        f'{VERSION} · {stamp} — Single-scale per section; TTM is <b>bold</b>; '
        f'“% of Total (TTM)” uses visible rows in that section.'
        f'</div>'
    )
    content = css + "\n" + caption + "\n" + "\n<hr/>\n".join(sections_html)
    table_path.write_text(content, encoding="utf-8")
    print(f"[{VERSION}] wrote {table_path} ({table_path.stat().st_size} bytes)")

    # Remove old duplicate tables
    _cleanup_legacy_tables(out_dir, ticker)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers_csv", default="tickers.csv")
    ap.add_argument("--output_dir", default="charts")
    args = ap.parse_args()
    p = Path(args.tickers_csv)
    if not p.is_file():
        print(f"ERROR: {p} missing or no 'Ticker' column")
        return
    df = pd.read_csv(p)
    if "Ticker" not in df.columns:
        print(f"ERROR: 'Ticker' column missing in {p}")
        return
    tickers = df["Ticker"].dropna().astype(str).str.upper().tolist()
    out_base = Path(args.output_dir)
    for i, tk in enumerate(tickers, start=1):
        out_dir = out_base / tk
        ensure_dir(out_dir)
        print(f"[{i}/{len(tickers)}] {tk} → {out_dir}")
        generate_segment_charts_for_ticker(tk, out_dir)

if __name__ == "__main__":
    main()
