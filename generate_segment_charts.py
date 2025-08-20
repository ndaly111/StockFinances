#!/usr/bin/env python3
"""
generate_segment_charts.py — SEGMENTS v2025-08-20 (axis-first, issuer-agnostic)

What this does:
• Builds one HTML section PER UNIQUE AxisType (issuer-agnostic: Products/Services, Regions, Customers, Channels, etc.).
• Shared y-axis across all per-segment charts (handles negatives).
• Compact, scaled HTML pivot tables (last 3 FY + TTM). TTM bolded; “% of Total (TTM)”.
• Single unit per section ($, $K, $M, $B, $T). Drops OI columns if fully missing for the section.
• Filters: elimination/reconciliation/unallocated buckets removed; negative latest-revenue rows removed.
• Writes to charts/{TICKER}/{TICKER}_segments_table.html + alias copies; cleans legacy PNGs.

Extractor contract:
    from sec_segment_data_arelle import get_segment_data
    -> returns DataFrame with columns at least:
        Segment(str), Year(str), Revenue(float), OpIncome(float|NaN), AxisType(str|NaN)
"""

from __future__ import annotations
import argparse, math, re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data

VERSION = "SEGMENTS v2025-08-20"

# ─────────────────────────── utilities ───────────────────────────

def read_tickers(csv_path: Path) -> List[str]:
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
    def key(y: str) -> Tuple[int, int | str]:
        if y == "TTM": return (2, 0)
        try: return (0, int(y))
        except Exception: return (1, y)
    return [y for _, y in sorted([(key(y), y) for y in years], key=lambda x: x[0])]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _humanize_segment_name(raw: str) -> str:
    """Clean noisy XBRL labels to human readable."""
    if not isinstance(raw, str) or not raw:
        return str(raw)
    # Remove trailing Member/Segment and join spaced initials (e.g., "U S" -> "US")
    name = str(raw)
    name = re.sub(r"\s*(Member|Segment)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b([A-Z])\s+([A-Z])\b", r"\1\2", name)  # join spaced initials
    # Split CamelCase → words
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    title = " ".join(w if w.isupper() else w.capitalize() for w in name.split())
    return title

def _norm_axis_label(axis: Optional[str]) -> str:
    """
    Convert raw AxisType (possibly namespaced) to a neat section title.
    Works for arbitrary axes (issuer-agnostic).
    """
    s = (axis or "").strip()
    s = re.sub(r".*:", "", s)                # strip ns prefix
    s = s.replace("Axis", "")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)   # camel → words
    s = s.replace("_", " ").strip()
    if not s:
        return "Unlabeled Axis"
    # Friendly labels
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
    if not isinstance(max_abs_value, (int, float)) or math.isnan(max_abs_value) or max_abs_value == 0:
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

def _last3_plus_ttm(years: List[str]) -> List[str]:
    nums = sorted({int(y) for y in years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(years):
        out.append("TTM")
    return out

def _safe_seg_filename(seg: str) -> str:
    return seg.replace("/", "_").replace(" ", "_")

# ───────────────────── cleanup helper ───────────────────────

def _cleanup_segment_pngs(out_dir: Path, ticker: str, keep_files: List[str]) -> None:
    """
    Remove legacy/duplicate PNGs after we write the canonical set.
    Only touch:
      - 'segment_performance.png' and '{ticker}_segment_performance.png'
      - files starting with '<TICKER>_'  not in keep set
    """
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

# ───────────────────── filters ──────────────────────────────

HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

# ───────────────────── main per-ticker routine ────────────────────

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """Generate per-axis sections (generic), charts, and compact pivot HTML."""
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
    df["Year"] = df["Year"].astype(str)
    df["Revenue"] = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    # If OpIncome completely missing or zeros, set NA so it displays "–"
    _op_all_missing = df["OpIncome"].isna().all() or (df["OpIncome"].fillna(0) == 0).all()
    if _op_all_missing:
        df["OpIncome"] = pd.NA

    # Shared y-axis for charts (across all segments)
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

    years_all = sort_years(sorted(set(df["Year"].tolist())))
    years_tbl = _last3_plus_ttm(df["Year"].tolist())

    # ── Charts per segment (y in $B) ──
    segments = sorted(set(df["Segment"].tolist()))
    written_pngs: List[str] = []
    for seg in segments:
        seg_df = df[df["Segment"] == seg]
        revenues   = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum() for y in years_all]
        op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum() for y in years_all]

        revenues_b   = [0.0 if pd.isna(v) else v / 1e9 for v in revenues]
        op_incomes_b = [0.0 if pd.isna(v) else v / 1e9 for v in op_incomes]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(years_all)))
        w = 0.35
        ax.bar([i - w/2 for i in x], revenues_b,  width=w, label="Revenue")
        ax.bar([i + w/2 for i in x], op_incomes_b, width=w, label="Operating Income")
        ax.set_xticks(x); ax.set_xticklabels(years_all)
        ax.set_ylim(min_y_plot / 1e9, max_y_plot / 1e9)
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

    # ── Compact pivot helper ──
    def pv(col: str, sub_df: pd.DataFrame) -> pd.DataFrame:
        p = sub_df[sub_df["Year"].isin(years_tbl)].pivot_table(
            index="Segment", columns="Year", values=col, aggfunc="sum"
        )
        return p.reindex(columns=[y for y in years_tbl if y in p.columns])

    # Ensure AxisType exists
    if "AxisType" not in df.columns or df["AxisType"].isna().all():
        df["AxisType"] = "UnlabeledAxis"

    # Build sections by AxisType
    sections_html: List[str] = []
    axes_found: List[str] = []

    for axis_value, group in df.groupby("AxisType", dropna=False):
        label = _norm_axis_label(axis_value)
        axes_found.append(label)

        rev_p = pv("Revenue", group)
        oi_p  = pv("OpIncome", group)

        # Choose sort col (TTM if present else latest year)
        sort_col = "TTM" if "TTM" in rev_p.columns else (rev_p.columns[-1] if len(rev_p.columns) else None)
        if sort_col:
            # Remove rows with missing latest revenue
            if sort_col in rev_p.columns:
                rev_p = rev_p[rev_p[sort_col].notna()]
            oi_p = oi_p.reindex(index=rev_p.index)

            # Remove elimination/reconciliation/unallocated/aggregation buckets
            hide_mask = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
            rev_p = rev_p[~hide_mask];  oi_p = oi_p.reindex(index=rev_p.index)

            # Remove negative latest revenue rows
            neg_mask = rev_p[sort_col] < 0
            rev_p = rev_p[~neg_mask];   oi_p = oi_p.reindex(index=rev_p.index)

            # Sort by latest revenue desc and align OI
            rev_p = rev_p.sort_values(by=sort_col, ascending=False)
            oi_p  = oi_p.loc[rev_p.index]

        # If nothing left, render a small notice section
        if rev_p.empty and oi_p.empty:
            sections_html.append(f"<h3>{label}</h3><div class='table-wrap'><p>No data for this axis.</p></div>")
            continue

        # % mix on visible rows
        pct_series = None
        if "TTM" in rev_p.columns:
            total_ttm = rev_p["TTM"].sum(skipna=True)
            if total_ttm:
                pct_series = (rev_p["TTM"] / total_ttm) * 100.0

        # Single unit per section
        max_val = pd.concat([rev_p, oi_p]).abs().max().max()
        div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

        # Build columns: Year Rev, Year OI … TTM Rev, TTM OI
        cols: List[Tuple[str, str]] = []
        for y in [c for c in _last3_plus_ttm(list(rev_p.columns)) if c != "TTM"]:
            cols += [(y, "Rev"), (y, "OI")]
        if "TTM" in rev_p.columns:
            cols += [("TTM", "Rev"), ("TTM", "OI")]

        out = pd.DataFrame(index=rev_p.index)
        for (y, kind) in cols:
            series = rev_p.get(y) if kind == "Rev" else oi_p.get(y)
            out[f"{y} {'Rev' if kind=='Rev' else 'OI'} ({unit})"] = series

        # Hide OpIncome columns entirely for this section if fully missing
        hide_oi = (group["OpIncome"].isna().all()) or ((group["OpIncome"].fillna(0) == 0).all())
        if hide_oi:
            out = out[[c for c in out.columns if " OI " not in c]]

        if pct_series is not None:
            out["% of Total (TTM)"] = pct_series

        # Format cells
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

    # CSS + caption + debug footer
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
        f'<div class="table-note">{VERSION} · {stamp} — Values use a single scale per section. '
        f'TTM is <b>bold</b>. “% of Total (TTM)” uses visible rows in that section.</div>'
    )
    debug = (
        "<div style='font-size:12px;color:#666;margin:6px 0'>"
        "Sections generated: " + (", ".join(sorted(set(axes_found))) if axes_found else "None") +
        ".</div>"
    )

    content = f"<!-- {VERSION} -->\n{css}\n{caption}\n" + "\n<hr/>\n".join(sections_html) + "\n" + debug

    # Write canonical + aliases
    canonical = out_dir / f"{ticker}_segments_table.html"
    aliases = [
        out_dir / "segments_table.html",
        out_dir / "segment_performance.html",
        out_dir / f"{ticker}_segment_performance.html",
    ]

    def write_file(p: Path, text: str):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        p.write_text(text, encoding="utf-8")
        print(f"[{VERSION}] wrote {p} ({p.stat().st_size} bytes)")

    write_file(canonical, content)
    for a in aliases:
        write_file(a, content)

# ─────────────────────────── CLI wrapper ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate axis-first segment charts and tables for tickers.")
    parser.add_argument("--tickers_csv", type=str, default="tickers.csv",
                        help="CSV with a 'Ticker' column")
    parser.add_argument("--output_dir", type=str, default="charts",
                        help="Output directory (default: charts/)")
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
