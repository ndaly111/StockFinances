#!/usr/bin/env python3
"""
generate_segment_charts.py
-----------------------------------

Reads tickers from tickers.csv and for each ticker:

1) Creates segment bar charts (Revenue & Operating Income) for each segment,
   using a consistent y-axis scale across all segments for the ticker.

2) Writes a compact pivot HTML table per ticker at:
   charts/{TICKER}/{TICKER}_segments_table.html

   - Columns grouped by segment (Revenue + Operating Income pairs)
   - Rows: last 3 fiscal years + TTM, plus a "% of Total (TTM)" mix row
   - Single, consistent unit across the whole table ($K/$M/$B/$T or $)
   - Revenue mix row displays percentages; Operating Income mix cells show em dashes
   - Segments ordered by latest (TTM) revenue contribution

Data source: sec_segment_data_arelle.get_segment_data(ticker)
Expected columns: Segment, Year, Revenue, OpIncome
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for CI
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data
try:
    from sec_segment_data_arelle import dump_item2_axis_ranking
except Exception:
    dump_item2_axis_ranking = None

VERSION = "SEGMENTS v2025-09-09"

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
    """Numeric years ascending; 'TTM' last; others after numeric."""
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
    if not isinstance(raw, str) or not raw:
        return raw
    name = raw.replace("SegmentMember", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    # drop common suffixes that linger after the initial replace
    name = re.sub(r"\b(Member|Segment)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
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
        "Wearables Homeand Accessories": "Wearables Home and Accessories",
    }
    title = " ".join(w if w.isupper() else w.capitalize() for w in name.split())
    return fixes.get(title, title)


def _segment_short_label(name: str) -> str:
    """Create compact labels for pivot-table column headers."""
    if not isinstance(name, str):
        return str(name)
    base = re.sub(r"\b(Member|Segment)\b", "", name, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    base = base.replace("Homeand", "Home and")
    replacements = {
        "Product": "Prod",
        "Products": "Prod",
        "Service": "Services",
        "Services": "Services",
        "I Phone": "iPhone",
        "Iphone": "iPhone",
        "I Pad": "iPad",
        "Ipad": "iPad",
        "Wearables Home And Accessories": "Wearables",
    }
    key = " ".join(part.capitalize() if not part.isupper() else part for part in base.split())
    return replacements.get(key, key)

def _to_float(x):
    if pd.isna(x): return pd.NA
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except: return pd.NA


def axis_has_meaningful_oi(df_axis: pd.DataFrame) -> bool:
    """
    True when OpIncome is actually disclosed (at least one non-zero, non-null value).
    If everything is NA or 0, treat OI as not meaningful and hide it from charts/tables.
    """
    if df_axis is None or df_axis.empty:
        return False
    if "OpIncome" not in df_axis.columns:
        return False
    oi = df_axis["OpIncome"].dropna()
    if oi.empty:
        return False
    try:
        return (oi.astype(float).abs() > 1e-9).any()
    except Exception:
        # If coercion fails, fail-open (show OI rather than accidentally hiding real data)
        return True

def _choose_scale(max_abs_value: float) -> Tuple[float, str]:
    """Pick a single divisor + unit label for the *whole* table."""
    if not isinstance(max_abs_value, (int, float)) or math.isnan(max_abs_value) or max_abs_value == 0:
        return (1.0, "$")
    v = abs(max_abs_value)
    if v >= 1e12: return (1e12, "$T")
    if v >= 1e9:  return (1e9,  "$B")
    if v >= 1e6:  return (1e6,  "$M")
    if v >= 1e3:  return (1e3,  "$K")
    return (1.0, "$")

def _fmt_scaled(x, div, unit) -> str:
    """Format number based on scale and unit."""
    if pd.isna(x):
        return "–"
    try:
        val = float(x) / float(div)
    except Exception:
        return "–"
    if abs(val) >= 10:
        fmt = "{:,.1f}" if unit == "$" else "{:.1f}"
        s = fmt.format(val)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    else:
        fmt = "{:,.2f}" if unit == "$" else "{:.2f}"
        s = fmt.format(val)
    if unit == "$":
        return f"${s}"
    return f"{s}{unit[-1]}"

def _last3_plus_ttm(years: List[str]) -> List[str]:
    nums = sorted({int(y) for y in years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(years):
        out.append("TTM")
    return out


def _cleanup_stale_axis_artifacts(out_dir: Path, ticker: str) -> None:
    """
    Remove old segment axis PNG/HTML artifacts so the site can't pick up stale charts.
    Targets both:
      - current naming: axis1_{TICKER}_*.png / axis2_{TICKER}_*.png + tables
      - legacy naming: {TICKER}_*_axis1.png / {TICKER}_*_axis2.png
    """
    stale_patterns = [
        # current naming (restrict to this ticker for safety)
        f"axis1_{ticker}_*",
        f"axis2_{ticker}_*",
        # legacy naming used on the site (e.g. C_Banking_axis2.png)
        f"{ticker}_*_axis1.png",
        f"{ticker}_*_axis2.png",
        f"{ticker}_*_axis1*.html",
        f"{ticker}_*_axis2*.html",
    ]
    for pattern in stale_patterns:
        for old_file in out_dir.glob(pattern):
            try:
                old_file.unlink()
            except Exception:
                pass


def _write_axis_placeholders(out_dir: Path, ticker: str, message: str) -> None:
    """
    Ensure axis tables always exist so the Segments section never disappears
    from the generated site when data fetch/parsing fails.
    """
    for idx in (1, 2):
        (out_dir / f"axis{idx}_{ticker}_segments_table.html").write_text(
            f"<p>{message} (axis {idx}).</p>",
            encoding="utf-8",
        )

# ───────────────────── prospectus seed support ────────────────────
# Some tickers (e.g. SPCX post-IPO) have no 10-K yet, so the iXBRL path has no
# clean fiscal years. data/segment_seed_{TICKER}.json (written by
# prospectus_segments.py from the S-1/424B4) supplies frozen annual segment
# data. mode="replace" ignores the scraped rows entirely; mode="fill" appends
# only fiscal years the scrape didn't return.

SEED_DIR = Path("data")

SEED_EXPENSE_METRICS = [
    ("Revenue", "Revenue"),
    ("CostOfRevenue", "Cost of revenue"),
    ("ResearchAndDevelopment", "Research and development"),
    ("SellingGeneralAndAdministrative", "Selling, general, and administrative"),
    ("RestructuringCharges", "Restructuring charges"),
    ("Impairment", "Impairment"),
    ("TotalCostsAndExpenses", "Total costs and expenses"),
    ("OpIncome", "Income (loss) from operations"),
]


def _load_segment_seed(ticker: str) -> dict | None:
    path = SEED_DIR / f"segment_seed_{ticker}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{VERSION}] could not read segment seed for {ticker}: {e}")
        return None


def _seed_to_df(seed: dict) -> pd.DataFrame:
    mult = 1e6 if seed.get("units") == "millions" else 1.0
    axis = seed.get("axis", "StatementBusinessSegmentsAxis")
    rows = []
    for r in seed.get("rows", []):
        rows.append({
            "Axis": axis,
            "Segment": r["Segment"],
            "Year": str(r["Year"]),
            "Revenue": (r["Revenue"] * mult) if r.get("Revenue") is not None else pd.NA,
            "OpIncome": (r["OpIncome"] * mult) if r.get("OpIncome") is not None else pd.NA,
            "AxisType": axis,
        })
    return pd.DataFrame(rows)


def _render_seed_expense_table(ticker: str, seed: dict) -> str | None:
    """Per-segment expense breakdown table (metrics × segments, grouped by
    year desc) appended below the axis1 segment table. Reuses .segment-pivot
    CSS already emitted in that file."""
    rows = seed.get("rows", [])
    segments = list(dict.fromkeys(r["Segment"] for r in rows))
    years = sorted({str(r["Year"]) for r in rows}, reverse=True)
    by_key = {(r["Segment"], str(r["Year"])): r for r in rows}
    present = [(k, lbl) for k, lbl in SEED_EXPENSE_METRICS
               if any(k in r for r in rows)]
    if len(present) <= 2 or not segments:
        return None  # seed carries no expense detail beyond Revenue/OpIncome

    mult = 1e6 if seed.get("units") == "millions" else 1.0
    max_abs = max((abs(r[k] * mult) for r in rows for k, _ in present
                   if r.get(k) is not None), default=0.0)
    div, unit = _choose_scale(max_abs)

    out = ['<h4 style="font-family:Arial,sans-serif;margin:18px 0 4px;">'
           f'{ticker} — Segment Expense Breakdown</h4>']
    src = seed.get("source", "S-1 prospectus")
    out.append(f'<div class="table-note">Units: <b>{unit}</b>. Source: {src}.</div>')
    out.append("<div class='table-wrap'><table class='segment-pivot' border=0>")
    header = "".join(f"<th>{s}</th>" for s in segments)
    for year in years:
        out.append(f"<thead><tr><th>FY{year}</th>{header}<th>Total</th></tr></thead>")
        out.append("<tbody>")
        for k, label in present:
            cells, total, seen = [], 0.0, False
            for s in segments:
                v = by_key.get((s, year), {}).get(k)
                if v is None:
                    cells.append("<td>–</td>")
                else:
                    total += v * mult
                    seen = True
                    cells.append(f"<td>{_fmt_scaled(v * mult, div, unit)}</td>")
            tot = _fmt_scaled(total, div, unit) if seen else "–"
            out.append(f"<tr><td>{label}</td>{''.join(cells)}<td>{tot}</td></tr>")
        out.append("</tbody>")
    out.append("</table></div>")
    return "\n".join(out)

# ───────────────────── main per-ticker routine ────────────────────

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """Generate charts and a compact pivot HTML table for a single ticker."""
    ticker = ticker.upper()

    # Ensure out_dir exists before dump_raw writes into it
    ensure_dir(out_dir)

    seed = _load_segment_seed(ticker)

    if seed and seed.get("mode") == "replace":
        # Frozen prospectus data replaces the scrape entirely (see seed notes).
        df = None
    else:
        try:
            df = get_segment_data(ticker, dump_raw=True, raw_dir=out_dir)
        except Exception as fetch_err:
            print(f"[{VERSION}] Error fetching segment data for {ticker}: {fetch_err}")
            if seed is None:
                # If we fail to fetch/parse, do NOT leave the site with no axis tables.
                # Also clean stale charts so we don't keep serving incorrect images.
                _cleanup_stale_axis_artifacts(out_dir, ticker)
                (out_dir / f"{ticker}_segments_table.html").write_text(
                    f"<p>Error fetching segment data for {ticker}: {fetch_err}</p>",
                    encoding="utf-8",
                )
                _write_axis_placeholders(out_dir, ticker, f"Error fetching segment data for {ticker}: {fetch_err}")
                return
            df = None  # fall through to the seed

    ensure_dir(out_dir)

    # Clean stale axis artifacts before regenerating.
    _cleanup_stale_axis_artifacts(out_dir, ticker)

    if (df is None or df.empty) and seed is None:
        (out_dir / f"{ticker}_segments_table.html").write_text(
            f"<p>No segment data available for {ticker}.</p>", encoding="utf-8"
        )
        # Critical: keep axis tables present so the site still renders the section.
        _write_axis_placeholders(out_dir, ticker, f"No segment data available for {ticker}")
        return

    if df is None or df.empty:
        df = pd.DataFrame(columns=["Axis", "Segment", "Year", "Revenue", "OpIncome", "AxisType"])

    df = df.copy()
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"] = df["Year"].astype(str)
    df["Revenue"] = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    if seed:
        seed_df = _seed_to_df(seed)  # names verbatim — no humanize (e.g. "AI")
        if seed.get("mode") == "replace":
            df = seed_df
        else:
            have = set(zip(df["AxisType"], df["Year"]))
            fresh = seed_df[[(a, y) not in have
                             for a, y in zip(seed_df["AxisType"], seed_df["Year"])]]
            df = pd.concat([df, fresh], ignore_index=True)
        print(f"[{VERSION}] applied segment seed for {ticker} "
              f"(mode={seed.get('mode', 'fill')}, rows={len(df)})")

    if dump_item2_axis_ranking:
        try:
            dump_item2_axis_ranking(ticker, df)
        except Exception:
            pass

    axes = []
    used_segments = set()
    for axis_type, sub in df.groupby("AxisType"):
        segs = set(sub["Segment"])
        segs = segs - used_segments
        if len(segs) <= 1:
            continue
        used_segments.update(segs)
        axes.append((axis_type, sub[sub["Segment"].isin(segs)].copy()))
    axes = axes[:2]

    for idx in range(1, 3):
        axis_label = f"axis{idx}"
        if idx <= len(axes):
            _, df_axis = axes[idx - 1]
            oi_present = axis_has_meaningful_oi(df_axis)

            vals = [df_axis["Revenue"].dropna()]
            if oi_present:
                vals.append(df_axis["OpIncome"].dropna())
            all_vals = pd.concat(vals, ignore_index=True)
            if all_vals.empty:
                min_y, max_y = 0.0, 0.0
            else:
                min_y, max_y = float(all_vals.min()), float(all_vals.max())
                if min_y > 0: min_y = 0.0
                if max_y < 0: max_y = 0.0
            spread = max_y - min_y
            margin = spread * 0.1 if spread else 1.0
            min_y_plot, max_y_plot = min_y - margin, max_y + margin

            years_all = sort_years(sorted(set(df_axis["Year"].tolist())))
            years_tbl = _last3_plus_ttm(df_axis["Year"].tolist())
            segments = sorted(set(df_axis["Segment"].tolist()))

            for seg in segments:
                seg_df = df_axis[df_axis["Segment"] == seg]
                revenues = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum(min_count=1) for y in years_all]
                op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum(min_count=1) for y in years_all]
                revenues_b = [float(v) / 1e9 if pd.notna(v) else float("nan") for v in revenues]
                op_incomes_b = [float(v) / 1e9 if pd.notna(v) else float("nan") for v in op_incomes]

                # If there is literally nothing to plot, do not emit a PNG.
                has_rev = not all(pd.isna(v) for v in revenues)
                has_oi = oi_present and (not all(pd.isna(v) for v in op_incomes))
                if not has_rev and not has_oi:
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))
                x_indices = list(range(len(years_all)))
                bar_width = 0.35
                ax.bar([x - bar_width / 2 for x in x_indices], revenues_b, width=bar_width, label="Revenue")
                if oi_present:
                    ax.bar([x + bar_width / 2 for x in x_indices], op_incomes_b, width=bar_width, label="Operating Income")
                ax.set_xticks(x_indices)
                ax.set_xticklabels(years_all)
                ax.set_ylim(min_y_plot / 1e9, max_y_plot / 1e9)
                ax.set_ylabel("Value ($B)")
                ax.set_title(seg)
                ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                if oi_present:
                    ax.legend(loc="upper left")
                plt.tight_layout()
                fig_path = out_dir / f"{axis_label}_{ticker}_{seg.replace('/', '_').replace(' ', '_')}.png"
                plt.savefig(fig_path)
                plt.close(fig)

                # Extra safety: remove tiny/blank charts so they don't get served.
                try:
                    if fig_path.exists() and fig_path.stat().st_size < 2048:
                        fig_path.unlink()
                        print(f"[{VERSION}] removed tiny/blank chart → {fig_path}")
                except Exception:
                    pass

            def pv(col):
                # IMPORTANT: pivot_table(..., sum) will turn "all missing" into 0.
                # Use groupby.sum(min_count=1) so missing stays missing.
                base = df_axis[df_axis["Year"].isin(years_tbl)].copy()
                g = base.groupby(["Segment", "Year"])[col].sum(min_count=1).unstack("Year")
                return g.reindex(columns=[y for y in years_tbl if y in g.columns])

            rev_p = pv("Revenue")
            oi_p  = pv("OpIncome") if oi_present else pd.DataFrame(index=rev_p.index)

            sort_col = "TTM" if "TTM" in rev_p.columns else (rev_p.columns[-1] if len(rev_p.columns) else None)
            if sort_col:
                rev_p = rev_p.sort_values(by=sort_col, ascending=False)
                oi_p = oi_p.reindex(index=rev_p.index)

            # Emit a manifest of segment filenames in size-desc order so the
            # page template can render the chart stack from largest to smallest
            # (the loop above iterates segments alphabetically when writing
            # PNGs, but display order should follow business size).
            try:
                import json as _json
                _sorted_segs = list(rev_p.index)
                _ordered_files = [
                    f"{axis_label}_{ticker}_{str(s).replace('/', '_').replace(' ', '_')}.png"
                    for s in _sorted_segs
                ]
                manifest_path = out_dir / f"{axis_label}_order.json"
                manifest_path.write_text(
                    _json.dumps({"segments": _sorted_segs, "files": _ordered_files}),
                    encoding="utf-8",
                )
            except Exception as _e:
                print(f"[{VERSION}] could not write segment order manifest: {_e}")

            pct_series = None
            if "TTM" in rev_p.columns:
                total_ttm = rev_p["TTM"].sum(skipna=True)
                if total_ttm and total_ttm != 0:
                    pct_series = (rev_p["TTM"] / total_ttm) * 100.0

            # Coerce non-numeric (Python None) to NaN so .abs() doesn't choke
            # on object-dtype columns. pd.to_numeric with errors='coerce'
            # leaves numerics alone and NaN'ifies anything else.
            _combined = pd.concat([rev_p, oi_p]).apply(pd.to_numeric, errors="coerce")
            max_val = _combined.abs().max().max()
            div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

            segment_labels = [(seg, _segment_short_label(seg)) for seg in rev_p.index]
            column_specs: List[Tuple[str, str, str | None]] = []
            for seg, label in segment_labels:
                column_specs.append((label, "Rev", seg))
                if oi_present:
                    column_specs.append((label, "OI", seg))
            column_specs.append(("Total", "Rev", None))
            if oi_present:
                column_specs.append(("Total", "OI", None))

            year_rows: List[dict] = []
            for year in years_tbl:
                row: dict[str, object] = {"Year": year}
                totals = {"Rev": 0.0, "OI": 0.0}
                totals_present = {"Rev": False, "OI": False}
                for label, kind, seg in column_specs:
                    key = f"{label} {kind}"
                    if seg is None:
                        continue
                    src_df = rev_p if kind == "Rev" else oi_p
                    if year in src_df.columns and seg in src_df.index:
                        raw_val = src_df.at[seg, year]
                    else:
                        raw_val = pd.NA
                    row[key] = raw_val
                    if pd.notna(raw_val):
                        totals[kind] += float(raw_val)
                        totals_present[kind] = True
                row["Total Rev"] = totals["Rev"] if totals_present["Rev"] else pd.NA
                if oi_present:
                    row["Total OI"] = totals["OI"] if totals_present["OI"] else pd.NA
                year_rows.append(row)

            display_rows: List[dict[str, object]] = []
            for raw_row in year_rows:
                disp_row = {"Year": raw_row["Year"]}
                for label, kind, _ in column_specs:
                    key = f"{label} {kind}"
                    disp_row[key] = _fmt_scaled(raw_row.get(key, pd.NA), div, unit)
                display_rows.append(disp_row)

            if pct_series is not None:
                pct_row: dict[str, object] = {"Year": "% of Total (TTM)"}
                for label, kind, seg in column_specs:
                    key = f"{label} {kind}"
                    if kind == "Rev":
                        if seg is None:
                            pct_row[key] = "100%"
                        else:
                            pct_val = pct_series.get(seg)
                            pct_row[key] = f"{float(pct_val):.1f}%" if pd.notna(pct_val) else "–"
                    else:
                        pct_row[key] = "—"
                display_rows.append(pct_row)

            col_order = ["Year"] + [f"{label} {kind}" for label, kind, _ in column_specs]
            out_disp = pd.DataFrame(display_rows, columns=col_order)

            css = """
<style>
.table-wrap{overflow:auto; max-width:100%;}
.segment-pivot { width:100%; border-collapse:collapse; font-family: Arial, sans-serif; font-size:14px; }
.segment-pivot thead th { position:sticky; top:0; background:#fff; z-index:1; border-bottom:1px solid #ddd; }
.segment-pivot th, .segment-pivot td { padding:6px 8px; border-bottom:1px solid #f0f0f0; }
.segment-pivot tbody tr:nth-child(even){ background:#fafafa; }
.segment-pivot td, .segment-pivot th { white-space:nowrap; }
.segment-pivot td { font-variant-numeric: tabular-nums; text-align:right; }
.segment-pivot td:first-child, .segment-pivot th:first-child { text-align:left; }
.table-note { font-size:12px; color:#666; margin:6px 0 8px; }
</style>
""".strip()

            stamp = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC")
            metric_phrase = "revenue" + (" and operating income" if oi_present else "")
            caption = (
                f'<div class="table-note">{VERSION} · {stamp} — Units: <b>{unit}</b>. '
                f"Rows list fiscal years (last 3 + TTM) with {metric_phrase} for each segment; "
                'the final row shows the TTM revenue mix (operating income columns display “—” where mix is not applicable).</div>'
            )
            html = out_disp.to_html(index=False, escape=False, classes="segment-pivot", border=0)
            table_content = css + "\n" + caption + f"\n<div class='table-wrap'>{html}</div>"

            out_path = out_dir / f"{axis_label}_{ticker}_segments_table.html"
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            out_path.write_text(table_content, encoding="utf-8")
            print(f"[{VERSION}] writing table → {out_path}")
            try:
                print(f"[{VERSION}] wrote {out_path.stat().st_size} bytes")
            except Exception:
                pass
        else:
            # write placeholder table
            out_path = out_dir / f"{axis_label}_{ticker}_segments_table.html"
            out_path.write_text(
                f"<p>No segment data available for {ticker} (axis {idx}).</p>",
                encoding="utf-8",
            )

    # Seeded tickers: append the prospectus expense breakdown below the axis1 table.
    if seed:
        expense_html = _render_seed_expense_table(ticker, seed)
        axis1_path = out_dir / f"axis1_{ticker}_segments_table.html"
        if expense_html and axis1_path.is_file():
            axis1_path.write_text(
                axis1_path.read_text(encoding="utf-8") + "\n" + expense_html,
                encoding="utf-8",
            )
            print(f"[{VERSION}] appended seed expense table → {axis1_path}")

# ─────────────────────────── CLI wrapper ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate segment charts and tables for a list of tickers.")
    parser.add_argument("--tickers_csv", type=str, default="tickers.csv")
    parser.add_argument("--output_dir", type=str, default="charts")
    args = parser.parse_args()

    print(f"{VERSION} starting…")

    csv_path = Path(args.tickers_csv)
    tickers = read_tickers(csv_path)
    if not tickers:
        print(f"Error: No tickers found in '{csv_path}'. Please provide a valid CSV file with a 'Ticker' column.")
        return

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] Processing {ticker}…")
        ticker_dir = output_dir / ticker.upper()
        generate_segment_charts_for_ticker(ticker, ticker_dir)

    print("Done.")

if __name__ == "__main__":
    main()
