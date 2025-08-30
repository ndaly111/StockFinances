#!/usr/bin/env python3
"""
generate_segment_charts.py — write charts (PNG) and a canonical table per ticker.

• PNGs: charts/<T>/<T>_<axis-slug>_<segment>.png
• Table: charts/<T>/<T>_segments_table.html
"""
from __future__ import annotations

# NEW (headless backend) – must come before pyplot import
import matplotlib
matplotlib.use("Agg")

import argparse, math, re, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # lightweight fallback for earnings date

from sec_segment_data_arelle import get_segment_data

VERSION = "SEGMENTS v2025-08-27"
DB_PATH = "Stock Data.db"                  # existing DB
STAMP_FILENAME = "_segments_stamp.json"    # per‑ticker freshness stamp

HIDE_RE = re.compile(
    r"(Eliminat|Reconcil|Intersegment|Unallocat|All Other|"
    r"Corporate(?!.*Bank)|Consolidat|Adjust|Aggregation)",
    re.IGNORECASE,
)

# ───────────────────────────────────────────────────────────
# tiny freshness helpers (self‑contained; no main_remote changes)
# ───────────────────────────────────────────────────────────
def _seg_stamp_path(ticker: str) -> Path:
    return Path("charts") / ticker / STAMP_FILENAME

def _read_seg_stamp(ticker: str) -> Optional[datetime]:
    p = _seg_stamp_path(ticker)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        dt = pd.to_datetime(data.get("updated_at"), errors="coerce")
        if pd.notna(dt):
            return dt.to_pydatetime().replace(tzinfo=None)
    except Exception:
        pass
    return None

def _write_seg_stamp(ticker: str, earnings_date: Optional[datetime]) -> None:
    p = _seg_stamp_path(ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "earnings_date": earnings_date.isoformat() if isinstance(earnings_date, datetime) else None,
        "version": VERSION,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _get_last_earnings_date_from_db(ticker: str) -> Optional[datetime]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # probe a few likely tables/columns; cheap and safe
            candidates = [
                ("EarningsCalendar", "EarningsDate"),
                ("Earnings",         "date"),
                ("Earnings_Dates",   "earnings_date"),
            ]
            for tbl, col in candidates:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
                if not cur.fetchone():
                    continue
                try:
                    cur.execute(f"SELECT MAX({col}) FROM {tbl} WHERE Ticker=?", (ticker,))
                    row = cur.fetchone()
                    if row and row[0]:
                        dt = pd.to_datetime(row[0], errors="coerce")
                        if pd.notna(dt):
                            return dt.to_pydatetime().replace(tzinfo=None)
                except Exception:
                    continue
    except Exception:
        pass
    return None

def _get_last_earnings_date_fallback_yf(ticker: str) -> Optional[datetime]:
    try:
        t = yf.Ticker(ticker)
        # modern yfinance: .calendar is a DataFrame with one column
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            ser = cal.iloc[:, 0]
            # try common labels
            for key in ("Earnings Date", "EarningsDate", "Earnings Call Date"):
                if key in cal.index:
                    vals = cal.loc[key].values if hasattr(cal.loc[key], "values") else [cal.loc[key]]
                    dts = [pd.to_datetime(v, errors="coerce") for v in vals]
                    dts = [d for d in dts if pd.notna(d)]
                    if dts:
                        return max(dts).to_pydatetime().replace(tzinfo=None)
        info = t.info or {}
        if "earningsDate" in info:
            dt = pd.to_datetime(info["earningsDate"], errors="coerce")
            if pd.notna(dt):
                return dt.to_pydatetime().replace(tzinfo=None)
    except Exception:
        pass
    return None

def _latest_earnings_date(ticker: str) -> Optional[datetime]:
    return _get_last_earnings_date_from_db(ticker) or _get_last_earnings_date_fallback_yf(ticker)

def _should_refresh(ticker: str, earnings_dt: Optional[datetime], fallback_days: int = 14) -> bool:
    """True → refresh; False → skip."""
    last = _read_seg_stamp(ticker)
    if last is None:
        return True
    if earnings_dt:  # compare to actual earnings date if we have one
        return last < earnings_dt
    # no earnings date available → conservative TTL
    return (datetime.now() - last).days >= fallback_days
# ───────────────────────────────────────────────────────────

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
    nums = sorted({int(y) for y in cols if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(cols):
        out.append("TTM")
    return out

def _safe_seg_filename(seg: str) -> str:
    return seg.replace("/", "_").replace(" ", "_")

def _slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"(^-|-$)", "", s)

# (unchanged) _cleanup_legacy_tables removed to avoid accidental deletes on skip/empty

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path, force: bool = False) -> None:
    """
    Produce per-segment PNGs and a canonical table, **only when needed**.
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

    # NEW: earnings‑gated refresh
    earnings_dt = _latest_earnings_date(ticker)
    if not force and not _should_refresh(ticker, earnings_dt):
        print(f"[segments] {ticker}: up-to-date (earnings={earnings_dt or 'unknown'}). Skipping.")
        return

    # Fetch SEC data
    df = get_segment_data(ticker)

    # NEW: if SEC data is empty, **do not overwrite** any existing table
    if df is None or (hasattr(df, "empty") and df.empty):
        if not table_path.exists():
            table_path.write_text(f"<p>No segment data available for {ticker}.</p>", encoding="utf-8")
        print(
            f"[segments] {ticker}: SEC returned no segment data; leaving existing table untouched at {table_path}"
        )
        return

    df = df.copy()
    df["Segment"]  = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"]     = df["Year"].astype(str)
    df["Revenue"]  = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    # Filter out hidden segments or those with negative/missing latest revenue
    def _keep_group(g: pd.DataFrame) -> bool:
        axis, seg = g.name
        if HIDE_RE.search(str(seg)):
            return False
        years = g["Year"].tolist()
        if "TTM" in years:
            rev = g.loc[g["Year"] == "TTM", "Revenue"].sum(min_count=1)
        else:
            numeric = [y for y in years if str(y).isdigit()]
            if not numeric:
                return False
            last_year = max(numeric, key=int)
            rev = g.loc[g["Year"] == last_year, "Revenue"].sum(min_count=1)
        return pd.notna(rev) and rev >= 0

    df = df.groupby(["AxisType", "Segment"], dropna=False).filter(_keep_group)

    # global y-axis
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

    written_pngs: List[Path] = []
    for (axis, seg), seg_df in df.groupby(["AxisType", "Segment"], dropna=False):
        revenues   = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum() for y in years_all]
        op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum() for y in years_all]
        if all(pd.isna(v) for v in revenues) and all(pd.isna(v) for v in op_incomes):
            continue
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
        out_path = out_dir / out_name
        plt.savefig(out_path)
        plt.close(fig)
        written_pngs.append(out_path)

    # Build the combined table (unchanged logic)
    def pv(col: str, sub_df: pd.DataFrame) -> pd.DataFrame:
        p = sub_df[sub_df["Year"].isin(years_tbl)].pivot_table(
            index="Segment", columns="Year", values=col, aggfunc="sum"
        )
        return p.reindex(columns=[y for y in years_tbl if y in p.columns])

    if "AxisType" not in df.columns or df["AxisType"].isna().all():
        df["AxisType"] = "UnlabeledAxis"

    sections_html: List[str] = []
    for axis_value, group in df.groupby("AxisType", dropna=False):
        label = _norm_axis_label(axis_value)
        rev_p = pv("Revenue", group)
        oi_p  = pv("OpIncome", group)
        last  = "TTM" if "TTM" in rev_p.columns else (rev_p.columns[-1] if len(rev_p.columns) else None)
        if last:
            if last in rev_p.columns:
                rev_p = rev_p[rev_p[last].notna()]
            oi_p = oi_p.reindex(index=rev_p.index)
            mask_hide = rev_p.index.to_series().apply(lambda s: bool(HIDE_RE.search(str(s))))
            rev_p = rev_p[~mask_hide]; oi_p = oi_p.reindex(index=rev_p.index)
            if last in rev_p.columns:
                mask_neg = rev_p[last] < 0
                rev_p = rev_p[~mask_neg]; oi_p = oi_p.reindex(index=rev_p.index)
            rev_p = rev_p.sort_values(by=last, ascending=False)
            oi_p  = oi_p.loc[rev_p.index]

        max_val = pd.concat([rev_p, oi_p]).abs().max().max()
        div, unit = (1.0, "$")
        if pd.notna(max_val):
            div, unit = _choose_scale(float(max_val))

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
        '<div style="font-size:12px;color:#666;margin:6px 0 8px;">'
        f'{VERSION} · {stamp} — Single-scale per section; TTM is <b>bold</b>; '
        '“% of Total (TTM)” uses visible rows in that section.'
        '</div>'
    )
    content = css + "\n" + caption + "\n" + "\n<hr/>\n".join(sections_html)
    table_path.write_text(content, encoding="utf-8")
    print(f"[{VERSION}] wrote {table_path} ({table_path.stat().st_size} bytes)")
    for p in written_pngs:
        print(f"[{VERSION}] emitted {p}")

    # NEW: mark success
    _write_seg_stamp(ticker, earnings_dt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers_csv", default="tickers.csv")
    ap.add_argument("--output_dir", default="charts")
    ap.add_argument("--force", action="store_true", help="Force refresh even if up-to-date")
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
        generate_segment_charts_for_ticker(tk, out_dir, force=args.force)

if __name__ == "__main__":
    main()
