# segment_formatting_helpers.py
# Formats the segment table BEFORE writing HTML (no filename changes)

import os
import re
import pandas as pd

# ────────── helpers ──────────

def _humanize_segment_name(raw: str) -> str:
    """Convert XBRL segment member names to readable labels."""
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


def _years_order(series) -> list:
    """Last 3 numeric fiscal years (ascending) + 'TTM' last if present."""
    nums = sorted({int(y) for y in series if str(y).isdigit()})
    nums = nums[-3:] if len(nums) > 3 else nums
    ordered = [str(y) for y in nums]
    if "TTM" in set(series):
        ordered.append("TTM")
    return ordered


# ────────── main API ──────────

def build_segment_table_html_formatted(df_raw: pd.DataFrame) -> str:
    """
    INPUT df_raw columns (labels can vary; we normalize to):
      - Segment
      - Year
      - Revenue
      - Operating Income
    OUTPUT: HTML for a single long table with:
      - humanized segment names
      - numbers formatted as $B with 1 decimal (e.g., $169.7B)
      - rows sorted by Segment; Year ordered (last 3 + TTM)
      - Year centered, numbers right-aligned (via CSS classes below)
    """
    # normalize headers
    lower = {c: str(c).strip().lower() for c in df_raw.columns}
    cmap = {}
    for c in df_raw.columns:
        lc = lower[c]
        if "segment" in lc:
            cmap[c] = "Segment"
        elif lc == "year":
            cmap[c] = "Year"
        elif "revenue" in lc:
            cmap[c] = "Revenue"
        elif "operating" in lc and ("income" in lc or "profit" in lc or "loss" in lc):
            cmap[c] = "Operating Income"
    df = df_raw.rename(columns=cmap)

    must = {"Segment", "Year", "Revenue", "Operating Income"}
    if not must.issubset(df.columns):
        # fallback: do not block build
        return df_raw.to_html(index=False, border=0)

    # clean + format
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"] = df["Year"].astype(str)
    df["Revenue_num"] = df["Revenue"].map(_to_float)
    df["OpInc_num"] = df["Operating Income"].map(_to_float)

    yr_order = _years_order(df["Year"])
    df["Year_cat"] = pd.Categorical(df["Year"], categories=yr_order, ordered=True)
    df = df.sort_values(["Segment", "Year_cat"]).drop(columns="Year_cat")

    fmt_b = lambda v: ("–" if pd.isna(v) else f"${v/1e9:.1f}B")
    df["Revenue"] = df["Revenue_num"].map(fmt_b)
    df["Operating Income"] = df["OpInc_num"].map(fmt_b)
    df = df[["Segment", "Year", "Revenue", "Operating Income"]]

    # HTML with classes used by your existing CSS
    html = df.to_html(index=False, escape=False, classes="segment-flat", border=0)
    caption = (
        '<div class="table-note">Values in <b>$B</b>. '
        'Years ordered as last 3 fiscal years + TTM.</div>'
    )
    return f'<div class="table-wrap">{html}</div>\n{caption}'



def write_formatted_segment_table(ticker: str, df_raw: pd.DataFrame, charts_dir: str = "charts"):
    """
    Writes charts/{ticker}/{ticker}_segments_table.html (same filename as before),
    but with formatted values and ordered rows/columns.
    """
    out_dir = os.path.join(charts_dir, ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_segments_table.html")
    html = build_segment_table_html_formatted(df_raw)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
