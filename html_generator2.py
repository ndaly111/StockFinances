#!/usr/bin/env python3
# html_generator2.py — layout-safe sequential segments (NO tabs), single-path table
# ----------------------------------------------------------------
from jinja2 import Environment, FileSystemLoader
import os, sqlite3, pandas as pd, re

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ───────── helpers (layout-safe) ───────────────────────────────
def get_file_or_placeholder(path: str, ph: str = "No data available") -> str:
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return ph

def get_segment_table_html(ticker: str, charts_dir_fs: str) -> str:
    """Single canonical read path: {charts_dir_fs}/{T}/{T}_segments_table.html"""
    path = f"{charts_dir_fs}/{ticker}/{ticker}_segments_table.html"
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return f"No segment data available for {ticker}."

# ───────── per-axis carousels (non-invasive) ───────────────────
_SLUG_TO_LABEL = {
    "products-services": "Products / Services",
    "product-line":      "Products / Services",
    "product":           "Products / Services",
    "product-category":  "Products / Services",
    "regions":           "Regions",
    "geographical-areas":"Regions",
    "geographical-regions":"Regions",
    "domestic-vs-foreign":"Domestic vs Foreign",
    "country":           "Country",
    "operating-segments":"Operating Segments",
    "major-customers":   "Major Customers",
    "sales-channels":    "Sales Channels",
    "unlabeled-axis":    "Unlabeled Axis",
}
_slug_pat = re.compile(r'^(?P<tkr>[A-Za-z0-9]+)_(?P<slug>[a-z0-9-]+)_.+\.png$', re.IGNORECASE)

def _group_segment_images_by_label(ticker: str, charts_dir_fs: str):
    seg_dir = os.path.join(charts_dir_fs, ticker)
    by_label, legacy = {}, []
    if not os.path.isdir(seg_dir):
        return by_label, legacy
    for f in sorted(os.listdir(seg_dir)):
        if not f.lower().endswith(".png"):
            continue
        m = _slug_pat.match(f)
        if not m or m.group("tkr").upper() != ticker.upper():
            legacy.append(f); continue
        lab = _SLUG_TO_LABEL.get((m.group("slug") or "").lower())
        if lab:
            by_label.setdefault(lab.lower(), []).append(f)
        else:
            legacy.append(f)
    return by_label, legacy

def _build_carousel_html_for_label(label_lower: str, ticker: str, charts_dir_fs: str, charts_dir_web: str):
    imgs_by_label, legacy = _group_segment_images_by_label(ticker, charts_dir_fs)
    files = imgs_by_label.get(label_lower, [])
    if not files and label_lower == "segments":
        files = legacy
    if not files:
        return ""
    items = [
        f'<div class="carousel-item"><img class="chart-img" src="{charts_dir_web}/{ticker}/{fn}" alt="{fn}"></div>'
        for fn in files
    ]
    return '<div class="carousel-container seg-carousel">\n' + "\n".join(items) + "\n</div>"

# ───────── sequential sections (carousel → table) ──────────────
def build_segment_sections_sequential(ticker: str, charts_dir_fs: str, charts_dir_web: str, raw_table_html: str) -> str:
    """
    For each <h3>Axis Name</h3> in the combined table:
      [carousel for that axis] then [that axis table]
    If the combined table is missing, render image-only sections per axis.
    """
    imgs_by_label, legacy = _group_segment_images_by_label(ticker, charts_dir_fs)
    out_parts = []

    sections, preface = [], ""
    if isinstance(raw_table_html, str) and raw_table_html.strip() and "No segment data available" not in raw_table_html:
        h3 = re.compile(r"<h3>(.*?)</h3>", re.IGNORECASE | re.DOTALL)
        pos = 0
        first = h3.search(raw_table_html, pos)
        if first:
            preface = raw_table_html[:first.start()].strip()
            while True:
                m = h3.search(raw_table_html, pos)
                if not m: break
                title = (m.group(1) or "").strip()
                start = m.end()
                m2 = h3.search(raw_table_html, start)
                end = m2.start() if m2 else len(raw_table_html)
                body = raw_table_html[start:end].strip()
                body = re.sub(r"^\s*(<hr\s*/?>)+", "", body, flags=re.IGNORECASE)
                sections.append((title, body))
                pos = end

    if preface:
        out_parts.append(preface)

    if sections:
        seen = set()
        for title, body in sections:
            label_lower = title.strip().lower()
            carousel = _build_carousel_html_for_label(label_lower, ticker, charts_dir_fs, charts_dir_web)
            has_wrap = ("class='table-wrap'" in body) or ('class="table-wrap"' in body)
            sec = ['<div class="seg-section">', f"<h3>{title}</h3>"]
            if carousel: sec.append(carousel)
            sec.append(body if has_wrap else f'<div class="table-wrap">{body}</div>')
            sec.append("</div>")
            out_parts.append("\n".join(sec))
            seen.add(label_lower)

        # image-only axes with no table section found
        for lab in sorted(imgs_by_label.keys()):
            if lab in seen: continue
            carousel = _build_carousel_html_for_label(lab, ticker, charts_dir_fs, charts_dir_web)
            if carousel:
                out_parts.append(
                    "<div class=\"seg-section\">\n"
                    f"<h3>{lab.title()}</h3>\n"
                    f"{carousel}\n"
                    "<div class=\"table-wrap\"><p>No table for this axis.</p></div>\n"
                    "</div>"
                )
    else:
        # no table found → image-only sections
        for lab in sorted(imgs_by_label.keys()):
            carousel = _build_carousel_html_for_label(lab, ticker, charts_dir_fs, charts_dir_web)
            if carousel:
                out_parts.append(
                    "<div class=\"seg-section\">\n"
                    f"<h3>{lab.title()}</h3>\n"
                    f"{carousel}\n"
                    "<div class=\"table-wrap\"><p>No table for this axis.</p></div>\n"
                    "</div>"
                )
        if legacy:
            carousel = _build_carousel_html_for_label("segments", ticker, charts_dir_fs, charts_dir_web)
            if carousel:
                out_parts.append(
                    "<div class=\"seg-section\">\n"
                    "<h3>Segments</h3>\n"
                    f"{carousel}\n"
                    "<div class=\"table-wrap\"><p>No table for this axis.</p></div>\n"
                    "</div>"
                )
        if not out_parts:
            return raw_table_html or ""

    return "\n".join(out_parts)

# ───────── page generation (does not overwrite your templates) ─
def prepare_and_generate_ticker_pages(tickers, charts_dir_fs="charts"):
    charts_dir_web = "../" + charts_dir_fs
    os.makedirs("pages", exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for t in tickers:
            table_raw = get_segment_table_html(t, charts_dir_fs)
            seg_html  = build_segment_sections_sequential(t, charts_dir_fs, charts_dir_web, table_raw)

            d = {
                "ticker":                        t,
                "company_name":                  get_company_short_name(t, cur),

                # existing fragments you already render
                "ticker_info":                   get_file_or_placeholder(f"{charts_dir_fs}/{t}_ticker_info.html"),
                "financial_table":               get_file_or_placeholder(f"{charts_dir_fs}/{t}_rev_net_table.html"),
                "yoy_growth_table_html":         get_file_or_placeholder(f"{charts_dir_fs}/{t}_yoy_growth_tbl.html"),
                "balance_sheet_table_html":      get_file_or_placeholder(f"{charts_dir_fs}/{t}_balance_sheet_table.html"),
                "valuation_info_table":          get_file_or_placeholder(f"{charts_dir_fs}/{t}_valuation_info.html"),
                "valuation_data_table":          get_file_or_placeholder(f"{charts_dir_fs}/{t}_valuation_table.html"),
                "expense_abs_html":              get_file_or_placeholder(f"{charts_dir_fs}/{t}_expense_absolute.html"),
                "expense_yoy_html":              get_file_or_placeholder(f"{charts_dir_fs}/{t}_yoy_expense_change.html"),
                "unmapped_expense_html":         get_file_or_placeholder(f"{charts_dir_fs}/{t}_unmapped_fields.html", "No unmapped expenses."),
                "implied_growth_table_html":     get_file_or_placeholder(f"{charts_dir_fs}/{t}_implied_growth_summary.html", "No implied growth data available."),

                # NEW sequential block (use either key depending on your template)
                "segment_sections_html":         seg_html,
                "segment_table_html":            seg_html,  # fills old placeholder {{ segment_table_html | safe }}

                # charts (web paths)
                "revenue_net_income_chart_path": f"{charts_dir_web}/{t}_revenue_net_income_chart.png",
                "eps_chart_path":                f"{charts_dir_web}/{t}_eps_chart.png",
                "forecast_rev_net_chart_path":   f"{charts_dir_web}/{t}_Revenue_Net_Income_Forecast.png",
                "forecast_eps_chart_path":       f"{charts_dir_web}/{t}_EPS_Forecast.png",
                "revenue_yoy_change_chart_path": f"{charts_dir_web}/{t}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":     f"{charts_dir_web}/{t}_eps_yoy_change.png",
                "expense_chart_path":            f"{charts_dir_web}/{t}_rev_expense_chart.png",
                "expense_percent_chart_path":    f"{charts_dir_web}/{t}_expense_percent_chart.png",
                "balance_sheet_chart_path":      f"{charts_dir_web}/{t}_balance_sheet_chart.png",
                "valuation_chart":               f"{charts_dir_web}/{t}_valuation_chart.png",
                "eps_dividend_chart_path":       f"{charts_dir_web}/{t}_eps_dividend_forecast.png",
                "implied_growth_chart_path":     f"{charts_dir_web}/{t}_implied_growth_plot.png",
            }

            # Use your existing templates as-is (no overwrites, no global CSS injection)
            tpl = env.get_template("ticker_template.html")
            rendered = tpl.render(ticker_data=d)
            with open(f"pages/{t}_page.html", "w", encoding="utf-8") as f:
                f.write(rendered)

def create_home_page(*_args, **_kwargs):
    """No-op here — keep using your existing home template and builder."""
    pass

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    # Respect your current pages; do not override templates or CSS
    prepare_and_generate_ticker_pages(tickers)
