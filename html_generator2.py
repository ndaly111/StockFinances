#!/usr/bin/env python3
# html_generator2.py  –  FULL FILE (final)
# -----------------------------------------------------------
# • Builds index.html, per-ticker pages, SPY/QQQ pages
# • Dashboard shows Implied-Growth Pctile column
# • All numeric cells carry data-order  →  perfect sorting
# -----------------------------------------------------------
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, pandas as pd, yfinance as yf

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ───────── helpers ─────────────────────────────────────────
def ensure_directory_exists(p):  os.makedirs(p, exist_ok=True) if p else None

def create_template(path, content):
    ensure_directory_exists(os.path.dirname(path))
    if not os.path.exists(path) or open(path, encoding="utf-8").read() != content:
        open(path, "w", encoding="utf-8").write(content)

def get_company_short_name(tk, cur):
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker=?", (tk,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    name = (yf.Ticker(tk).info or {}).get("shortName", "").strip() or tk
    cur.execute("UPDATE Tickers_Info SET short_name=? WHERE ticker=?", (name, tk))
    cur.connection.commit()
    return name

def get_file_or_placeholder(p, ph="No data available"):
    try:
        return open(p, encoding="utf-8").read()
    except FileNotFoundError:
        return ph

# ───────── template creation (home only shown) ─────────────
def ensure_templates_exist():
    home_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>
    td.positive{color:green;} td.negative{color:red;}
    td.pct::after{content:'%';}                 /* add visible percent sign */
    .center-table{margin:0 auto;width:100%%}
  </style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(function(){
      $('#sortable-table').DataTable({
        pageLength:100,
        scrollX:true,
        createdRow:function(row){
          $('td',row).each(function(){
            if(!$(this).attr('data-order')) return;
            var n = parseFloat($(this).data('order'));
            if(isNaN(n)) return;
            var col = $(this).index();
            if(col===6){ $(this).addClass(n<50 ? 'negative':'positive'); } /* pctile */
            else if(col>=2 && col<=5){ $(this).addClass(n<0 ? 'negative':'positive'); }
          });
        }
      });
    });
  </script>
</head><body>
<div class="container">
  <header><h1>Financial Overview</h1></header>

  <nav class="navigation">
    {% for t in tickers %}
      <a href="pages/{{t}}_page.html" class="home-button">{{t}}</a>{% if not loop.last %} | {% endif %}
    {% endfor %}
  </nav>

  <div id="spy-qqq-growth" class="center-table">
    <h2>SPY vs QQQ Overview</h2>
    {{ spy_qqq_growth | safe }}
  </div>

  <div class="center-table">
    <h2>Past Earnings (Last 7 Days)</h2>
    {{ earnings_past | safe }}
    <h2>Upcoming Earnings</h2>
    {{ earnings_upcoming | safe }}
  </div>

  <div>{{ dashboard_table | safe }}</div>

  <footer><p>Nick's Financial Data Dashboard</p></footer>
</div></body></html>"""
    create_template("templates/home_template.html", home_tpl)
    # ticker_template.html and spy/qqq templates remain unchanged.

# ───────── dashboard builder (manual HTML) ─────────────────
def generate_dashboard_table(raw_rows):
    base_cols = ["Ticker", "Share Price",
                 "Nick's TTM Value", "Nick's Forward Value",
                 "Finviz TTM Value", "Finviz Forward Value"]

    df = pd.DataFrame(raw_rows, columns=base_cols)

    # latest growth percentile (TTM)
    with sqlite3.connect(DB_PATH) as conn:
        pct = pd.read_sql_query(
            """SELECT Ticker, Percentile FROM Index_Growth_Pctile
               WHERE Growth_Type='TTM'
                 AND Date=(SELECT MAX(Date) FROM Index_Growth_Pctile)""", conn)
    df = df.merge(pct, how="left", on="Ticker")

    # ---- share-price helpers ----
    sp_num = pd.to_numeric(df["Share Price"], errors="coerce")
    df["Share Price_num"]  = sp_num
    df["Share Price_disp"] = sp_num.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")

    # ---- % column helpers ----
    pct_cols = base_cols[2:]
    for col in pct_cols:
        num = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce')
        df[col+"_num"]  = num
        df[col+"_disp"] = num.map(lambda x: f"{x:.1f}" if pd.notnull(x) else "–")

    # implied-growth percentile helpers
    df["Implied-Growth Pctile_num"]  = df["Percentile"]
    df["Implied-Growth Pctile_disp"] = df["Percentile"].map(
        lambda x: f"{x:.0f}" if pd.notnull(x) else "–")
    df.drop(columns="Percentile", inplace=True)

    # hyperlink ticker
    def link(t):
        return f'<a href="{"spy_growth.html" if t=="SPY" else "qqq_growth.html" if t=="QQQ" else f"pages/{t}_page.html"}">{t}</a>'
    df["Ticker"] = df["Ticker"].apply(link)

    # default sort by Nick's TTM %
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    # ---- build the table body ----
    body_rows = []
    for _, r in df.iterrows():
        cells = []

        # ticker
        cells.append(f"<td>{r['Ticker']}</td>")

        # share price
        p_num, p_disp = r["Share Price_num"], r["Share Price_disp"]
        p_key = p_num if pd.notnull(p_num) else -999
        cells.append(f'<td data-order="{p_key}">{p_disp}</td>')

        # six percentage columns
        for col in pct_cols:
            num, disp = r[col+"_num"], r[col+"_disp"]
            key = num if pd.notnull(num) else -999
            cells.append(f'<td class="pct" data-order="{key}">{disp}</td>')

        # implied-growth percentile
        num, disp = r["Implied-Growth Pctile_num"], r["Implied-Growth Pctile_disp"]
        key = num if pd.notnull(num) else -999
        cells.append(f'<td class="pct" data-order="{key}">{disp}</td>')

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    headers = base_cols + ["Implied-Growth Pctile"]
    thead   = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    dash_html = (
        '<table id="sortable-table" class="table table-striped" style="width:100%">'
        + thead + "<tbody>" + "".join(body_rows) + "</tbody></table>"
    )

    # ---- summary rows (unchanged) ----
    pc = lambda s: f"{s:.1f}" if pd.notnull(s) else "–"
    ttm, fwd = df["Nick's TTM Value_num"].dropna(),   df["Nick's Forward Value_num"].dropna()
    fttm,ffwd= df["Finviz TTM Value_num"].dropna(),   df["Finviz Forward Value_num"].dropna()
    summary=[["Average", pc(ttm.mean()), pc(fwd.mean()), pc(fttm.mean()), pc(ffwd.mean())],
             ["Median",  pc(ttm.median()),pc(fwd.median()),pc(fttm.median()),pc(ffwd.median())]]
    avg_html=pd.DataFrame(summary, columns=["Metric"]+pct_cols).to_html(
        index=False, classes="table table-striped", escape=False)

    ensure_directory_exists("charts")
    open("charts/dashboard.html","w",encoding="utf-8").write(avg_html+dash_html)

    return avg_html+dash_html, {
        "Nicks_TTM_Value_Average":ttm.mean(),"Nicks_TTM_Value_Median":ttm.median(),
        "Nicks_Forward_Value_Average":fwd.mean(),"Nicks_Forward_Value_Median":fwd.median(),
        "Finviz_TTM_Value_Average":fttm.mean() if not fttm.empty else None,
        "Finviz_TTM_Value_Median":fttm.median() if not fttm.empty else None,
        "Finviz_Forward_Value_Average":ffwd.mean() if not ffwd.empty else None,
        "Finviz_Forward_Value_Median":ffwd.median() if not ffwd.empty else None
    }

# ───────── ancillary page builders (unchanged) ────────────
def render_spy_qqq_growth_pages():
    chart_dir,out_dir = "charts","."
    for key in ("spy","qqq"):
        tpl=Template(get_file_or_placeholder(f"templates/{key}_growth_template.html"))
        rendered=tpl.render(**{
            f"{key}_growth_summary":get_file_or_placeholder(f"{chart_dir}/{key}_growth_summary.html"),
            f"{key}_pe_summary":    get_file_or_placeholder(f"{chart_dir}/{key}_pe_summary.html")})
        open(f"{out_dir}/{key}_growth.html","w",encoding="utf-8").write(rendered)

def prepare_and_generate_ticker_pages(tickers,charts_dir="charts"):
    ensure_directory_exists("pages")
    with sqlite3.connect(DB_PATH) as conn:
        cur=conn.cursor()
        for t in tickers:
            d={
              "ticker":t,"company_name":get_company_short_name(t,cur),
              "ticker_info":get_file_or_placeholder(f"{charts_dir}/{t}_ticker_info.html"),
              "revenue_net_income_chart_path":f"{charts_dir}/{t}_revenue_net_income_chart.png",
              "eps_chart_path":f"{charts_dir}/{t}_eps_chart.png",
              "financial_table":get_file_or_placeholder(f"{charts_dir}/{t}_rev_net_table.html"),
              "forecast_rev_net_chart_path":f"{charts_dir}/{t}_Revenue_Net_Income_Forecast.png",
              "forecast_eps_chart_path":f"{charts_dir}/{t}_EPS_Forecast.png",
              "yoy_growth_table_html":get_file_or_placeholder(f"{charts_dir}/{t}_yoy_growth_tbl.html"),
              "expense_chart_path":f"{charts_dir}/{t}_rev_expense_chart.png",
              "expense_percent_chart_path":f"{charts_dir}/{t}_expense_percent_chart.png",
              "expense_abs_html":get_file_or_placeholder(f"{charts_dir}/{t}_expense_absolute.html"),
              "expense_yoy_html":get_file_or_placeholder(f"{charts_dir}/{t}_yoy_expense_change.html"),
              "balance_sheet_chart_path":f"{charts_dir}/{t}_balance_sheet_chart.png",
              "balance_sheet_table_html":get_file_or_placeholder(f"{charts_dir}/{t}_balance_sheet_table.html"),
              "revenue_yoy_change_chart_path":f"{charts_dir}/{t}_revenue_yoy_change.png",
              "eps_yoy_change_chart_path":f"{charts_dir}/{t}_eps_yoy_change.png",
              "valuation_chart":f"{charts_dir}/{t}_valuation_chart.png",
              "valuation_info_table":get_file_or_placeholder(f"{charts_dir}/{t}_valuation_info.html"),
              "valuation_data_table":get_file_or_placeholder(f"{charts_dir}/{t}_valuation_table.html"),
              "unmapped_expense_html":get_file_or_placeholder(f"{charts_dir}/{t}_unmapped_fields.html","No unmapped expenses."),
              "eps_dividend_chart_path":f"{charts_dir}/{t}_eps_dividend_forecast.png",
              "implied_growth_chart_path":f"{charts_dir}/{t}_implied_growth_plot.png",
              "implied_growth_table_html":get_file_or_placeholder(
                    f"{charts_dir}/{t}_implied_growth_summary.html","No implied growth data available.")}
            open(f"pages/{t}_page.html","w",encoding="utf-8")\
                .write(env.get_template("ticker_template.html").render(ticker_data=d))

def create_home_page(tickers,dashboard_html,avg_vals,spy_qqq_html,
                     earnings_past="",earnings_upcoming=""):
    tpl=env.get_template("home_template.html")
    open("index.html","w",encoding="utf-8").write(
        tpl.render(tickers=tickers, dashboard_table=dashboard_html,
                   dashboard_data=avg_vals, spy_qqq_growth=spy_qqq_html,
                   earnings_past=earnings_past, earnings_upcoming=earnings_upcoming))

# ───────── main wrapper (matches main_remote.py) ───────────
def html_generator2(tickers,financial_data,full_dashboard_html,
                    avg_values,spy_qqq_growth_html=""):

    ensure_templates_exist()
    create_home_page(
        tickers, full_dashboard_html, avg_values, spy_qqq_growth_html,
        get_file_or_placeholder("charts/earnings_past.html"),
        get_file_or_placeholder("charts/earnings_upcoming.html")
    )
    prepare_and_generate_ticker_pages(tickers)
    render_spy_qqq_growth_pages()

# -----------------------------------------------------------
if __name__ == "__main__":
    print("html_generator2 is meant to be called from main_remote.py")
