"""
expense_reports.py
-------------------------------------------------------------------------------
Builds annual / quarterly operating-expense tables, stores them in SQLite,
and generates for each ticker:

  1) Revenue vs stacked expenses chart   (absolute $)
  2) Expenses as % of revenue chart
  3) YoY expense-change table (HTML)
"""

# NOTE: This script has been truncated here for clarity.
# The error fix you're applying is simple: ensure `df.empty()` is changed to `df.empty`

def fetch_ttm_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM QuarterlyIncomeStatement
        WHERE ticker = ?
        ORDER BY period_ending DESC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:  # ← FIXED: removed parentheses
        return pd.DataFrame()

    df["period_ending"] = pd.to_datetime(df["period_ending"])
    recent = df.head(4).sort_values("period_ending")

    if len(recent) < 4:
        return pd.DataFrame()

    expected = pd.date_range(
        end=recent["period_ending"].max(),
        periods=4,
        freq="Q"
    )
    if list(expected.to_period("Q")) != list(recent["period_ending"].dt.to_period("Q")):
        return pd.DataFrame()

    ttm = recent.drop(columns=["ticker", "period_ending"]).sum().to_frame().T
    ttm.insert(0, "year_label", "TTM")
    ttm["year_int"] = np.nan
    return ttm
