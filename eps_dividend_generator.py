import os
import sqlite3
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

DB_PATH = "Stock Data.db"
CHART_DIR = "charts"

def _build_chart(tic: str, conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    path = os.path.join(CHART_DIR, f"{tic}_eps_dividend_forecast.png")

    # Fetch trailing EPS (up to 10 years)
    cur.execute("""
        SELECT Date, EPS FROM Annual_Data
        WHERE Symbol=? ORDER BY Date ASC LIMIT 10;
    """, (tic,))
    trailing = [(int(d[:4]), float(eps) if eps is not None else 0.0)
                for d, eps in cur.fetchall()]

    # Fetch forward EPS (up to 3 years)
    cur.execute("""
        SELECT Date, ForwardEPS FROM ForwardFinancialData
        WHERE Ticker=? ORDER BY Date ASC LIMIT 3;
    """, (tic,))
    forward = [(int(d[:4]), float(v)) for d, v in cur.fetchall()]

    # Fetch dividend data for trailing years
    years = [y for y, _ in trailing]
    q = ",".join("?" * len(years)) if years else "NULL"
    cur.execute(f"""
        SELECT year, dividend FROM Dividends
        WHERE ticker=? AND year IN ({q});
    """, (tic, *years))
    div_map = {int(y): float(d) for y, d in cur.fetchall()}

    # Fetch TTM EPS and Dividend
    cur.execute("SELECT TTM_EPS, TTM_Dividend FROM TTM_Data WHERE Symbol=?;", (tic,))
    ttm_eps, ttm_div = cur.fetchone() or (0.0, 0.0)

    # Check if all dividends are zero
    if all(v == 0.0 for v in div_map.values()) and ttm_div == 0.0:
        # Generate fallback chart
        os.makedirs(CHART_DIR, exist_ok=True)
        plt.figure(figsize=(4, 2), dpi=100)
        plt.text(0.5, 0.5, "no dividend", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"📄 {tic}: No dividends – fallback chart saved.")
        return path

    # Assemble data for chart
    labels, eps_hist, eps_fwd, divs = [], [], [], []
    for yr, eps in trailing:
        labels.append(str(yr))
        eps_hist.append(eps)
        eps_fwd.append(0)
        divs.append(div_map.get(yr, 0.0))

    labels.append("TTM")
    eps_hist.append(ttm_eps)
    eps_fwd.append(0)
    divs.append(ttm_div)

    for yr, fwd in forward:
        labels.append(str(yr))
        eps_hist.append(0)
        eps_fwd.append(fwd)
        divs.append(0)

    # Plot
    x = range(len(labels))
    w = .25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    bars_eps_hist = ax.bar([i - w for i in x], eps_hist, w, label="Trailing EPS")
    bars_eps_fwd = ax.bar(x, eps_fwd, w, label="Forecast EPS", color="#70a6ff")
    bars_divs = ax.bar([i + w for i in x], divs, w, label="Dividend", color="orange")

    # Add data labels
    for bars in [bars_eps_hist, bars_eps_fwd, bars_divs]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("USD per share")
    ax.set_title(f"{tic} – EPS (Actual & Forecast) vs Dividend")
    ax.legend()

    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


# Top-level callable function
def generate_eps_dividend(tickers, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    paths = {}
    os.makedirs(CHART_DIR, exist_ok=True)

    for tic in tickers:
        try:
            print(f"🔧 Building chart for {tic}")
            chart_path = _build_chart(tic, conn)
            paths[tic] = chart_path
        except Exception as e:
            print(f"❌ Failed for {tic}: {e}")
    conn.close()
    return paths


# Mini-main for import or direct script call
def eps_dividend_generator():
    from ticker_manager import read_tickers
    tickers = read_tickers("tickers.csv")
    return generate_eps_dividend(tickers)

if __name__ == "__main__":
    print(eps_dividend_generator())
