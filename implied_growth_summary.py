import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os

def generate_implied_growth_chart(df, ticker):
    df_ttm = df[(df["ticker"] == ticker) & (df["growth_type"] == "TTM")].copy()
    df_fwd = df[(df["ticker"] == ticker) & (df["growth_type"] == "Forward")].copy()

    if df_ttm.empty or df_fwd.empty:
        print(f"Not enough data for {ticker}, skipping chart.")
        return None

    df_ttm.sort_values("date_recorded", inplace=True)
    df_fwd.sort_values("date_recorded", inplace=True)

    dates = pd.to_datetime(df_ttm["date_recorded"])
    ttm_vals = df_ttm["growth_value"]
    fwd_vals = df_fwd.set_index("date_recorded").reindex(df_ttm["date_recorded"])["growth_value"]

    # Stats for TTM
    mean_val = ttm_vals.mean()
    median_val = ttm_vals.median()
    std_val = ttm_vals.std()

    upper_band = mean_val + std_val
    lower_band = mean_val - std_val

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ttm_vals, label="TTM Implied Growth", color="blue", linewidth=1.5)
    ax.plot(dates, fwd_vals, label="Forward Implied Growth", color="green", linewidth=1.5)

    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1, label="TTM Avg")
    ax.axhline(median_val, color="gray", linestyle=":", linewidth=1, label="TTM Median")
    ax.axhline(upper_band, color="lightgray", linestyle="-", linewidth=1, label="+1 Std Dev")
    ax.axhline(lower_band, color="lightgray", linestyle="-", linewidth=1, label="-1 Std Dev")

    ax.set_title(f"{ticker} Implied Growth History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()

    file_path = f"charts/{ticker}_implied_growth_plot.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    return file_path
