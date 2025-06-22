import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# (Assuming OUTPUT_DIR is defined at module level)
# OUTPUT_DIR = "charts"


def _format_short(x, _pos=None, dec=1):
    """Callable for FuncFormatter – converts large numbers to K/M/B/T suffix."""
    if pd.isna(x):
        return "$0"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.{dec}f} T"
    if absx >= 1e9:
        return f"${x/1e9:.{dec}f} B"
    if absx >= 1e6:
        return f"${x/1e6:.{dec}f} M"
    if absx >= 1e3:
        return f"${x/1e3:.{dec}f} K"
    return f"${x:.{dec}f}"


def plot_expense_percent_chart(full: pd.DataFrame, ticker: str) -> None:
    """
    Plots operating expenses as a % of revenue, stacked by category,
    skipping any year where total_revenue == 0 to avoid division errors.
    """
    # Make a copy so we don't alter the caller's DataFrame
    full = full.copy()

    # Sort by the integer year key (TTM will have NaN and float to bottom)
    full.sort_values("year_int", inplace=True)

    # X-axis labels are the string labels (e.g. "2020", "TTM")
    x_labels = full["year_label"].tolist()

    # Decide which expense categories to plot
    use_combined = full["sga_combined"].notna().any()
    categories = [
        ("Cost of Revenue",     "cost_of_revenue",          "#6d6d6d"),
        ("R&D",                 "research_and_development", "blue"),
        ("G&A",                 "general_and_admin",        "#ffb3c6"),
        ("Selling & Marketing", "selling_and_marketing",    "#ffc6e2"),
        ("SG&A",                "sga_combined",             "#c2a5ff"),
        ("Facilities / D&A",    "facilities_da",            "orange"),
    ]
    if use_combined:
        # If SG&A is combined, drop the separate G&A and S&M
        categories = [
            (lbl, col, c) for lbl, col, c in categories
            if col not in ("general_and_admin", "selling_and_marketing")
        ]

    # ── Calculate percentages, skipping zero-revenue rows ────────────────────
    for _lbl, col, _c in categories:
        pct_col = col + "_pct"
        # initialize the column with NaN
        full[pct_col] = np.nan
        # create a mask of rows where revenue is non-zero
        mask = full["total_revenue"] != 0
        # compute percentage only on those rows
        full.loc[mask, pct_col] = (full.loc[mask, col] / full.loc[mask, "total_revenue"]) * 100

    # ── Plot stacked percent bars ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(full))
    for label, col, color in categories:
        vals = full[col + "_pct"].fillna(0).values
        ax.bar(x_labels, vals, bottom=bottoms, label=label, color=color, width=0.6)
        bottoms += vals
        # annotate slices larger than 4%
        for x, y0, val in zip(x_labels, bottoms - vals, vals):
            if val > 4:
                ax.text(
                    x, y0 + val / 2, f"{val:.1f} %",
                    ha="center", va="center", fontsize=8, color="white"
                )

    # ── Y-axis and reference line ────────────────────────────────────────────
    max_total = bottoms.max()
    ylim_max  = max(110, math.ceil(max_total / 10) * 10 + 10)
    ax.set_ylim(0, ylim_max)
    ax.axhline(100, linestyle="--", linewidth=1, color="black",
               label="100 % of revenue", zorder=5)

    ax.set_ylabel("Percent of Revenue")
    ax.set_title(f"Expenses as % of Revenue — {ticker}")
    ax.set_yticks(np.arange(0, ylim_max + 1, 10))
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_expenses_pct_of_rev.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[{ticker}] expense % chart saved to {out_path}")
