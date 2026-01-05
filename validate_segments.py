#!/usr/bin/env python3
"""
Quick validation for segment outputs.

Usage: python validate_segments.py
"""
from __future__ import annotations

import sys
from typing import Dict

import pandas as pd

from sec_segment_data_arelle import get_segment_data, get_company_totals
from generate_segment_charts import axis_has_meaningful_oi, _coverage

TICKERS = ["AAPL", "TSLA", "META", "AMZN", "GOOGL", "KO", "SBUX"]


def totals_map(df: pd.DataFrame) -> Dict[int, dict]:
    df_fy = df[df.get("PeriodType") == "FY"] if not df.empty else df
    return {
        int(r["Year"]): {"Revenue": r.get("Revenue"), "OpIncome": r.get("OpIncome"), "PeriodType": r.get("PeriodType")}
        for _, r in df_fy.iterrows()
        if pd.notna(r.get("Year"))
    }


def main() -> int:
    failures = []
    for ticker in TICKERS:
        print(f"=== {ticker} ===")
        seg = get_segment_data(ticker)
        totals = totals_map(get_company_totals(ticker))
        if seg is None or seg.empty:
            print(" no segment data")
            continue
        ttm_present = not seg[seg["Year"].astype(str) == "TTM"].empty
        print(f" ttm_present={ttm_present}")
        for axis_type, sub in seg.groupby("AxisType"):
            if len(set(sub["Segment"])) <= 1:
                continue
            years_fy = sorted({int(y) for y in sub[sub.get("PeriodType") == "FY"]["Year"] if str(y).isdigit()})
            latest_year = years_fy[-1] if years_fy else None
            cov = _coverage(sub, totals, latest_year) if latest_year else None
            oi_present = axis_has_meaningful_oi(sub)
            print(
                f" axis {axis_type}: members={len(set(sub['Segment']))} years={len(years_fy)} "
                f"coverage={cov if cov is not None else 'n/a'} oi_present={oi_present}"
            )
            if cov is not None and cov > 1.25:
                failures.append(f"{ticker} axis {axis_type} coverage too high {cov:.2f}")
            fy_rows = sub[sub.get("PeriodType") == "FY"]
            if not fy_rows.empty and fy_rows["PeriodType"].isna().any():
                failures.append(f"{ticker} axis {axis_type} has unclassified FY periods")
    if failures:
        print("\nValidation issues:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll segment validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
