"""SPCX segment seed: frozen S-1/424B4 prospectus data must stay internally
consistent and must render through the segment pipeline (Nick, 2026-08-15)."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED_PATH = ROOT / "data" / "segment_seed_SPCX.json"

# Prospectus ground truth, $M (424B4 filed 2026-06-12, Note 19 tables).
TOTAL_REVENUE = {"2023": 10_387, "2024": 14_015, "2025": 18_674}
TOTAL_OPINC = {"2023": -3_505, "2024": 466, "2025": -2_589}
FY25_AI_RND = 5_064


def _seed():
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def test_seed_shape():
    seed = _seed()
    assert seed["mode"] == "replace"
    assert seed["units"] == "millions"
    rows = seed["rows"]
    assert len(rows) == 9  # 3 segments x FY2023-2025
    assert {r["Segment"] for r in rows} == {"Space", "Connectivity", "AI"}
    assert {r["Year"] for r in rows} == set(TOTAL_REVENUE)


def test_totals_match_prospectus():
    rows = _seed()["rows"]
    for year in TOTAL_REVENUE:
        yr = [r for r in rows if r["Year"] == year]
        assert sum(r["Revenue"] for r in yr) == TOTAL_REVENUE[year]
        assert sum(r["OpIncome"] for r in yr) == TOTAL_OPINC[year]
        for r in yr:
            # P&L identity per segment: revenue - total costs = op income
            assert r["Revenue"] - r["TotalCostsAndExpenses"] == r["OpIncome"]
    ai25 = next(r for r in rows if r["Segment"] == "AI" and r["Year"] == "2025")
    assert ai25["ResearchAndDevelopment"] == FY25_AI_RND


def test_seed_renders_tables(tmp_path):
    from generate_segment_charts import generate_segment_charts_for_ticker

    generate_segment_charts_for_ticker("SPCX", tmp_path)

    axis1 = (tmp_path / "axis1_SPCX_segments_table.html").read_text(encoding="utf-8")
    for needle in ("Connectivity", "AI", "Space", "2025", "2023",
                   "Segment Expense Breakdown", "Research and development"):
        assert needle in axis1, f"missing {needle!r} in axis1 table"
    # replace mode: no doubled iXBRL rows (Connectivity FY25 is $11.4B, not $23.7B)
    assert "23.7" not in axis1 and "11.4" in axis1
    # axis2 falls through to placeholder
    axis2 = (tmp_path / "axis2_SPCX_segments_table.html").read_text(encoding="utf-8")
    assert "No segment data" in axis2
