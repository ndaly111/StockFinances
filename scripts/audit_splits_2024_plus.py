#!/usr/bin/env python3
"""
Audit split events since 2024 against stored EPS and recorded split metadata.

The report highlights tickers where split-driven EPS adjustments appear missing
or mismatched so operators can run the split-adjustment utility in a targeted
way and re-run the audit until no issues remain.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from split_utils import ensure_splits_table, fetch_split_history

DEFAULT_DB_PATH = "Stock Data.db"
DEFAULT_START_DATE = date(2024, 1, 1)
DEFAULT_CSV = "split_audit_2024_plus.csv"
DEFAULT_HTML = "split_audit_2024_plus.html"


@dataclass
class AnnualEpsPoint:
    as_of: date
    eps: float


@dataclass
class TtmSnapshot:
    eps: float | None
    quarter: date | None


@dataclass
class AuditResult:
    ticker: str
    split_date: date
    ratio: float
    recorded: bool
    adjustment_present: bool
    observed_ratio: float | None
    before_date: date | None
    before_eps: float | None
    after_date: date | None
    after_eps: float | None
    after_source: str
    eps_status: str
    recommendation: str
    note: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit split adjustments since 2024.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite database path to inspect.")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help="Earliest split date to include (YYYY-MM-DD, default: 2024-01-01).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional list of tickers to audit. When omitted, all tickers in Annual_Data are used.",
    )
    parser.add_argument("--output-csv", default=DEFAULT_CSV, help="Path for the CSV summary.")
    parser.add_argument("--output-html", default=DEFAULT_HTML, help="Path for the HTML summary.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Allowed relative deviation between observed and expected EPS ratios (default: 0.25 = 25%%).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def _as_date(raw: str | date | None) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return datetime.fromisoformat(str(raw)).date()
    except Exception:
        return None


def _as_float(raw) -> float | None:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _load_tickers(cur: sqlite3.Cursor, provided: Sequence[str] | None) -> List[str]:
    if provided:
        return sorted({t.upper() for t in provided})
    cur.execute("SELECT DISTINCT Symbol FROM Annual_Data ORDER BY Symbol;")
    return [row[0] for row in cur.fetchall()]


def _load_annual_eps(cur: sqlite3.Cursor, tickers: Iterable[str]) -> Dict[str, List[AnnualEpsPoint]]:
    eps_map: Dict[str, List[AnnualEpsPoint]] = {t: [] for t in tickers}
    cur.execute("SELECT Symbol, Date, EPS FROM Annual_Data;")
    for symbol, raw_date, eps in cur.fetchall():
        if symbol not in eps_map:
            continue
        dt = _as_date(raw_date)
        val = _as_float(eps)
        if dt is None or val is None:
            continue
        eps_map[symbol].append(AnnualEpsPoint(dt, val))
    for symbol, points in eps_map.items():
        points.sort(key=lambda p: p.as_of)
    return eps_map


def _load_ttm(cur: sqlite3.Cursor, tickers: Iterable[str]) -> Dict[str, TtmSnapshot]:
    ttm_map: Dict[str, TtmSnapshot] = {}
    cur.execute("SELECT Symbol, TTM_EPS, Quarter FROM TTM_Data;")
    for symbol, eps, quarter in cur.fetchall():
        if symbol not in tickers:
            continue
        ttm_map[symbol] = TtmSnapshot(_as_float(eps), _as_date(quarter))
    return ttm_map


def _load_recorded_splits(cur: sqlite3.Cursor, start_date: date) -> Dict[str, Dict[date, float]]:
    ensure_splits_table(cur)
    recorded: Dict[str, Dict[date, float]] = {}
    cur.execute("SELECT Symbol, Date, Ratio FROM Splits WHERE Date >= ?;", (start_date.isoformat(),))
    for symbol, raw_date, ratio in cur.fetchall():
        dt = _as_date(raw_date)
        val = _as_float(ratio)
        if dt is None or val is None:
            continue
        recorded.setdefault(symbol, {})[dt] = val
    return recorded


def _nearest_eps(points: Sequence[AnnualEpsPoint], split_date: date) -> Tuple[AnnualEpsPoint | None, AnnualEpsPoint | None]:
    before = None
    after = None
    for pt in points:
        if pt.as_of < split_date:
            before = pt
        elif pt.as_of >= split_date:
            after = pt
            break
    return before, after


def _eps_ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before == 0 or after == 0:
        return None
    if (before < 0) != (after < 0):
        return None
    try:
        return abs(before) / abs(after)
    except Exception:
        return None


def _matches_expected(observed: float | None, expected: float, tolerance: float) -> bool:
    if observed is None:
        return False
    if expected == 0:
        return False
    return abs(observed - expected) / expected <= tolerance


def _recommendation(recorded: bool, eps_status: str) -> str:
    if eps_status == "adjusted":
        return "skip" if recorded else "verify"
    if eps_status == "unadjusted":
        return "apply split"
    if eps_status == "mismatch":
        return "verify"
    return "verify"


def _analyze_event(
    ticker: str,
    split_date: date,
    ratio: float,
    recorded: bool,
    annual_points: Sequence[AnnualEpsPoint],
    ttm_snapshot: TtmSnapshot | None,
    tolerance: float,
) -> AuditResult:
    before, after = _nearest_eps(annual_points, split_date)
    after_source = "n/a"
    if after is not None:
        after_source = "annual"
    if (
        after is None
        and ttm_snapshot
        and ttm_snapshot.eps is not None
        and ttm_snapshot.quarter
        and ttm_snapshot.quarter >= split_date
    ):
        after = AnnualEpsPoint(ttm_snapshot.quarter, ttm_snapshot.eps)
        after_source = "ttm"

    observed_ratio = _eps_ratio(before.eps if before else None, after.eps if after else None)

    if observed_ratio is None:
        eps_status = "inconclusive"
        note = "Insufficient EPS continuity to assess adjustment."
    elif _matches_expected(observed_ratio, 1.0, tolerance):
        eps_status = "adjusted"
        note = f"Observed EPS ratio ~{observed_ratio:.2f}, suggesting prior split adjustment."
    elif _matches_expected(observed_ratio, ratio, tolerance):
        eps_status = "unadjusted"
        note = (
            f"Observed EPS ratio ~{observed_ratio:.2f} aligns with split ratio {ratio:.2f}; data likely unadjusted."
        )
    else:
        eps_status = "mismatch"
        note = f"Observed EPS ratio {observed_ratio:.2f} differs from both 1.0 and expected {ratio:.2f}."

    detail_bits = []
    if before:
        detail_bits.append(f"before {before.as_of.isoformat()}={before.eps:.4f}")
    else:
        detail_bits.append("before missing")
    if after:
        detail_bits.append(f"after ({after_source}) {after.as_of.isoformat()}={after.eps:.4f}")
    else:
        detail_bits.append("after missing")
    note = f"{note} [{'; '.join(detail_bits)}]"

    recommendation = _recommendation(recorded, eps_status)
    return AuditResult(
        ticker=ticker,
        split_date=split_date,
        ratio=ratio,
        recorded=recorded,
        adjustment_present=eps_status == "adjusted",
        observed_ratio=observed_ratio,
        before_date=before.as_of if before else None,
        before_eps=before.eps if before else None,
        after_date=after.as_of if after else None,
        after_eps=after.eps if after else None,
        after_source=after_source,
        eps_status=eps_status,
        recommendation=recommendation,
        note=note,
    )


def _write_csv(path: Path, rows: Sequence[AuditResult]) -> None:
    headers = [
        "Ticker",
        "Split Date",
        "Ratio",
        "Recorded In DB",
        "Adjustment Present",
        "Observed EPS Ratio",
        "Before EPS Date",
        "Before EPS",
        "After EPS Date",
        "After EPS",
        "After Source",
        "EPS Status",
        "Recommendation",
        "Notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row.ticker,
                    row.split_date.isoformat(),
                    f"{row.ratio:.4f}",
                    "yes" if row.recorded else "no",
                    "yes" if row.adjustment_present else "no",
                    f"{row.observed_ratio:.4f}" if row.observed_ratio is not None else "",
                    row.before_date.isoformat() if row.before_date else "",
                    f"{row.before_eps:.4f}" if row.before_eps is not None else "",
                    row.after_date.isoformat() if row.after_date else "",
                    f"{row.after_eps:.4f}" if row.after_eps is not None else "",
                    row.after_source,
                    row.eps_status,
                    row.recommendation,
                    row.note,
                ]
            )


def _write_html(path: Path, rows: Sequence[AuditResult]) -> None:
    headers = [
        "Ticker",
        "Split Date",
        "Ratio",
        "Recorded In DB",
        "Adjustment Present",
        "Observed EPS Ratio",
        "Before EPS Date",
        "Before EPS",
        "After EPS Date",
        "After EPS",
        "After Source",
        "EPS Status",
        "Recommendation",
        "Notes",
    ]
    with path.open("w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Split Audit (2024+)</title>")
        f.write(
            "<style>body{font-family:Arial,sans-serif;margin:1.5rem;}table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ddd;padding:8px;}th{background:#f4f4f4;text-align:left;}tr:nth-child(even){background:#fafafa;}"
            ".warn{color:#c45800;font-weight:bold;}.ok{color:#1b6a1b;font-weight:bold;}"
            "</style></head><body>"
        )
        f.write("<h1>Split Audit (2024+)</h1>")
        f.write(f"<p>Total events: {len(rows)}</p>")
        f.write("<table><thead><tr>")
        for h in headers:
            f.write(f"<th>{h}</th>")
        f.write("</tr></thead><tbody>")
        for row in rows:
            status_class = "ok" if row.adjustment_present else "warn"
            f.write("<tr>")
            f.write(f"<td>{row.ticker}</td>")
            f.write(f"<td>{row.split_date.isoformat()}</td>")
            f.write(f"<td>{row.ratio:.4f}</td>")
            f.write(f"<td>{'yes' if row.recorded else 'no'}</td>")
            f.write(f"<td class='{status_class}'>{'yes' if row.adjustment_present else 'no'}</td>")
            f.write(f"<td>{'' if row.observed_ratio is None else f'{row.observed_ratio:.4f}'}</td>")
            f.write(f"<td>{row.before_date.isoformat() if row.before_date else ''}</td>")
            f.write(f"<td>{'' if row.before_eps is None else f'{row.before_eps:.4f}'}</td>")
            f.write(f"<td>{row.after_date.isoformat() if row.after_date else ''}</td>")
            f.write(f"<td>{'' if row.after_eps is None else f'{row.after_eps:.4f}'}</td>")
            f.write(f"<td>{row.after_source}</td>")
            f.write(f"<td>{row.eps_status}</td>")
            f.write(f"<td>{row.recommendation}</td>")
            f.write(f"<td>{row.note}</td>")
            f.write("</tr>")
        f.write("</tbody></table></body></html>")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    try:
        start_date = datetime.fromisoformat(args.start_date).date()
    except Exception as exc:
        raise SystemExit(f"Invalid start date '{args.start_date}'. Expected YYYY-MM-DD.") from exc

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    tickers = _load_tickers(cur, args.tickers)
    annual_eps = _load_annual_eps(cur, tickers)
    ttm_map = _load_ttm(cur, tickers)
    recorded_splits = _load_recorded_splits(cur, start_date)

    results: List[AuditResult] = []

    def _provider_splits(tkr: str) -> List[Tuple[date, float]]:
        try:
            return [(dt, ratio) for dt, ratio, _ in fetch_split_history(tkr) if dt and ratio and dt >= start_date]
        except Exception as exc:
            logging.warning("Failed to fetch splits for %s: %s", tkr, exc)
            return []

    for ticker in tickers:
        provider_events = _provider_splits(ticker)
        if not provider_events:
            continue
        for split_date, ratio in provider_events:
            recorded = split_date in recorded_splits.get(ticker, {})
            result = _analyze_event(
                ticker,
                split_date,
                ratio,
                recorded,
                annual_eps.get(ticker, []),
                ttm_map.get(ticker),
                args.tolerance,
            )
            results.append(result)

    if not results:
        logging.info("No split events found on or after %s.", start_date.isoformat())
        return

    csv_path = Path(args.output_csv)
    html_path = Path(args.output_html)
    _write_csv(csv_path, results)
    _write_html(html_path, results)

    outstanding = sum(1 for r in results if r.recommendation != "skip")
    logging.info("Wrote CSV summary to %s", csv_path.resolve())
    logging.info("Wrote HTML summary to %s", html_path.resolve())
    logging.info("Outstanding actions: %d of %d events require attention.", outstanding, len(results))


if __name__ == "__main__":
    main()
