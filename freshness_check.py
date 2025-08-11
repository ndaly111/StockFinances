#!/usr/bin/env python3
import os, sys, re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

STAMP_FILE = Path("charts/_build_stamp.txt")

# Exclusions: files we don't care about or that legitimately don't change every run
EXCLUDE_PATTERNS = [
    r"/\.DS_Store$",
    r"/index\.html$",
    r"/dashboard\.html$",
    r"/earnings_(past|upcoming)\.html$",
]
EXCLUDE_RE = [re.compile(p) for p in EXCLUDE_PATTERNS]

# Which filetypes to check under charts/
CHECK_EXTS = {".html", ".png", ".svg", ".webp"}  # add/remove as needed

def is_excluded(p: Path) -> bool:
    s = str(p).replace("\\", "/")
    return any(rx.search(s) for rx in EXCLUDE_RE)

def read_stamp() -> datetime:
    if not STAMP_FILE.is_file():
        print(f"[freshness] ERROR: build stamp not found: {STAMP_FILE}", file=sys.stderr)
        sys.exit(2)
    ts = STAMP_FILE.read_text(encoding="utf-8").strip()
    try:
        # allow ...Z or +00:00
        ts = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(ts)
    except Exception:
        print(f"[freshness] ERROR: unreadable stamp: {ts}", file=sys.stderr)
        sys.exit(2)

def scan_charts(stamp: datetime) -> List[Tuple[str, float]]:
    """Return list of (path, age_seconds_before_stamp) for stale artifacts."""
    base = Path("charts")
    if not base.is_dir():
        print("[freshness] charts/ not found; nothing to check.")
        return []
    stale = []
    for p in base.rglob("*"):
        if not p.is_file():           continue
        if is_excluded(p):            continue
        if p.suffix.lower() not in CHECK_EXTS: continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if mtime < stamp:
            age = (stamp - mtime).total_seconds()
            stale.append((str(p), age))
    return stale

def write_report(stale: List[Tuple[str, float]], report_path="charts/freshness_report.html"):
    Path("charts").mkdir(parents=True, exist_ok=True)
    def fmt_age(sec: float) -> str:
        h = sec/3600.0
        return f"{h:.1f} h"
    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{path}</td><td>{fmt_age(age)}</td></tr>"
        for i,(path,age) in enumerate(sorted(stale, key=lambda x: -x[1]))
    )
    html = f"""<!doctype html><meta charset="utf-8">
<title>Freshness Report</title>
<h1>Freshness Report</h1>
<p>Artifacts older than this run’s build stamp are listed below.</p>
<table border="1" cellpadding="6" cellspacing="0">
<thead><tr><th>#</th><th>Path</th><th>Hours older than stamp</th></tr></thead>
<tbody>
{rows if rows else '<tr><td colspan="3">All good — nothing stale.</td></tr>'}
</tbody></table>
"""
    Path(report_path).write_text(html, encoding="utf-8")
    print(f"[freshness] wrote report → {report_path}")

if __name__ == "__main__":
    stamp = read_stamp()
    stale = scan_charts(stamp)
    write_report(stale)
    if stale:
        print(f"[freshness] FAIL: {len(stale)} stale artifact(s) found.")
        for p, age in stale[:10]:
            print(f" - {p}")
        sys.exit(1)
    print("[freshness] PASS: all checked artifacts are fresh.")
