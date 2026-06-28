"""Bottom-up index forward EPS from constituent forward EPS x shares."""
from __future__ import annotations
import logging, sqlite3
from typing import Dict, List
logger = logging.getLogger(__name__)


def load_constituent_financials(conn, tickers) -> Dict[str, dict]:
    out = {}
    for tk in tickers:
        row = conn.execute("SELECT TTM_EPS, Shares_Outstanding FROM TTM_Data WHERE Symbol=?",
                           (tk,)).fetchone()
        if not row or row[0] is None or row[1] is None:
            continue
        fy = {}
        for label in ("This FY", "Next FY"):
            r = conn.execute("""SELECT forward_eps FROM Forward_EPS_FY_History
                                WHERE ticker=? AND period_label=? AND forward_eps IS NOT NULL
                                ORDER BY date_recorded DESC LIMIT 1""", (tk, label)).fetchone()
            fy[label] = r[0] if r else None
        if fy["This FY"] is None:
            continue
        out[tk.upper()] = {"ttm_eps": float(row[0]), "shares": float(row[1]),
                           "this_fy": float(fy["This FY"]),
                           "next_fy": float(fy["Next FY"]) if fy["Next FY"] is not None else None}
    return out


def aggregate(holdings, fin) -> dict:
    covered = {tk.upper() for tk, _ in holdings if tk.upper() in fin}
    total_w = sum(w for _, w in holdings) or 1.0
    cov_w = sum(w for tk, w in holdings if tk.upper() in covered) / total_w
    ttm = thisfy = nextfy = 0.0
    nextfy_ok = True
    for tk in covered:
        f = fin[tk]
        ttm += f["ttm_eps"] * f["shares"]
        thisfy += f["this_fy"] * f["shares"]
        if f["next_fy"] is None:
            nextfy_ok = False
        else:
            nextfy += f["next_fy"] * f["shares"]
    if ttm <= 0 or not covered:
        return {"growth_this_fy": None, "growth_next_fy": None,
                "coverage_weight": cov_w, "fwd_earnings_this_fy": thisfy,
                "trailing_earnings": ttm}
    return {"growth_this_fy": thisfy / ttm - 1.0,
            "growth_next_fy": (nextfy / thisfy - 1.0) if (nextfy_ok and thisfy > 0) else None,
            "coverage_weight": cov_w, "fwd_earnings_this_fy": thisfy,
            "trailing_earnings": ttm}
