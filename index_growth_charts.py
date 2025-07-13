# ── Add this helper just above _row ─────────────────────────
def _pct_fmt(x: float) -> str:
    """Return e.g. 0.1923 → '19.23 %'  (two decimals, thin space before %)"""
    return f"{x * 100:.2f} %"          # U+00A0 NBSP keeps % with the number

def _row(label: str, s: pd.Series, as_pct: bool = False) -> dict:
    """
    Build one summary-row dict.
    • If as_pct=True  → numeric columns are pre-formatted as XX.XX %
    • Else            → numbers are kept as floats for later plain formatting
    """
    if s.empty:
        empty = {"Metric": label, "Latest": "N/A", "Avg": "N/A", "Med": "N/A",
                 "Min": "N/A", "Max": "N/A", "%ctile": "—"}
        return empty

    # core stats
    stats = {
        "Metric": label,
        "Latest": s.iloc[-1],
        "Avg":    s.mean(),
        "Med":    s.median(),
        "Min":    s.min(),
        "Max":    s.max(),
        "%ctile": round(s.rank(pct=True).iloc[-1] * 100, 2)
    }

    if as_pct:                       # convert numeric stats to percentage strings
        for k in ("Latest", "Avg", "Med", "Min", "Max"):
            stats[k] = _pct_fmt(stats[k])

    return stats
