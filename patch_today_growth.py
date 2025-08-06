- name: Recalculate today’s implied growth
  env:
    MANUAL_YIELD: ${{ inputs.yield_decimal }}   # optional textbox in UI
  run: |
    python <<'PY'
    import os, sqlite3
    from datetime import datetime

    DB   = "Stock Data.db"
    IDX  = ["SPY", "QQQ"]
    TOD  = datetime.utcnow().strftime("%Y-%m-%d")   # runner’s “today”

    def to_dec(v):  # 4.2 → 0.042
        return v/100 if 0.5 <= v < 20 else v/1000 if v >= 20 else v

    conn = sqlite3.connect(DB)
    cur  = conn.cursor()

    # ── 1. get latest yield row (could be 5th even if runner date = 6th)
    yrow = cur.execute(
        "SELECT Date,TenYr FROM Treasury_Yield_History "
        "ORDER BY Date DESC LIMIT 1").fetchone()

    if not yrow:
        raise SystemExit("⚠️  Treasury_Yield_History empty – run main job first.")

    y_date, y_val = yrow
    y = to_dec(float(y_val))

    # if manual yield supplied and fresher, override
    man = os.getenv("MANUAL_YIELD")
    if man:
        y = float(man)
        y_date = f"{TOD} (manual)"

    print(f"Using yield {y:.4f} from row date {y_date}")

    # helper: latest PE
    def latest_pe(tk):
        r=cur.execute("SELECT PE_Ratio FROM Index_PE_History "
                      "WHERE Ticker=? AND PE_Type='TTM' "
                      "ORDER BY Date DESC LIMIT 1",(tk,)).fetchone()
        return r[0] if r else None

    for tk in IDX:
        pe = latest_pe(tk)
        if pe is None:
            print(f"[{tk}] skipped – no PE"); continue
        g = (pe/10)**0.1 + y - 1
        print(f"[{tk}] PE={pe:.2f} → g={g:.4%}")
        cur.execute("INSERT OR REPLACE INTO Index_Growth_History "
                    "VALUES (?,?, 'TTM', ?)",
                    (TOD, tk, g))
    conn.commit()
    print("✓ Implied-growth rows patched.")
    PY
