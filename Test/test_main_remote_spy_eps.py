import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_remote


def test_ensure_spy_monthly_eps_and_derived_pe_loads_data(tmp_path: Path):
    db_path = tmp_path / "test.db"
    eps_csv = tmp_path / "spy_eps.csv"
    price_csv = tmp_path / "spy_price.csv"

    eps_csv.write_text(
        "\n".join(
            [
                "Date,SPY_EPS",
                "2024-01-31,10",
                "2024-02-29,12",
            ]
        ),
        encoding="utf-8",
    )
    price_csv.write_text(
        "\n".join(
            [
                "Date,Close",
                "2024-01-31,100",
                "2024-02-29,120",
            ]
        ),
        encoding="utf-8",
    )

    main_remote.ensure_spy_monthly_eps_and_derived_pe(
        str(db_path),
        eps_csv_path=eps_csv,
        price_csv_path=price_csv,
    )

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM Index_EPS_History WHERE Ticker=? AND EPS_Type=?",
            ("SPY", "TTM_REPORTED"),
        )
        assert cur.fetchone()[0] > 0

        cur.execute(
            "SELECT COUNT(*) FROM Index_Price_History_Monthly WHERE Ticker=?",
            ("SPY",),
        )
        assert cur.fetchone()[0] > 0

        cur.execute(
            "SELECT COUNT(*) FROM Index_PE_History WHERE Ticker=? AND PE_Type=?",
            ("SPY", "TTM_DERIVED_MONTHLY"),
        )
        assert cur.fetchone()[0] > 0
