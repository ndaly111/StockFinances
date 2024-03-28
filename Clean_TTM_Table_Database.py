import sqlite3
import datetime

db_path = 'Stock Data.db'

def clean_ttm_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update missing 'Last_Updated' fields with the current timestamp
    cursor.execute("""
        UPDATE TTM_Data
        SET Last_Updated = ?
        WHERE Last_Updated IS NULL
    """, (current_timestamp,))
    conn.commit()

    # Remove old TTM entries, keeping only the most recent quarter data for each symbol
    cursor.execute("""
        DELETE FROM TTM_Data
        WHERE rowid NOT IN (
            SELECT MAX(rowid)
            FROM TTM_Data
            GROUP BY Symbol
        )
    """)
    conn.commit()

    print("Old TTM entries have been removed and missing Last_Updated fields updated.")

    cursor.close()
    conn.close()




def check_and_update_annual_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Select rows in Annual Data with missing values
    cursor.execute("""
        SELECT Symbol, Date, Revenue, Net_Income, EPS, Last_Updated 
        FROM Annual_Data 
        WHERE Revenue IS NULL OR Net_Income IS NULL OR EPS IS NULL OR Last_Updated IS NULL
    """)
    rows_with_missing_data = cursor.fetchall()

    # Prompt for missing values and update the 'Last_Updated' field
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows_with_missing_data:
        symbol, date, revenue, net_income, eps, last_updated = row
        print(f"Checking missing fields for {symbol} in {date}")

        if revenue is None:
            revenue = input(f"Please enter the Revenue for {symbol} in {date}: ")
        if net_income is None:
            net_income = input(f"Please enter the Net Income for {symbol} in {date}: ")
        if eps is None:
            eps = input(f"Please enter the EPS for {symbol} in {date}: ")

        # Update the row with any new values and set 'Last_Updated' to the current time if it was NULL
        cursor.execute("""
            UPDATE Annual_Data 
            SET Revenue = COALESCE(?, Revenue), 
                Net_Income = COALESCE(?, Net_Income), 
                EPS = COALESCE(?, EPS), 
                Last_Updated = COALESCE(?, Last_Updated)
            WHERE Symbol = ? AND Date = ?
        """, (revenue, net_income, eps, current_time if last_updated is None else last_updated, symbol, year))
        conn.commit()
        print(f"Updated data for {symbol} in {date}.")

    cursor.close()
    conn.close()

# Run the functions
clean_ttm_data(db_path)
check_and_update_annual_data(db_path)


