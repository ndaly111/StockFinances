import sqlite3
import pandas as pd

# Data
data = {
    "TICKER": [
        "GOOGL", "AMZN", "META", "AAPL", "TSLA", "JPM", "CVS", "TSN", "BRK.B",
        "CCL", "RCL", "NCLH", "F", "MSFT", "NFLX", "GE", "CRM", "UBER", "DKNG",
        "GM", "DIS", "AMD", "NVDA", "ADBE", "COIN", "SNAP", "INTC", "DPZ", "HOOD",
        "EVGO", "BABA", "PLTR", "PYPL", "V", "MU", "VZ", "CMG", "MA", "Sbux", "MRNA",
        "PFE", "LRCX", "FSLR", "SCHW", "GTIM", "BIRD", "BA", "CELH", "COST"
    ],
    "Expected Growth": [
        "12.0%", "20.0%", "13.0%", "8.0%", "20.0%", "4.0%", "5.0%", "2.0%", "8.0%",
        "2.0%", "3.0%", "5.0%", "5.0%", "13.0%", "15.0%", "0.0%", "20.0%", "20.0%",
        "20.0%", "2.5%", "8.0%", "10.0%", "15.0%", "12.0%", "14.3%", "25.0%", "2.5%",
        "11.5%", "20.0%", "45.0%", "4.0%", "25.0%", "18.0%", "12.0%", "12.0%", "0.0%",
        "20.0%", "17.5%", "9.0%", "0.0%", "9.0%", "6%", "6%", "6%", "6%", "0%", "5%", "20%", "11%"
    ]
}

# Convert data to DataFrame and clean up the growth rates
df = pd.DataFrame(data)
df['Expected Growth'] = df['Expected Growth'].str.rstrip('%').astype(float)  # Strip the '%' and convert to float

# Database connection
db_path = "Stock Data.db"  # Path to your SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Assuming the table 'Tickers_Info' already exists and it has a column 'nicks_growth_rate'
# Check if the column needs to be created (uncomment the next lines if you need to create it)
# cursor.execute('''
# ALTER TABLE Tickers_Info ADD COLUMN nicks_growth_rate REAL;
# ''')

# Update the database
for index, row in df.iterrows():
    cursor.execute('''
        UPDATE Tickers_Info
        SET nicks_growth_rate = ?
        WHERE ticker = ?;
    ''', (row['Expected Growth'], row['TICKER']))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Growth rates have been successfully updated.")
