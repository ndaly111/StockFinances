import sqlite3
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Convert data to DataFrame and clean up the growth rates
df = pd.DataFrame(data)
df['Expected Growth'] = df['Expected Growth'].str.rstrip('%').astype(float)  # Strip the '%' and convert to float

# Database connection
db_path = "Stock Data.db"  # Path to your SQLite database
logging.info(f"Connecting to database at {db_path}")
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
    logging.info(f"Updated {row['TICKER']} with growth rate {row['Expected Growth']}")

# Commit changes and close the connection
conn.commit()
conn.close()

logging.info("Growth rates have been successfully updated.")
print("Growth rates have been successfully updated.")
