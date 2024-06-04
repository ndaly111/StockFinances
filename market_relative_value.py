import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the SQLite database
import sqlite3
conn = sqlite3.connect('/Users/nicholasdaly/PycharmProjects/Stock Finances/Stock Data.db')
query = "SELECT date, avg_ttm_valuation, avg_forward_valuation FROM AverageValuations"
df = pd.read_sql_query(query, conn)

# Ensure the 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Set 'date' as the index
df.set_index('date', inplace=True)

# Create a directory to save the images if it doesn't exist
output_dir = '/Users/nicholasdaly/PycharmProjects/Stock Finances/output'
os.makedirs(output_dir, exist_ok=True)

# Improved line chart for all valuations
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['avg_ttm_valuation'], label='Avg TTM Valuation', marker='o', linestyle='-', linewidth=2)
plt.plot(df.index, df['avg_forward_valuation'], label='Avg Forward Valuation', marker='s', linestyle='--', linewidth=2)
plt.title('Valuation Metrics Over Time', fontsize=16)
plt.suptitle('Comparison of Average TTM and Forward Valuations', fontsize=12, y=0.95)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Valuation', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'valuation_metrics_over_time_improved.png'))
plt.close()

conn.close()
