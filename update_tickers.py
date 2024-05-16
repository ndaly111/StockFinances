import sys
import pandas as pd
import sqlite3

def update_csv(ticker, action, growth_rate=None, profit_margin=None):
    df = pd.read_csv('tickers.csv')

    if action == 'add':
        if ticker not in df['ticker'].values:
            new_row = {'ticker': ticker, 'growth_rate': growth_rate, 'profit_margin': profit_margin}
            df = df.append(new_row, ignore_index=True)
    elif action == 'remove':
        df = df[df['ticker'] != ticker]
    elif action == 'update':
        df.loc[df['ticker'] == ticker, 'growth_rate'] = growth_rate
        df.loc[df['ticker'] == ticker, 'profit_margin'] = profit_margin

    df.to_csv('tickers.csv', index=False)

def update_database(ticker, action, growth_rate=None, profit_margin=None):
    conn = sqlite3.connect('Stock Data.db')
    cursor = conn.cursor()

    if action == 'add':
        cursor.execute('INSERT INTO Tickers_Info (ticker, nicks_growth_rate, projected_profit_margin) VALUES (?, ?, ?)',
                       (ticker, growth_rate, profit_margin))
    elif action == 'remove':
        cursor.execute('DELETE FROM Tickers_Info WHERE ticker = ?', (ticker,))
    elif action == 'update':
        cursor.execute('UPDATE Tickers_Info SET nicks_growth_rate = ?, projected_profit_margin = ? WHERE ticker = ?',
                       (growth_rate, profit_margin, ticker))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    action = sys.argv[1]
    ticker = sys.argv[2]
    growth_rate = sys.argv[3] if len(sys.argv) > 3 else None
    profit_margin = sys.argv[4] if len(sys.argv) > 4 else None

    update_csv(ticker, action, growth_rate, profit_margin)
    update_database(ticker, action, growth_rate, profit_margin)
