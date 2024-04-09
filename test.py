import yfinance as yf

# Replace 'AAPL' with the ticker of the company you're interested in
ticker = 'brk-b'

# Use yfinance to download the income statement
company = yf.Ticker(ticker)
quarterly_financials = company.info

print(quarterly_financials)
# Convert the income statement to a string and save it to a text file
with open('income_statement.txt', 'w') as f:
    # The income statement DataFrame is converted to a string for writing to the file
    f.write(quarterly_financials.to_string())
