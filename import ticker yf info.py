import yfinance as yf

stock = yf.Ticker("AAPL")  # Create a Ticker object

stock_data = stock.info  # Access the info attribute

# Print all the data fields and their values
for key, value in stock_data.items():
    print(f"{key}: {value}")
