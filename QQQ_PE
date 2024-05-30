import yfinance as yf
import pandas as pd

def get_qqq_holdings():
    qqq = yf.Ticker("QQQ")
    holdings = qqq.fund_holdings['holdings']
    
    # Extract stock symbols and holding percentages
    stocks = [holding['symbol'] for holding in holdings]
    percentages = [holding['holdingPercent'] for holding in holdings]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Stock Symbol': stocks,
        'Holding Percentage': percentages
    })
    
    return df

def calculate_weighted_eps_and_pe(df):
    # Initialize cumulative EPS
    cumulative_eps = 0
    
    for index, row in df.iterrows():
        ticker = row['Stock Symbol']
        holding_percentage = row['Holding Percentage']
        
        try:
            stock = yf.Ticker(ticker)
            eps = stock.info.get('trailingEps', 0)  # Get TTM EPS
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            eps = 0
        
        weighted_eps = eps * holding_percentage
        cumulative_eps += weighted_eps
    
    # Get the closing price of QQQ
    qqq = yf.Ticker("QQQ")
    qqq_price = qqq.history(period="1d")['Close'].iloc[0]
    
    # Calculate QQQ P/E ratio
    qqq_pe_ratio = qqq_price / cumulative_eps if cumulative_eps != 0 else float('inf')
    
    print(f"Cumulative Weighted EPS: {cumulative_eps}")
    print(f"QQQ Closing Price: {qqq_price}")
    print(f"QQQ P/E Ratio: {qqq_pe_ratio}")
    
    return qqq_pe_ratio

def save_as_html(qqq_pe_ratio):
    # Create a simple HTML table
    html_content = f"""
    <html>
    <head><title>QQQ P/E Ratio</title></head>
    <body>
    <table border="1">
        <tr>
            <td>QQQ P/E:</td>
            <td>{qqq_pe_ratio}</td>
        </tr>
    </table>
    </body>
    </html>
    """
    
    # Save the HTML content to a file
    with open('qqq_pe_ratio.html', 'w') as file:
        file.write(html_content)
    
    print("QQQ P/E ratio saved to qqq_pe_ratio.html")

def main():
    df = get_qqq_holdings()
    df.to_csv('qqq_holdings.csv', index=False)
    print("QQQ holdings saved to qqq_holdings.csv")
    
    qqq_pe_ratio = calculate_weighted_eps_and_pe(df)
    
    save_as_html(qqq_pe_ratio)

# Run the main function
main()
