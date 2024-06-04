import requests
import json
import pandas as pd
from bs4 import BeautifulSoup

# Load the API key from the JSON file
with open('config.json') as f:
    config = json.load(f)

api_key = config['api_key']

# Initialize the base URL for the sec-api
base_url = "https://api.sec-api.io"

# List of tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Replace with your tickers


# Function to get financial data using sec-api
def get_financial_data(ticker):
    query = {
        "query": {
            "query_string": {
                "query": f"ticker:{ticker} AND formType:(10-K OR 10-Q)"
            }
        },
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    response = requests.post(f"{base_url}?token={api_key}", json=query)
    if response.status_code != 200:
        print(f"Failed to fetch data for ticker {ticker}, status code: {response.status_code}")
        return None

    data = response.json()
    financials = []
    for filing in data['filings']:
        filing_date = filing['filedAt']
        form_type = filing['formType']

        if form_type in ['10-K', '10-Q']:
            filing_url = filing['linkToFilingDetails']
            response = requests.get(filing_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find the section with "CONSOLIDATED STATEMENTS OF OPERATIONS"
                consolidated_section = soup.find(string="CONSOLIDATED STATEMENTS OF OPERATIONS")
                if consolidated_section:
                    table = consolidated_section.find_next('table')
                    if table:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all(['td', 'th'])
                            cols = [ele.get_text(strip=True) for ele in cols]
                            if len(cols) > 1:
                                # Adjust the logic based on the actual structure of the table
                                if 'Revenue' in cols[0]:
                                    revenue = cols[1]
                                elif 'Net Income' in cols[0]:
                                    net_income = cols[1]
                                elif 'Earnings Per Share' in cols[0]:
                                    eps = cols[1]
                        financials.append({
                            'date': filing_date,
                            'form': form_type,
                            'revenue': revenue if 'revenue' in locals() else None,
                            'net_income': net_income if 'net_income' in locals() else None,
                            'eps': eps if 'eps' in locals() else None
                        })
            else:
                print(
                    f"Failed to fetch document for ticker {ticker}, document URL: {filing_url}, status code: {response.status_code}")
    return financials


# Collect financial data for each ticker
all_financials = {}
for ticker in tickers:
    print(f"Fetching financial data for ticker: {ticker}")
    financial_data = get_financial_data(ticker)
    if financial_data:
        all_financials[ticker] = financial_data
    else:
        print(f"No financial data found for ticker: {ticker}")

# Convert the financial data to a pandas DataFrame
financials_df = pd.DataFrame([
    {'ticker': ticker, **entry}
    for ticker, financials in all_financials.items()
    for entry in financials
])

# Save the DataFrame to a text file
output_file = 'financial_data.txt'
with open(output_file, 'w') as f:
    f.write(financials_df.to_string(index=False))

print(f"Financial data has been saved to '{output_file}'")
