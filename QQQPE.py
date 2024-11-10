import requests
import pandas as pd
import yfinance as yf
from io import BytesIO
from datetime import datetime

# URL to download QQQ holdings Excel data
holdings_url = 'https://www.invesco.com/QQQ-etf/en/holdings/download?audienceType=Institutional'

# Headers to mimic a browser request (if necessary)
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'Referer': 'https://www.invesco.com/QQQ-etf/en/holdings'
}

# Fetch the holdings Excel data
response = requests.get(holdings_url, headers=headers)

if response.status_code == 200:
    # The content is in Excel format
    excel_content = response.content

    # Read the Excel content using pandas
    holdings = pd.read_excel(BytesIO(excel_content), sheet_name=0)

    # Ensure the necessary columns are present
    if {'Security Name', 'Ticker', 'Weight (%)'}.issubset(holdings.columns):
        holdings.rename(columns={'Ticker': 'Symbol', 'Weight (%)': 'Weight'}, inplace=True)

        # Initialize variables
        weighted_pe_sum = 0
        total_weight = 0

        # Prepare to collect data for the HTML table
        result_data = []

        # Loop through each holding
        for index, row in holdings.iterrows():
            ticker = row['Symbol']
            weight = row['Weight']  # Weight in percentage

            # Get stock info
            try:
                stock = yf.Ticker(ticker)
                pe_ratio = stock.info.get('trailingPE', None)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                pe_ratio = None

            if pe_ratio is not None and pe_ratio > 0:
                weighted_pe = pe_ratio * weight
                weighted_pe_sum += weighted_pe
                total_weight += weight

                # Append data for HTML table
                result_data.append({
                    'Ticker': ticker,
                    'Weight (%)': weight,
                    'P/E Ratio': round(pe_ratio, 2),
                    'Weighted P/E': round(weighted_pe, 2)
                })
            else:
                print(f"P/E ratio not available for {ticker}")

        # Compute weighted average P/E ratio
        if total_weight > 0:
            weighted_avg_pe = weighted_pe_sum / total_weight
            print(f"QQQ P/E Ratio: {weighted_avg_pe:.2f}")

            # Add total row to the data
            result_data.append({
                'Ticker': 'Total',
                'Weight (%)': round(total_weight, 2),
                'P/E Ratio': '',
                'Weighted P/E': round(weighted_pe_sum, 2)
            })

            # Create a DataFrame for the HTML table
            output_df = pd.DataFrame(result_data)

            # Generate HTML table
            html_table = output_df.to_html(index=False)

            # Include the date and P/E ratio in the HTML output
            current_date = datetime.now().strftime('%Y-%m-%d')
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QQQ P/E Ratio</title>
                <style>
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        text-align: center;
                    }}
                    td {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>QQQ P/E Ratio as of {current_date}</h1>
                <p>Weighted Average P/E Ratio: <strong>{weighted_avg_pe:.2f}</strong></p>
                {html_table}
            </body>
            </html>
            """

            # Save the HTML content to a file
            with open('qqq_pe_ratio.html', 'w') as f:
                f.write(html_content)
        else:
            print("Could not compute P/E ratio. Total weight is zero.")
    else:
        print("Required columns are missing in the holdings data.")
else:
    print(f"Failed to download holdings data. Status code: {response.status_code}")
