import csv

def read_tickers(file_path):
    print("ticker manager 1 reading tickers")
    """Reads tickers from a CSV file."""
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        ticker_data = set()
        next(reader, None)  # Skip header
        for row in reader:
            if row:  # Checking if row is not empty
                ticker = row[0].strip()
                ticker_data.add(ticker)
        return ticker_data



def modify_tickers(ticker_data, is_remote=False):
    print("ticker manager 2 modifying tickers")
    # If running remotely, skip user prompts and return the sorted tickers.
    if is_remote:
        print("Running in remote mode. Skipping ticker modification.")
        return sorted(ticker_data)

    while True:
        # Convert set to list and sort for display
        sorted_tickers = sorted(list(ticker_data))
        print("Current tickers:", ', '.join(sorted_tickers))
        action = input(
            "Do you want to add, remove, sort the tickers, or make no changes? (add/remove/sort/n): ").lower()

        if action == 'add':
            new_tickers = input("Enter tickers to add (comma-separated): ").upper().split(',')
            for ticker in new_tickers:
                ticker = ticker.strip()
                if ticker not in ticker_data:
                    ticker_data.add(ticker)  # Add new ticker


        elif action == 'remove':
            remove_tickers = input("Enter tickers to remove (comma-separated): ").upper().split(',')
            for ticker in remove_tickers:
                ticker = ticker.strip()
                if ticker in ticker_data:
                    ticker_data.remove(ticker)  # Remove ticker

        elif action == 'sort':
            print("Tickers are automatically sorted after add/remove actions.")

        elif action == 'n':
            break

        else:
            print("Invalid action. Please choose add, remove, sort, or n for no changes.")

    # Return a sorted list instead of a set
    return sorted(list(ticker_data))


def write_tickers(ticker_data, file_path):
    print("ticker manage 3 writing tickers")
    """Writes the updated list of tickers to the CSV file, sorted alphabetically."""
    sorted_tickers = sorted(ticker_data)  # Sort the tickers alphabetically
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ticker'])  # Write the header
        for ticker in sorted_tickers:  # Iterate through sorted list
            writer.writerow([ticker])

