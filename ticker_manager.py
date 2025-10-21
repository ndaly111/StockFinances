import csv
import logging
from typing import Iterable, List, Sequence, Set

logger = logging.getLogger(__name__)


def read_tickers(file_path: str) -> List[str]:
    """Return the distinct tickers stored in ``file_path`` sorted alphabetically."""
    logger.debug("Reading tickers from %s", file_path)
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        tickers: Set[str] = set()
        for row in reader:
            if not row:
                continue
            ticker = row[0].strip()
            if ticker:
                tickers.add(ticker)
    ordered = sorted(tickers)
    logger.debug("Loaded %d tickers", len(ordered))
    return ordered


def modify_tickers(ticker_data: Iterable[str], is_remote: bool = False) -> List[str]:
    """Optionally interact with the user to update the ticker list.

    When ``is_remote`` is ``True`` we bypass interactive prompts and simply
    return the distinct tickers in sorted order. Locally we allow users to add
    or remove tickers before returning the sorted result.
    """

    tickers: Set[str] = {ticker.strip().upper() for ticker in ticker_data if ticker}
    if is_remote:
        logger.debug("Remote mode detected; skipping interactive ticker edits.")
        return sorted(tickers)

    while True:
        sorted_tickers = sorted(tickers)
        print("Current tickers:", ", ".join(sorted_tickers))
        action = input(
            "Do you want to add, remove, sort the tickers, or make no changes? (add/remove/sort/n): "
        ).lower()

        if action == "add":
            new_tickers = input("Enter tickers to add (comma-separated): ").upper().split(",")
            for ticker in new_tickers:
                ticker = ticker.strip()
                if ticker:
                    tickers.add(ticker)
        elif action == "remove":
            remove_tickers = input("Enter tickers to remove (comma-separated): ").upper().split(",")
            for ticker in remove_tickers:
                ticker = ticker.strip()
                if ticker:
                    tickers.discard(ticker)
        elif action == "sort":
            print("Tickers are automatically sorted after add/remove actions.")
        elif action == "n":
            break
        else:
            print("Invalid action. Please choose add, remove, sort, or n for no changes.")

    return sorted(tickers)


def write_tickers(ticker_data: Sequence[str], file_path: str) -> None:
    """Persist ``ticker_data`` to ``file_path``.

    The caller is expected to supply data in the desired order so we do not
    perform an additional sort here.
    """

    ordered = list(ticker_data)
    logger.debug("Writing %d tickers to %s", len(ordered), file_path)
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ticker"])
        for ticker in ordered:
            writer.writerow([ticker])

