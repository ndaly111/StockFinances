# StockFinances

Utilities for populating and maintaining the SQLite databases that back the finance dashboards.

## Licensed data providers

All new balance sheet and annual income statement data now comes from a licensed provider. The repository ships with an implementation for [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/), and additional providers can be added under `data_providers/` by extending `DataProvider`.

### Configuration

Set the following environment variables before running any scripts that fetch remote data:

- `FMP_API_KEY` – Required. API key for Financial Modeling Prep. Without this key, provider-backed fetches will be skipped.
- `ALLOW_YAHOO_STORAGE` – Optional. Defaults to `false`. When left unset/false, any attempt to fetch or store Yahoo-derived data (including TTM helpers) will raise an error to avoid persisting unlicensed data.

Example:

```bash
export FMP_API_KEY="your-fmp-api-key"
export ALLOW_YAHOO_STORAGE=false
```

## Data ingestion notes

- `data_fetcher.py` uses the licensed provider to pull annual revenue, net income, and EPS and aligns the columns to the `Annual_Data` table schema. Yahoo-based helpers are gated to prevent persistence by default.
- `balance_sheet_data_fetcher.py` now retrieves the latest balance sheet values from the same provider and maps them to the `BalanceSheetData` table schema.

## Licensing and permitted use

Financial Modeling Prep data is provided under their licensed terms. Ensure your usage complies with their developer agreement, and do not redistribute or cache data beyond what your license permits. Yahoo Finance data should not be persisted unless you have explicitly enabled it via `ALLOW_YAHOO_STORAGE` and confirmed that doing so aligns with your licensing obligations.
