import os

ALLOW_YAHOO_STORAGE = os.getenv("ALLOW_YAHOO_STORAGE", "false").lower() == "true"


def get_fmp_api_key() -> str:
    key = os.getenv("FMP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY environment variable must be set to use licensed Financial Modeling Prep data.")
    return key
