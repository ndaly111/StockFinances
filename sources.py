"""Utilities for accessing EPA and logo assets from documented sources.

This module keeps code and documentation aligned by encapsulating the URLs
and download helpers described in the README. Only the standard library is
used to avoid extra runtime dependencies.
"""
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import shutil

EPA_BASE_URL = "https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data"
LOGO_BASE_URL = "https://raw.githubusercontent.com/ryanmcdermott/nfl-logos/master"

LogoFormat = Literal["png", "svg"]

def epa_csv_url(season: int) -> str:
    """Build the nflfastR play-by-play CSV URL for a season.

    Args:
        season: Season year (e.g., 2023).

    Returns:
        Fully qualified HTTPS URL for the gzipped CSV file.
    """
    return f"{EPA_BASE_URL}/play_by_play_{season}.csv.gz"

def logo_url(team_abbr: str, fmt: LogoFormat = "png") -> str:
    """Build the URL for a team logo asset.

    Args:
        team_abbr: Team abbreviation (e.g., "KC", "PHI").
        fmt: File format extension (png or svg).

    Returns:
        HTTPS URL for the requested logo asset.
    """
    normalized_team = team_abbr.lower()
    return f"{LOGO_BASE_URL}/{fmt}/{normalized_team}.{fmt}"

def download_file(url: str, destination: Path) -> Path:
    """Download a remote asset to a destination path.

    Parent directories are created automatically. The function streams the
    response directly to disk to keep memory usage predictable for large EPA
    CSV files.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"Remote file not found at {url}") from exc
        raise
    except URLError as exc:
        raise ConnectionError(f"Failed to reach {url}: {exc.reason}") from exc

    return destination

def download_epa_csv(season: int, target_dir: Path | None = None) -> Path:
    """Download a season's nflfastR play-by-play CSV.

    Args:
        season: Season year to retrieve.
        target_dir: Optional directory for the downloaded file. Defaults to
            a local "data" directory under the repository root.

    Returns:
        Path to the downloaded gzipped CSV file.
    """
    directory = target_dir or Path("data")
    url = epa_csv_url(season)
    destination = directory / f"play_by_play_{season}.csv.gz"

    try:
        return download_file(url, destination)
    except FileNotFoundError as exc:
        try:
            return download_epa_csv_in_progress(season, target_dir)
        except Exception:
            raise FileNotFoundError(f"Remote file not found at {url}") from exc

def download_team_logo(
    team_abbr: str,
    fmt: LogoFormat = "png",
    target_dir: Path | None = None,
) -> Path:
    """Download a team logo asset.

    Args:
        team_abbr: Team abbreviation matching logo filenames (e.g., "KC").
        fmt: Desired logo format (png or svg).
        target_dir: Optional directory for downloaded logos. Defaults to the
            "assets/logos" directory under the repository root.

    Returns:
        Path to the downloaded logo file.
    """
    directory = target_dir or Path("assets") / "logos" / fmt
    url = logo_url(team_abbr, fmt)
    destination = directory / f"{team_abbr.lower()}.{fmt}"
    return download_file(url, destination)

def download_epa_csv_in_progress(
    season: int, target_dir: Path | None = None
) -> Path:
    """Scrape play-by-play data for a season that is still in progress.

    When the nflfastR project has not yet published a full-season play-by-play
    file (for example, early in a new season), the standard download_epa_csv
    helper above will raise a FileNotFoundError.  This helper uses the
    ``nfl_data_py`` package to import play-by-play data for the specified
    season so far, writes it to the same location and format expected by the
    rest of the pipeline, and returns the path to the generated file.

    Args:
        season: The season year to scrape (e.g., 2025).
        target_dir: Optional directory for the downloaded file.  Defaults to
            a local "data" directory under the repository root.

    Returns:
        Path to the gzipped CSV file containing all available plays for the
        requested season.

    Raises:
        ImportError: If the ``nfl_data_py`` package cannot be imported.
    """
    try:
        from nfl_data_py import import_data
    except Exception as exc:
        raise ImportError(
            "nfl_data_py is required to scrape in-progress data. "
            "Add it to your environment or requirements.txt"
        ) from exc

    directory = target_dir or Path("data")
    destination = directory / f"play_by_play_{season}.csv.gz"

    print(f"Scraping in-progress play-by-play data for {season}...")

    df = import_data.import_pbp_data([season])

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, compression="gzip")

    return destination
