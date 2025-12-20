"""Utilities to download play-by-play data and compute team EPA splits."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import nfl_data_py as nfl


DEFAULT_YEAR = 2025


def download_pbp(year: int, *, cache: bool = True) -> pd.DataFrame:
    """Download play-by-play data for a single season.

    Args:
        year: Season to download.
        cache: Whether to allow nfl_data_py to cache data locally.

    Returns:
        The full play-by-play dataframe, including EPA metrics.
    """

    pbp = nfl.import_pbp_data(years=[year], cache=cache)
    return pbp


def _aggregate_epa_by_team(pbp: pd.DataFrame, team_column: str, *, sign_flip: bool) -> pd.DataFrame:
    """Aggregate EPA totals and per-play averages for a team column.

    Args:
        pbp: The play-by-play dataframe containing an ``epa`` column.
        team_column: Column name representing the grouping (``posteam`` or ``defteam``).
        sign_flip: Flip the EPA sign before computing per-play values. Defensive EPA is
            flipped so that lower (better) defensive EPA appears as a higher positive number.

    Returns:
        DataFrame with total EPA, play counts, and per-play EPA for each team.
    """

    totals = (
        pbp.groupby(team_column)["epa"]
        .agg(["sum", "count"])
        .rename(columns={"sum": f"EPA_{team_column}_total", "count": f"Plays_{team_column}"})
    )

    per_play_column = f"EPA_{team_column}_per_play"
    totals[per_play_column] = (totals[f"EPA_{team_column}_total"] * (-1 if sign_flip else 1)) / totals[
        f"Plays_{team_column}"
    ]

    return totals


def compute_team_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team offensive and defensive EPA summaries.

    Offensive EPA uses ``posteam`` while defensive EPA flips the EPA sign to represent
    value prevented rather than allowed.
    """

    filtered = pbp[pbp["epa"].notna()]
    offense = _aggregate_epa_by_team(filtered, "posteam", sign_flip=False)
    defense = _aggregate_epa_by_team(filtered, "defteam", sign_flip=True)

    combined = pd.concat([offense, defense], axis=1).fillna(0)
    return combined


def build_epa_summary(year: int = DEFAULT_YEAR) -> pd.DataFrame:
    """Download PBP data for ``year`` and compute team EPA splits."""

    pbp = download_pbp(year)
    summary = compute_team_epa(pbp)
    return summary


def persist_summary(summary: pd.DataFrame, year: int, *, output_dir: Path | str | None = None) -> Path:
    """Persist the EPA summary to CSV and return the path."""

    destination_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_path = destination_dir / f"nfl_{year}_team_epa.csv"
    summary.to_csv(output_path)
    return output_path


__all__: Iterable[str] = [
    "DEFAULT_YEAR",
    "build_epa_summary",
    "compute_team_epa",
    "download_pbp",
    "persist_summary",
]
