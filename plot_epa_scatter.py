"""Generate an EPA per play scatter plot for NFL teams.

The script downloads play-by-play data, aggregates offense and defense EPA
per play, and plots them with team logos. It is designed to run in CI to
produce the latest chart for the requested season.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - Pillow is optional
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

from sources import download_epa_csv, download_team_logo


@dataclass
class EPAPlotConfig:
    season: int
    output_path: Path = Path("plots/epa_scatter.png")
    data_dir: Path = Path("data")
    logos_dir: Path = Path("assets/logos/png")
    zoom: float = 0.2


def load_epa_data(season: int, data_dir: Path) -> pd.DataFrame:
    csv_path = download_epa_csv(season, data_dir)
    return pd.read_csv(csv_path)


def team_colors(team: str) -> tuple[int, int, int]:
    # Deterministic but visually varied color seed based on team abbreviation.
    seed = sum(ord(c) for c in team)
    return (64 + seed * 3 % 192, 64 + seed * 5 % 192, 64 + seed * 7 % 192)


def placeholder_logo(team: str, size: int = 256) -> bytes:
    """Create a simple placeholder logo as PNG bytes."""
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow not available for placeholder logos")

    img = Image.new("RGBA", (size, size), team_colors(team) + (255,))
    draw = ImageDraw.Draw(img)
    text = team.upper()
    try:
        font = ImageFont.truetype("arial.ttf", size // 3)
    except Exception:
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255, 255), font=font)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def ensure_team_logos(teams: Iterable[str], logos_dir: Path) -> None:
    for team in teams:
        path = logos_dir / f"{team.lower()}.png"
        if path.exists():
            continue
        download_team_logo(team, "png", logos_dir.parent)


def aggregate_epa(df: pd.DataFrame) -> pd.DataFrame:
    offense = (
        df.dropna(subset=["posteam", "epa"])
        .groupby("posteam")
        ["epa"]
        .mean()
        .rename("off_epa_per_play")
    )
    defense = (
        df.dropna(subset=["defteam", "epa"])
        .groupby("defteam")
        ["epa"]
        .mean()
        .mul(-1)
        .rename("def_epa_per_play")
    )
    combined = pd.concat([offense, defense], axis=1).reset_index().rename(columns={"posteam": "team"})
    return combined.dropna(subset=["off_epa_per_play", "def_epa_per_play"])


def logo_image(path: Path, zoom: float = 0.2) -> Optional[OffsetImage]:
    try:
        if not path.exists():
            return None
        if Image is not None:
            with path.open("rb") as fh:
                img = Image.open(fh).convert("RGBA")
            return OffsetImage(img, zoom=zoom)
        return OffsetImage(plt.imread(path), zoom=zoom)
    except Exception:
        return None


def draw_logos(ax: plt.Axes, df: pd.DataFrame, logos_dir: Path, zoom: float = 0.2) -> None:
    """Place team logos (or fallback text) at the provided coordinates."""
    for row in df.itertuples(index=False):
        team = str(row.team).upper()
        x = row.off_epa_per_play
        y = row.def_epa_per_play
        if pd.isna(x) or pd.isna(y):
            continue

        image = logo_image(logos_dir / f"{team}.png", zoom=zoom)
        if image:
            ab = AnnotationBbox(image, (x, y), frameon=False)
            ax.add_artist(ab)
        else:
            # Attempt to create a colored placeholder square with the team abbreviation.
            if placeholder_logo and Image is not None:
                try:
                    # Generate placeholder PNG bytes with same size as logos (256).
                    placeholder_bytes = placeholder_logo(team, 256)
                    # Load the bytes into a Pillow image using RGBA.
                    img = Image.open(BytesIO(placeholder_bytes)).convert("RGBA")  # type: ignore[arg-type]
                    # Create an offsetImage for the placeholder and add it to the plot.
                    placeholder_img = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(placeholder_img, (x, y), frameon=False)
                    ax.add_artist(ab)
                    continue
                except Exception:
                    # Fall back to simple scatter/text if placeholder generation fails.
                    pass
            # Default fallback: use a black point with text annotation.
            ax.scatter(x, y, color="black", s=20, zorder=5)
            ax.text(x, y, team, fontsize=8, ha="center", va="center")
            continue


def add_reference_lines(ax: plt.Axes) -> None:
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.text(0.02, 0.98, "Better defense", transform=ax.transAxes, va="top")
    ax.text(0.98, 0.02, "Better offense", transform=ax.transAxes, ha="right")


def plot_epa_scatter(config: EPAPlotConfig) -> Path:
    df = load_epa_data(config.season, config.data_dir)
    team_epa = aggregate_epa(df)
    ensure_team_logos(team_epa["team"], config.logos_dir)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(team_epa["off_epa_per_play"], team_epa["def_epa_per_play"], alpha=0)
    add_reference_lines(ax)
    draw_logos(ax, team_epa, config.logos_dir, zoom=config.zoom)

    ax.set_title(f"EPA per Play (Season {config.season})")
    ax.set_xlabel("Offense EPA per play")
    ax.set_ylabel("Defense EPA allowed per play (lower is better)")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.output_path, dpi=300)
    plt.close(fig)
    return config.output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate EPA scatter plot")
    parser.add_argument("season", type=int, help="Season year, e.g. 2024")
    parser.add_argument("--output", type=Path, default=EPAPlotConfig.output_path)
    args = parser.parse_args()

    config = EPAPlotConfig(season=args.season, output_path=args.output)
    output = plot_epa_scatter(config)
    print(f"Saved EPA scatter plot to {output}")


if __name__ == "__main__":
    main()
