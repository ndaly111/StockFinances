"""Entry point for downloading and displaying NFL EPA splits."""
from __future__ import annotations

import argparse
from pathlib import Path

from epa_o_d_fetcher import DEFAULT_YEAR, build_epa_summary, persist_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and display team EPA splits.")
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Season year to download (defaults to %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to store the generated CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Downloading play-by-play data for {args.year}...")
    summary = build_epa_summary(args.year)

    print("Team EPA splits (offense and defense):")
    print(summary)

    output_path = persist_summary(summary, args.year, output_dir=args.output_dir)
    print(f"Saved EPA summary to {output_path}")


if __name__ == "__main__":
    main()
