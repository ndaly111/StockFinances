#!/usr/bin/env python3
"""
Deprecated entrypoint.

This repository's canonical orchestrator is main_remote.py.
Use this wrapper so `python main.py` continues to work.
"""

from main_remote import mini_main


def main() -> None:
    mini_main()


if __name__ == "__main__":
    main()
