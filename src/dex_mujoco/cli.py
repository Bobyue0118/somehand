"""Compatibility wrapper for the CLI interface layer."""

from dex_mujoco.interfaces.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
