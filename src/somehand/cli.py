"""CLI entry point."""

from somehand.interfaces.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
