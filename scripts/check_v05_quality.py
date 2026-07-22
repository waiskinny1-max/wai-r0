"""Backward-compatible entry point for the canonical quality gate."""

from check_quality import main

if __name__ == "__main__":
    raise SystemExit(main())
