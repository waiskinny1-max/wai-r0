"""Compatibility import for the pre-v0.6 native CLI path."""

from wai_r0.app.cli import _delegate_legacy, _should_delegate, main

__all__ = ["_delegate_legacy", "_should_delegate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
