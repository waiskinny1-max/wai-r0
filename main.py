from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_layout() -> None:
    """Allow ``python main.py`` from an uninstalled source checkout."""

    source = Path(__file__).resolve().parent / "src"
    source_text = str(source)
    if source.is_dir() and source_text not in sys.path:
        sys.path.insert(0, source_text)


def main() -> int:
    _bootstrap_src_layout()
    from wai_r0.app.cli import main as entrypoint

    return entrypoint()


if __name__ == "__main__":
    raise SystemExit(main())
