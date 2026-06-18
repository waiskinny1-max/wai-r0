from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_source_tree() -> None:
    """Make `python main.py` work from a fresh source checkout.

    WAI-R0 uses a standard `src/` layout. That is correct for packaging, but a
    direct `python main.py` invocation does not automatically add `src/` to
    `sys.path`. Keep this bootstrap small and local to the executable wrapper so
    the package itself still behaves normally when installed with `pip install -e .`.
    """

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        src_text = str(src_dir)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)


_bootstrap_source_tree()

from wai_r0.cli import main  # noqa: E402  # import after source-tree bootstrap


if __name__ == "__main__":
    raise SystemExit(main())
