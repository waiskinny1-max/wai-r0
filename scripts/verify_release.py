from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wai_r0.quality.release import inspect_release, write_release_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify WAI-R0 release readiness")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_release(args.repository)
    if args.output:
        write_release_report(args.output, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
