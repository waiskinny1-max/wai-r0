from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable
import json

from wai_r0.symbolic import Grid, mirror_x, mirror_y, rotate90, rotate180, rotate270

Transform = Callable[[Grid], Grid]

_TRANSFORMS: dict[str, Transform] = {
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "mirror_x": mirror_x,
    "mirror_y": mirror_y,
}


@dataclass(frozen=True)
class GeneratedTaskSpec:
    task_id: str
    transform: str
    size: int
    palette_size: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _random_grid(rng: Random, size: int, palette_size: int) -> Grid:
    if size < 2:
        raise ValueError("generated grid size must be at least 2")
    if palette_size < 2:
        raise ValueError("palette_size must be at least 2")
    rows = [[rng.randrange(palette_size) for _ in range(size)] for _ in range(size)]
    return Grid.from_lists(rows)


def _make_task(index: int, seed: int, rng: Random) -> dict[str, Any]:
    name = list(_TRANSFORMS)[index % len(_TRANSFORMS)]
    transform = _TRANSFORMS[name]
    size = 2 + (index % 3)
    palette_size = 3 + (index % 4)
    train_in = _random_grid(rng, size, palette_size)
    test_in = _random_grid(rng, size, palette_size)
    task_id = f"synthetic_{seed}_{index:04d}_{name}"
    return {
        "id": task_id,
        "meta": GeneratedTaskSpec(task_id, name, size, palette_size, seed).to_dict(),
        "train": [{"input": train_in.to_lists(), "output": transform(train_in).to_lists()}],
        "test": [{"input": test_in.to_lists(), "output": transform(test_in).to_lists()}],
    }


def generate_holdout_tasks(count: int, seed: int = 1337) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = Random(seed)
    return [_make_task(index, seed, rng) for index in range(count)]


def write_holdout_tasks(output_dir: str | Path, count: int, seed: int = 1337) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for task in generate_holdout_tasks(count=count, seed=seed):
        path = root / f"{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths
