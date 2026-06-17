import json
from pathlib import Path

from wai_r0.eval.holdout import generate_holdout_tasks, write_holdout_tasks
from wai_r0.eval.leakage_guard import LeakageGuard
from wai_r0.training.probes import run_tiny_probe
from wai_r0.config import ReasonerConfig


def test_generate_holdout_tasks_are_deterministic(tmp_path: Path):
    first = generate_holdout_tasks(3, seed=99)
    second = generate_holdout_tasks(3, seed=99)
    assert first == second
    paths = write_holdout_tasks(tmp_path, count=2, seed=99)
    assert len(paths) == 2
    assert json.loads(paths[0].read_text())["id"].startswith("synthetic_99_")


def test_leakage_guard_detects_cross_split_duplicate(tmp_path: Path):
    task = {
        "id": "same",
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    guard = LeakageGuard(tmp_path / "manifest.json")
    first = guard.check_file(path, split="dev", register=True)
    guard.save()
    second = LeakageGuard(tmp_path / "manifest.json").check_file(path, split="public_eval", register=False)
    assert first.status == "new"
    assert second.status == "cross_split_duplicate"


def test_tiny_probe_reports_length_eval():
    cfg = ReasonerConfig(vocab_size=32, d_model=32, d_ff=64, n_heads=4, n_kv_heads=4, n_layers=1, max_seq_len=16)
    result = run_tiny_probe(cfg, task="copy", examples=4, batch_size=2, train_len=4, eval_lens=(4, 8))
    data = result.to_dict()
    assert data["train_len"] == 4
    assert set(data["eval_token_accuracy"]) == {"4", "8"}
