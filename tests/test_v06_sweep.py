from __future__ import annotations

import json
from pathlib import Path

import pytest

from wai_r0.app.cli import main
from wai_r0.experiments.sweep import build_sweep_plan, load_sweep_spec


def test_sweep_plan_is_deterministic_and_resolves_configs(tmp_path: Path) -> None:
    spec = Path("configs/sweeps/recurrent_depth.yaml")
    first = build_sweep_plan(spec)
    second = build_sweep_plan(spec)
    assert first.plan_hash == second.plan_hash
    assert len(first.trials) == 6
    assert len({trial.trial_id for trial in first.trials}) == 6
    for trial in first.trials:
        candidate = Path(str(trial.manifest.execution["candidate_config"]))
        control = Path(str(trial.manifest.execution["control_config"]))
        assert candidate.is_absolute() and candidate.is_file()
        assert control.is_absolute() and control.is_file()

    output = tmp_path / "plan"
    assert main(["experiment", "sweep-plan", str(spec), "--output-dir", str(output)]) == 0
    payload = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert payload["trial_count"] == 6
    assert len(list((output / "trials").glob("*.yaml"))) == 6


def test_sweep_trial_ceiling_fails_closed(tmp_path: Path) -> None:
    spec = tmp_path / "sweep.yaml"
    spec.write_text(
        "\n".join(
            [
                "id: too-large",
                f"base_manifest: {Path('configs/experiments/recurrent_ood.yaml').resolve()}",
                "maximum_trials: 3",
                "grid:",
                "  execution.candidate_recurrent_steps: [1, 2]",
                "  execution.train_steps: [1, 2]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="above maximum_trials"):
        load_sweep_spec(spec)
