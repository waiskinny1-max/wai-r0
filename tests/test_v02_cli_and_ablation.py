from pathlib import Path

from wai_r0.benchmarks import ablate
from wai_r0.cli import main
from wai_r0.config import ReasonerConfig


def test_generate_holdout_cli(tmp_path: Path):
    out = tmp_path / "holdouts"
    code = main(["generate-holdout", "--output-dir", str(out), "--count", "2", "--seed", "7"])
    assert code == 0
    assert len(list(out.glob("*.json"))) == 2


def test_ablation_includes_symbolic_variants(tmp_path: Path):
    matrix = tmp_path / "ablation.yaml"
    matrix.write_text(
        """
seeds: [1337]
symbolic_budget_s: 1.0
symbolic_max_depth: 1
variants:
  - {name: A7, attention: none, moe: false, recurrent: false, symbolic_shell: true}
  - {name: A8, attention: mha, moe: false, recurrent: false, symbolic_shell: true}
""".strip(),
        encoding="utf-8",
    )
    cfg = ReasonerConfig(vocab_size=32, d_model=32, d_ff=64, n_heads=4, n_kv_heads=4, n_layers=1, max_seq_len=16)
    report = ablate(cfg, matrix, seeds=[1337], tasks="examples/tasks", tiny_examples=4)
    names = {row["variant"] for row in report.raw_metrics["variants"]}
    assert {"A7", "A8"}.issubset(names)
    assert "decisions" in report.raw_metrics
