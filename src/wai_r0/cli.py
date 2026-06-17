from __future__ import annotations

import argparse
import json
from pathlib import Path

from wai_r0.benchmarks import ablate, memory, symbolic, tiny_train, zero_neural
from wai_r0.config import ReasonerConfig
from wai_r0.eval.holdout import write_holdout_tasks
from wai_r0.eval.leakage_guard import LeakageGuard
from wai_r0.report import BenchmarkReport, markdown_from_json


def seqs(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def duration(value: str) -> float:
    text = str(value).strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) * 0.001
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def cfg(path: str | None) -> ReasonerConfig:
    return ReasonerConfig.from_yaml(path or "configs/model/nano.yaml")


def emit(report: BenchmarkReport, out: str | None = None, base: str = "latest") -> None:
    outdir = Path(out).parent if out else Path("reports")
    stem = Path(out).stem if out else base
    json_path, md_path = report.write(outdir, stem)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    Path("reports/latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json_path)
    print(md_path)


def _cmd_zero(args: argparse.Namespace) -> BenchmarkReport:
    return zero_neural(cfg(args.config), args.batch_size, args.seq_len)


def _cmd_memory(args: argparse.Namespace) -> BenchmarkReport:
    return memory(cfg(args.config), args.baseline, args.candidate, args.seq_lens, args.batch_size)


def _cmd_symbolic(args: argparse.Namespace) -> BenchmarkReport:
    return symbolic(
        args.tasks,
        args.budget,
        args.max_depth,
        leakage_manifest=args.leakage_manifest,
        split=args.split,
        register_leakage=args.register_leakage,
    )


def _cmd_tiny(args: argparse.Namespace) -> BenchmarkReport:
    return tiny_train(
        cfg(args.config),
        args.task,
        args.examples,
        batch_size=args.batch_size,
        train_len=args.train_len,
        eval_lens=tuple(args.eval_lens),
    )


def _cmd_ablate(args: argparse.Namespace) -> BenchmarkReport:
    return ablate(
        cfg(args.config),
        args.matrix,
        seeds=args.seeds,
        tasks=args.tasks,
        tiny_examples=args.tiny_examples,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wai-r0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    zero = sub.add_parser("zero-neural")
    zero.add_argument("--config", default="configs/model/nano.yaml")
    zero.add_argument("--batch-size", type=int, default=1)
    zero.add_argument("--seq-len", type=int, default=16)
    zero.add_argument("--output")

    mem = sub.add_parser("memory")
    mem.add_argument("--config", default="configs/model/nano.yaml")
    mem.add_argument("--baseline", default="mha")
    mem.add_argument("--candidate", default="mla_lite")
    mem.add_argument("--seq-lens", type=seqs, required=True)
    mem.add_argument("--batch-size", type=int, default=1)
    mem.add_argument("--output")

    sym = sub.add_parser("symbolic-arc")
    sym.add_argument("--tasks", required=True)
    sym.add_argument("--budget", type=duration, default=10.0)
    sym.add_argument("--max-depth", type=int, default=2)
    sym.add_argument("--leakage-manifest")
    sym.add_argument("--split", default="dev")
    sym.add_argument("--register-leakage", action="store_true")
    sym.add_argument("--output")

    tiny = sub.add_parser("tiny-train")
    tiny.add_argument("--config", "--model", dest="config", default="configs/model/nano.yaml")
    tiny.add_argument("--task", choices=["copy", "reverse", "parity"], default="copy")
    tiny.add_argument("--examples", type=int, default=32)
    tiny.add_argument("--batch-size", type=int, default=4)
    tiny.add_argument("--train-len", type=int, default=8)
    tiny.add_argument("--eval-lens", type=seqs, default=[8, 16])
    tiny.add_argument("--output")

    abl = sub.add_parser("ablate")
    abl.add_argument("--config", default="configs/model/nano.yaml")
    abl.add_argument("--matrix", required=True)
    abl.add_argument("--seeds", type=seqs)
    abl.add_argument("--tasks", default="examples/tasks")
    abl.add_argument("--tiny-examples", type=int, default=8)
    abl.add_argument("--output")

    holdout = sub.add_parser("generate-holdout")
    holdout.add_argument("--output-dir", required=True)
    holdout.add_argument("--count", type=int, default=8)
    holdout.add_argument("--seed", type=int, default=2026)

    leak = sub.add_parser("leakage-check")
    leak.add_argument("--tasks", required=True)
    leak.add_argument("--split", default="dev")
    leak.add_argument("--manifest", default="reports/leakage_manifest.json")
    leak.add_argument("--register", action="store_true")

    rep = sub.add_parser("report")
    rep.add_argument("--input", required=True)
    rep.add_argument("--format", choices=["md"], default="md")
    rep.add_argument("--output")

    args = parser.parse_args(argv)

    if args.cmd == "zero-neural":
        emit(_cmd_zero(args), args.output, "zero_neural")
    elif args.cmd == "memory":
        emit(_cmd_memory(args), args.output, "memory")
    elif args.cmd == "symbolic-arc":
        emit(_cmd_symbolic(args), args.output, "symbolic_arc")
    elif args.cmd == "tiny-train":
        emit(_cmd_tiny(args), args.output, "tiny_train")
    elif args.cmd == "ablate":
        emit(_cmd_ablate(args), args.output, "ablation")
    elif args.cmd == "generate-holdout":
        for path in write_holdout_tasks(args.output_dir, args.count, args.seed):
            print(path)
    elif args.cmd == "leakage-check":
        guard = LeakageGuard(args.manifest)
        findings = guard.scan_directory(args.tasks, args.split, args.register)
        print(json.dumps(guard.summary(findings), indent=2, sort_keys=True))
    elif args.cmd == "report":
        md = markdown_from_json(args.input)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(args.output)
        else:
            print(md)
    return 0
