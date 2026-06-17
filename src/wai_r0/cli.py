from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from wai_r0.benchmarks import ablate, architecture_priors, memory, symbolic, tiny_train, zero_neural
from wai_r0.config import ReasonerConfig
from wai_r0.eval.holdout import write_holdout_tasks
from wai_r0.eval.leakage_guard import LeakageGuard
from wai_r0.report import BenchmarkReport, markdown_from_json
from wai_r0.eval.suite import run_suite
from wai_r0.training.language_csv import csv_language_probe_report, inspect_language_csv
from wai_r0.training.markdown_plan import run_markdown_training_plan



def normalize_legacy_train_args(argv: list[str] | None) -> list[str] | None:
    """Support `python main.py -train training.md` and CSV shorthand.

    The project CLI is subcommand-based. The legacy `-train` spelling is kept
    as a convenience alias because early WAI-R0 examples used it directly. A
    `.csv` path is normalized to `train-csv --csv ...`; everything else is
    treated as a Markdown training plan and normalized to `train ...`.
    """

    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    if args and args[0] in {"-train", "--train"}:
        if len(args) < 2:
            raise SystemExit("-train requires a training file, e.g. python main.py -train training.md or training.csv")
        training_path = args[1]
        if Path(training_path).suffix.lower() == ".csv":
            return ["train-csv", "--csv", training_path, *args[2:]]
        return ["train", training_path, *args[2:]]
    return args

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



def _cmd_prior(args: argparse.Namespace) -> BenchmarkReport:
    return architecture_priors(
        cfg(args.config),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        recurrent_depths=tuple(args.recurrent_depths),
    )


def _cmd_suite(args: argparse.Namespace) -> None:
    result = run_suite(cfg(args.config), args.suite)
    for json_path, md_path in result.written:
        print(json_path)
        print(md_path)

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


def _cmd_train(args: argparse.Namespace) -> tuple[BenchmarkReport, str | None]:
    path = Path(args.plan)
    if path.suffix.lower() == ".csv":
        report = csv_language_probe_report(
            cfg(args.config),
            csv_path=path,
            text_column=args.text_column,
            target_column=args.target_column,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_rows=args.max_rows,
            lr=args.lr,
            eval_rows=args.eval_rows,
            checkpoint_path=args.checkpoint,
        )
        return report, args.output
    report, plan = run_markdown_training_plan(args.plan)
    return report, args.output or plan.output

def _cmd_inspect_csv(args: argparse.Namespace) -> None:
    inspection = inspect_language_csv(
        args.csv,
        text_column=args.text_column,
        target_column=args.target_column,
        sample_rows=args.sample_rows,
    )
    payload = json.dumps(inspection.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)


def _cmd_train_csv(args: argparse.Namespace) -> BenchmarkReport:
    return csv_language_probe_report(
        cfg(args.config),
        csv_path=args.csv,
        text_column=args.text_column,
        target_column=args.target_column,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_rows=args.max_rows,
        lr=args.lr,
        eval_rows=args.eval_rows,
        checkpoint_path=args.checkpoint,
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
    argv = normalize_legacy_train_args(argv)
    parser = argparse.ArgumentParser(prog="wai-r0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    zero = sub.add_parser("zero-neural")
    zero.add_argument("--config", default="configs/model/nano.yaml")
    zero.add_argument("--batch-size", type=int, default=1)
    zero.add_argument("--seq-len", type=int, default=16)
    zero.add_argument("--output")

    prior = sub.add_parser("architecture-priors")
    prior.add_argument("--config", default="configs/model/nano.yaml")
    prior.add_argument("--batch-size", type=int, default=2)
    prior.add_argument("--seq-len", type=int, default=16)
    prior.add_argument("--recurrent-depths", type=seqs, default=[1, 2, 4])
    prior.add_argument("--output")

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

    train = sub.add_parser("train", help="run training from Markdown or CSV")
    train.add_argument("plan", help="Markdown training plan or CSV language data file")
    train.add_argument("--output", help="override the output stem/path declared in the Markdown plan")
    train.add_argument("--config", "--model", dest="config", default="configs/model/nano.yaml")
    train.add_argument("--text-column")
    train.add_argument("--target-column")
    train.add_argument("--steps", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seq-len", type=int, default=64)
    train.add_argument("--max-rows", type=int)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--eval-rows", type=int, default=8)
    train.add_argument("--checkpoint")

    inspect_csv = sub.add_parser("inspect-csv", help="inspect a CSV file before language-probe training")
    inspect_csv.add_argument("--csv", required=True, help="CSV path, for example training/basic_lang.csv")
    inspect_csv.add_argument("--text-column")
    inspect_csv.add_argument("--target-column")
    inspect_csv.add_argument("--sample-rows", type=int, default=1000)
    inspect_csv.add_argument("--output")

    train_csv = sub.add_parser("train-csv", help="run a byte-level CSV language probe")
    train_csv.add_argument("--csv", required=True, help="CSV path, for example training/basic_lang.csv")
    train_csv.add_argument("--config", "--model", dest="config", default="configs/model/nano.yaml")
    train_csv.add_argument("--text-column")
    train_csv.add_argument("--target-column")
    train_csv.add_argument("--steps", type=int, default=25)
    train_csv.add_argument("--batch-size", type=int, default=4)
    train_csv.add_argument("--seq-len", type=int, default=64)
    train_csv.add_argument("--max-rows", type=int)
    train_csv.add_argument("--lr", type=float, default=3e-4)
    train_csv.add_argument("--eval-rows", type=int, default=8)
    train_csv.add_argument("--checkpoint")
    train_csv.add_argument("--output")

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

    suite = sub.add_parser("suite")
    suite.add_argument("--config", default="configs/model/nano.yaml")
    suite.add_argument("--suite", default="configs/benchmark/suite.yaml")

    rep = sub.add_parser("report")
    rep.add_argument("--input", required=True)
    rep.add_argument("--format", choices=["md"], default="md")
    rep.add_argument("--output")

    args = parser.parse_args(argv)

    if args.cmd == "zero-neural":
        emit(_cmd_zero(args), args.output, "zero_neural")
    elif args.cmd == "architecture-priors":
        emit(_cmd_prior(args), args.output, "architecture_priors")
    elif args.cmd == "memory":
        emit(_cmd_memory(args), args.output, "memory")
    elif args.cmd == "symbolic-arc":
        emit(_cmd_symbolic(args), args.output, "symbolic_arc")
    elif args.cmd == "tiny-train":
        emit(_cmd_tiny(args), args.output, "tiny_train")
    elif args.cmd == "train":
        report, output = _cmd_train(args)
        emit(report, output, "train_md")
    elif args.cmd == "inspect-csv":
        _cmd_inspect_csv(args)
    elif args.cmd == "train-csv":
        emit(_cmd_train_csv(args), args.output, "csv_language_probe")
    elif args.cmd == "ablate":
        emit(_cmd_ablate(args), args.output, "ablation")
    elif args.cmd == "generate-holdout":
        for path in write_holdout_tasks(args.output_dir, args.count, args.seed):
            print(path)
    elif args.cmd == "leakage-check":
        guard = LeakageGuard(args.manifest)
        findings = guard.scan_directory(args.tasks, args.split, args.register)
        print(guard.summary(findings))
    elif args.cmd == "suite":
        _cmd_suite(args)
    elif args.cmd == "report":
        md = markdown_from_json(args.input)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(args.output)
        else:
            print(md)
    return 0
