from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from wai_r0.benchmarks import ablate, architecture_priors, memory, symbolic, tiny_train, zero_neural
from wai_r0.config import ReasonerConfig
from wai_r0.eval.holdout import write_holdout_tasks
from wai_r0.eval.leakage_guard import LeakageGuard
from wai_r0.eval.suite import run_suite
from wai_r0.report import BenchmarkReport, markdown_from_json
from wai_r0.training.language_csv import (
    CSVSplitSpec,
    CSVTrainingStep,
    audit_language_csv,
    csv_language_probe_report,
    generate_from_csv_checkpoint,
    inspect_language_csv,
    iter_generate_from_csv_checkpoint,
)
from wai_r0.training.markdown_plan import run_markdown_training_plan


def normalize_legacy_train_args(argv: list[str] | None) -> list[str] | None:
    """Support `python main.py -train training.md` and CSV shorthand."""

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


def _cmd_prior(args: argparse.Namespace) -> BenchmarkReport:
    return architecture_priors(
        cfg(args.config),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        recurrent_depths=tuple(args.recurrent_depths),
    )


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


def _cmd_train(args: argparse.Namespace) -> tuple[BenchmarkReport, str | None]:
    path = Path(args.plan)
    if path.suffix.lower() == ".csv":
        report = _csv_report_from_args(args, csv_path=path)
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
    _write_json_payload(inspection.to_dict(), args.output)


def _cmd_audit_csv(args: argparse.Namespace) -> None:
    audit = audit_language_csv(
        args.csv,
        text_column=args.text_column,
        target_column=args.target_column,
        max_rows=args.max_rows,
        split_spec=CSVSplitSpec(args.train_fraction, args.val_fraction, args.test_fraction, args.split_seed),
        use_declared_split=args.respect_csv_split,
    )
    _write_json_payload(audit.to_dict(), args.output)


def _write_json_payload(payload: dict, output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path)
    else:
        print(text)


def _stream_training_step(step: CSVTrainingStep) -> None:
    print("[train] " + json.dumps(step.to_dict(), sort_keys=True), flush=True)


def _csv_report_from_args(args: argparse.Namespace, csv_path: str | Path | None = None) -> BenchmarkReport:
    return csv_language_probe_report(
        cfg(args.config),
        csv_path=csv_path or args.csv,
        text_column=args.text_column,
        target_column=args.target_column,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_rows=args.max_rows,
        lr=args.lr,
        eval_rows=args.eval_rows,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        resume_from=args.resume_from,
        eval_interval=args.eval_interval,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        baseline_rows=args.baseline_rows,
        use_declared_split=args.respect_csv_split,
        allow_train_eval_fallback=args.allow_train_eval_fallback,
        progress_callback=_stream_training_step if getattr(args, "stream", False) else None,
    )


def _cmd_train_csv(args: argparse.Namespace) -> BenchmarkReport:
    return _csv_report_from_args(args)


def _cmd_ablate(args: argparse.Namespace) -> BenchmarkReport:
    return ablate(
        cfg(args.config),
        args.matrix,
        seeds=args.seeds,
        tasks=args.tasks,
        tiny_examples=args.tiny_examples,
    )


def _cmd_suite(args: argparse.Namespace) -> None:
    result = run_suite(cfg(args.config), args.suite)
    for json_path, md_path in result.written:
        print(json_path)
        print(md_path)


def _cmd_sample_csv(args: argparse.Namespace) -> None:
    if args.stream:
        chunks: list[str] = []
        for chunk in iter_generate_from_csv_checkpoint(args.checkpoint, prompt=args.prompt, max_new_tokens=args.max_new_tokens):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        text = "".join(chunks)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"\n{path}")
        else:
            print()
        return
    text = generate_from_csv_checkpoint(args.checkpoint, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        print(text)


def add_csv_training_args(parser: argparse.ArgumentParser, *, include_csv_path: bool) -> None:
    if include_csv_path:
        parser.add_argument("--csv", required=True, help="CSV path, for example training/basic_lang.csv")
    parser.add_argument("--config", "--model", dest="config", default="configs/model/nano.yaml")
    parser.add_argument("--text-column")
    parser.add_argument("--target-column")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-rows", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--baseline-rows", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--respect-csv-split", action="store_true", help="use declared split column instead of default hash split")
    parser.add_argument(
        "--allow-train-eval-fallback",
        action="store_true",
        help="allow validation to fall back to train rows; use only for smoke tests",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-from")
    parser.add_argument("--log")
    parser.add_argument("--output")
    parser.add_argument("--stream", action="store_true", help="print compact JSON progress lines while training")


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if not raw_args:
        from wai_r0.gui import launch_gui

        launch_gui()
        return 0
    argv = normalize_legacy_train_args(raw_args)
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
    add_csv_training_args(train, include_csv_path=False)

    inspect_csv = sub.add_parser("inspect-csv", help="quickly inspect a CSV file before language-probe training")
    inspect_csv.add_argument("--csv", required=True, help="CSV path, for example training/basic_lang.csv")
    inspect_csv.add_argument("--text-column")
    inspect_csv.add_argument("--target-column")
    inspect_csv.add_argument("--sample-rows", type=int, default=1000)
    inspect_csv.add_argument("--output")

    audit_csv = sub.add_parser("audit-csv", help="stream a CSV audit with split counts and duplicate rate")
    audit_csv.add_argument("--csv", required=True, help="CSV path, for example training/basic_lang.csv")
    audit_csv.add_argument("--text-column")
    audit_csv.add_argument("--target-column")
    audit_csv.add_argument("--max-rows", type=int)
    audit_csv.add_argument("--train-fraction", type=float, default=0.90)
    audit_csv.add_argument("--val-fraction", type=float, default=0.05)
    audit_csv.add_argument("--test-fraction", type=float, default=0.05)
    audit_csv.add_argument("--split-seed", type=int, default=1337)
    audit_csv.add_argument("--respect-csv-split", action="store_true", help="use declared split column instead of hash split")
    audit_csv.add_argument("--output")

    train_csv = sub.add_parser("train-csv", help="run a held-out byte-level CSV language-readiness experiment")
    add_csv_training_args(train_csv, include_csv_path=True)

    sample_csv = sub.add_parser("sample-csv", help="greedy byte-level sample from a CSV checkpoint")
    sample_csv.add_argument("--checkpoint", required=True)
    sample_csv.add_argument("--prompt", default="")
    sample_csv.add_argument("--max-new-tokens", type=int, default=64)
    sample_csv.add_argument("--stream", action="store_true", help="stream generated text chunks as they are produced")
    sample_csv.add_argument("--output")

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

    sub.add_parser("gui", help="open the Tkinter local training and checkpoint console")

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
    elif args.cmd == "audit-csv":
        _cmd_audit_csv(args)
    elif args.cmd == "train-csv":
        emit(_cmd_train_csv(args), args.output, "csv_language_readiness")
    elif args.cmd == "sample-csv":
        _cmd_sample_csv(args)
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
    elif args.cmd == "gui":
        from wai_r0.gui import launch_gui

        launch_gui()
    elif args.cmd == "report":
        md = markdown_from_json(args.input)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(args.output)
        else:
            print(md)
    return 0
