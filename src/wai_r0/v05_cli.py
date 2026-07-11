from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from wai_r0.app.services import (
    LanguageTrainingRequest,
    build_dataset_manifest,
    run_language_training,
)
from wai_r0.config import ReasonerConfig
from wai_r0.core.reproducibility import atomic_write_json
from wai_r0.data.csv_reader import audit_conversation_csv
from wai_r0.data.splits import SplitSpec
from wai_r0.experiments.manifest import load_experiment_manifest
from wai_r0.experiments.runner import run_experiment
from wai_r0.model import ModelOutput, ReasonerCore
from wai_r0.profiler import profile_model
from wai_r0.reporting import load_report, write_rendered_report
from wai_r0.training.checkpoint import inspect_checkpoint
from wai_r0.version import __version__

_NATIVE_COMMANDS = {
    "version",
    "doctor",
    "config",
    "data",
    "model",
    "profile",
    "train",
    "experiment",
    "report",
    "checkpoint",
    "reproduce",
}


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--respect-declared-split", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wai-r0",
        description="WAI-R0 v0.5 evidence-first architecture research tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the canonical package version")
    subparsers.add_parser("doctor", help="inspect the local runtime without changing it")

    config_parser = subparsers.add_parser("config", help="model configuration operations")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_validate = config_subparsers.add_parser("validate", help="validate a model YAML file")
    config_validate.add_argument("path", type=Path)

    data_parser = subparsers.add_parser("data", help="dataset operations")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_audit = data_subparsers.add_parser("audit", help="stream and audit a conversation CSV")
    data_audit.add_argument("path", type=Path)
    data_audit.add_argument("--output", type=Path)
    data_audit.add_argument("--rejection-samples", type=int, default=20)
    data_audit.add_argument("--max-rows", type=int)
    _split_arguments(data_audit)
    data_manifest = data_subparsers.add_parser(
        "manifest", help="audit a CSV and write a hash-verified data manifest"
    )
    data_manifest.add_argument("path", type=Path)
    data_manifest.add_argument("--output", type=Path, required=True)
    data_manifest.add_argument("--max-rows", type=int)
    _split_arguments(data_manifest)

    model_parser = subparsers.add_parser("model", help="model inspection operations")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    model_inspect = model_subparsers.add_parser("inspect", help="construct and inspect a model")
    model_inspect.add_argument("--config", type=Path, required=True)
    model_inspect.add_argument("--batch-size", type=int, default=1)
    model_inspect.add_argument("--seq-len", type=int, default=16)
    model_inspect.add_argument("--diagnostics", action="store_true")

    profile_parser = subparsers.add_parser("profile", help="measure prefill/decode and cache use")
    profile_parser.add_argument("--config", type=Path, required=True)
    profile_parser.add_argument("--batch-size", type=int, default=1)
    profile_parser.add_argument("--seq-len", type=int, default=32)
    profile_parser.add_argument("--warmup-runs", type=int, default=2)
    profile_parser.add_argument("--measured-runs", type=int, default=5)
    profile_parser.add_argument(
        "--cpu-threads",
        type=int,
        help="temporary PyTorch intra-op thread count for CPU measurements",
    )
    profile_parser.add_argument("--output", type=Path)

    train_parser = subparsers.add_parser("train", help="native v0.5 training operations")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_csv = train_subparsers.add_parser(
        "csv", help="train or exactly resume the byte-level conversation control"
    )
    train_csv.add_argument("path", type=Path)
    train_csv.add_argument("--config", type=Path, required=True)
    train_csv.add_argument("--output-dir", type=Path, required=True)
    budget_group = train_csv.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--max-steps", type=int)
    budget_group.add_argument("--max-target-tokens", type=int)
    train_csv.add_argument("--batch-size", type=int, default=8)
    train_csv.add_argument("--seq-len", type=int, default=128)
    train_csv.add_argument("--learning-rate", type=float, default=3e-4)
    train_csv.add_argument("--weight-decay", type=float, default=0.01)
    train_csv.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_csv.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="none")
    train_csv.add_argument("--schedule", choices=["constant", "linear", "cosine"], default="cosine")
    train_csv.add_argument("--warmup-steps", type=int, default=0)
    train_csv.add_argument("--checkpoint-every", type=int, default=0)
    train_csv.add_argument("--evaluate-every", type=int, default=0)
    train_csv.add_argument("--validation-batches", type=int, default=8)
    train_csv.add_argument("--max-rows", type=int)
    train_csv.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=256,
        help="bounded deterministic training shuffle; 0 disables shuffling",
    )
    train_csv.add_argument("--shuffle-seed", type=int)
    train_csv.add_argument(
        "--cpu-threads",
        type=int,
        help="temporary PyTorch intra-op thread count for CPU training",
    )
    train_csv.add_argument(
        "--pack-sequences",
        action="store_true",
        help="greedily pack examples with block-diagonal causal attention",
    )
    train_csv.add_argument("--resume-from", type=Path)
    train_csv.add_argument(
        "--allow-missing-checkpoint-digest",
        action="store_true",
        help="permit resume from a trusted local checkpoint without its .sha256 sidecar",
    )
    _split_arguments(train_csv)

    experiment_parser = subparsers.add_parser("experiment", help="experiment operations")
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command", required=True
    )
    experiment_validate = experiment_subparsers.add_parser(
        "validate", help="validate and hash an experiment manifest"
    )
    experiment_validate.add_argument("path", type=Path)
    experiment_run = experiment_subparsers.add_parser(
        "run", help="execute a preregistered candidate/control experiment"
    )
    experiment_run.add_argument("path", type=Path)
    experiment_run.add_argument("--output", type=Path, required=True)
    experiment_run.add_argument("--repository", type=Path, default=Path.cwd())
    experiment_run.add_argument(
        "--render", choices=["none", "markdown", "html", "both"], default="both"
    )

    report_parser = subparsers.add_parser("report", help="v0.5 report operations")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_validate = report_subparsers.add_parser("validate", help="validate a v0.5 report")
    report_validate.add_argument("path", type=Path)
    report_render = report_subparsers.add_parser(
        "render", help="render JSON report to Markdown or HTML"
    )
    report_render.add_argument("path", type=Path)
    report_render.add_argument("--output", type=Path, required=True)
    report_render.add_argument("--format", choices=["markdown", "html"])

    checkpoint_parser = subparsers.add_parser("checkpoint", help="checkpoint operations")
    checkpoint_subparsers = checkpoint_parser.add_subparsers(
        dest="checkpoint_command", required=True
    )
    checkpoint_inspect = checkpoint_subparsers.add_parser(
        "inspect", help="inspect a trusted local checkpoint"
    )
    checkpoint_inspect.add_argument("path", type=Path)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="verify report provenance and optionally rerun its experiment manifest",
    )
    reproduce.add_argument("path", type=Path)
    reproduce.add_argument("--execute", action="store_true")
    reproduce.add_argument("--output", type=Path)
    return parser


def _split_spec(args: argparse.Namespace) -> SplitSpec:
    return SplitSpec(
        train=args.train_fraction,
        val=args.val_fraction,
        test=args.test_fraction,
        seed=args.split_seed,
        respect_declared=args.respect_declared_split,
    )


def _cuda_bf16_supported(device_index: int) -> bool:
    with torch.cuda.device(device_index):
        return bool(torch.cuda.is_bf16_supported())


def _doctor() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "wai_r0_version": __version__,
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 10),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "yaml": getattr(yaml, "__version__", "unknown"),
        "tkinter_available": importlib.util.find_spec("tkinter") is not None,
    }
    if torch.cuda.is_available():
        checks.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_devices": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
                "bf16_supported": all(
                    _cuda_bf16_supported(index) for index in range(torch.cuda.device_count())
                ),
            }
        )
    checks["status"] = "ok" if checks["python_supported"] else "unsupported_python"
    return checks


def _model_inspect(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.seq_len < 1:
        raise ValueError("batch-size and seq-len must be positive")
    config = ReasonerConfig.from_yaml(args.config)
    if args.seq_len > config.max_seq_len:
        raise ValueError("seq-len exceeds max_seq_len")
    core = ReasonerCore(config)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        device=core.device_obj,
    )
    with torch.inference_mode():
        output = core(tokens, return_dict=True, collect_diagnostics=args.diagnostics)
    if not isinstance(output, ModelOutput):
        raise RuntimeError("structured output was not returned")
    parameter = next(core.parameters())
    return {
        "version": __version__,
        "config": config.to_dict(),
        "parameter_counts": core.parameter_counts(),
        "effective_device": str(parameter.device),
        "effective_dtype": str(parameter.dtype).removeprefix("torch."),
        "logits_shape": list(output.logits.shape),
        "finite": bool(torch.isfinite(output.logits).all().detach().cpu()),
        "estimated_memory": core.estimate_memory_cost(args.seq_len, args.batch_size),
        "diagnostics": output.diagnostics,
    }


def _training_request(args: argparse.Namespace) -> LanguageTrainingRequest:
    return LanguageTrainingRequest(
        model_config=args.config,
        csv_path=args.path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_target_tokens=args.max_target_tokens,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        checkpoint_every=args.checkpoint_every,
        evaluate_every=args.evaluate_every,
        validation_batches=args.validation_batches,
        max_rows=args.max_rows,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        respect_declared_split=args.respect_declared_split,
        resume_from=args.resume_from,
        require_checkpoint_digest=not args.allow_missing_checkpoint_digest,
        shuffle_buffer_size=args.shuffle_buffer_size,
        shuffle_seed=args.shuffle_seed,
        pack_sequences=args.pack_sequences,
        cpu_threads=args.cpu_threads,
    )


def _render_experiment_outputs(args: argparse.Namespace, report_path: Path) -> dict[str, str]:
    report = load_report(report_path)
    outputs: dict[str, str] = {"json": str(report_path)}
    stem = report_path.with_suffix("")
    if args.render in {"markdown", "both"}:
        outputs["markdown"] = str(write_rendered_report(stem.with_suffix(".md"), report))
    if args.render in {"html", "both"}:
        outputs["html"] = str(write_rendered_report(stem.with_suffix(".html"), report))
    return outputs


def _reproduce(args: argparse.Namespace) -> dict[str, Any]:
    report = load_report(args.path)
    manifest_path = report.artifacts.get("manifest") or report.provenance.get("manifest_path")
    summary: dict[str, Any] = {
        "report_valid": True,
        "run_id": report.identity.run_id,
        "experiment_hash": report.identity.experiment_hash,
        "manifest": manifest_path,
        "command": report.identity.command,
        "executed": False,
    }
    if not args.execute:
        return summary
    if not manifest_path:
        raise ValueError(
            "report does not identify an experiment manifest; refusing arbitrary replay"
        )
    manifest = load_experiment_manifest(manifest_path)
    if manifest.manifest_hash != report.identity.experiment_hash:
        raise ValueError("current experiment manifest hash differs from the report")
    destination = args.output or args.path.with_name(args.path.stem + ".reproduced.json")
    reproduced = run_experiment(manifest_path, output=destination, repository=Path.cwd())
    summary.update(
        {
            "executed": True,
            "output": str(destination),
            "reproduced_run_id": reproduced.identity.run_id,
            "reproduced_decision": reproduced.decision,
            "decision_matches": reproduced.decision == report.decision,
        }
    )
    return summary


def _run_native(args: argparse.Namespace) -> int:
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "doctor":
        _json_print(_doctor())
        return 0
    if args.command == "config" and args.config_command == "validate":
        config = ReasonerConfig.from_yaml(args.path)
        _json_print({"valid": True, "config": config.to_dict()})
        return 0
    if args.command == "data" and args.data_command == "audit":
        audit_payload = audit_conversation_csv(
            args.path,
            rejection_sample_limit=args.rejection_samples,
            max_rows=args.max_rows,
            split_spec=_split_spec(args),
        ).to_dict()
        if args.output:
            atomic_write_json(args.output, audit_payload)
        _json_print(audit_payload)
        return 0
    if args.command == "data" and args.data_command == "manifest":
        dataset_audit, output = build_dataset_manifest(
            args.path,
            output=args.output,
            split_spec=_split_spec(args),
            max_rows=args.max_rows,
        )
        _json_print({"output": str(output), "audit": dataset_audit.to_dict()})
        return 0
    if args.command == "model" and args.model_command == "inspect":
        _json_print(_model_inspect(args))
        return 0
    if args.command == "profile":
        config = ReasonerConfig.from_yaml(args.config)
        core = ReasonerCore(config)
        result = profile_model(
            core.transformer,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            cpu_threads=args.cpu_threads,
        ).to_dict()
        if args.output:
            atomic_write_json(args.output, result)
        _json_print(result)
        return 0
    if args.command == "train" and args.train_command == "csv":
        _json_print(run_language_training(_training_request(args)).to_dict())
        return 0
    if args.command == "experiment" and args.experiment_command == "validate":
        manifest = load_experiment_manifest(args.path)
        _json_print(
            {"valid": True, "manifest_hash": manifest.manifest_hash, "manifest": manifest.to_dict()}
        )
        return 0
    if args.command == "experiment" and args.experiment_command == "run":
        report = run_experiment(args.path, output=args.output, repository=args.repository)
        _json_print(
            {
                "run_id": report.identity.run_id,
                "decision": report.decision,
                "outputs": _render_experiment_outputs(args, args.output),
            }
        )
        return 0
    if args.command == "report" and args.report_command == "validate":
        report = load_report(args.path)
        _json_print(
            {
                "valid": True,
                "schema_version": report.schema_version,
                "run_id": report.identity.run_id,
                "decision": report.decision,
            }
        )
        return 0
    if args.command == "report" and args.report_command == "render":
        report = load_report(args.path)
        output = write_rendered_report(args.output, report, format=args.format)
        _json_print({"output": str(output), "run_id": report.identity.run_id})
        return 0
    if args.command == "checkpoint" and args.checkpoint_command == "inspect":
        _json_print(inspect_checkpoint(args.path))
        return 0
    if args.command == "reproduce":
        _json_print(_reproduce(args))
        return 0
    raise RuntimeError("unhandled command")


def _delegate_legacy(arguments: Sequence[str]) -> int:
    try:
        from wai_r0.cli import main as legacy_main
    except ImportError as exc:
        raise SystemExit(
            "This command belongs to the v0.4 compatibility surface, but wai_r0.cli "
            "could not be imported. Apply this patch over the existing repository."
        ) from exc
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        result = legacy_main()
    finally:
        sys.argv = original
    return int(result) if isinstance(result, int) else 0


def _should_delegate(arguments: Sequence[str]) -> bool:
    if not arguments:
        # Preserve `python main.py` opening the existing GUI/terminal workbench
        # when v0.5 is applied as an overlay to the live repository.
        return importlib.util.find_spec("wai_r0.cli") is not None
    if arguments[0] not in _NATIVE_COMMANDS:
        return True
    # Preserve the existing `wai-r0 train <plan.md>` and `wai-r0 report --input ...`
    # surfaces. Native v0.5 forms always use an explicit nested operation.
    if arguments[0] == "train" and (len(arguments) < 2 or arguments[1] != "csv"):
        return True
    return arguments[0] == "report" and (
        len(arguments) < 2 or arguments[1] not in {"validate", "render"}
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _should_delegate(arguments):
        return _delegate_legacy(arguments)
    parser = _build_parser()
    try:
        parsed = parser.parse_args(arguments)
        return _run_native(parsed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
