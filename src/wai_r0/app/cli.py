from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from wai_r0.app.v06_services import CompiledTrainingRequest, run_compiled_training
from wai_r0.config import ReasonerConfig
from wai_r0.core.reproducibility import atomic_write_json
from wai_r0.core.runtime import temporary_torch_threads
from wai_r0.data.chat import encode_chat_prompt
from wai_r0.data.compiled import (
    CompiledDatasetManifest,
    StatefulCompiledBatchStream,
    compile_conversation_dataset,
    verify_compiled_dataset,
)
from wai_r0.data.csv_reader import audit_conversation_csv
from wai_r0.data.splits import SplitSpec
from wai_r0.eval.context import evaluate_context_task
from wai_r0.eval.generation import diagnose_generation
from wai_r0.eval.language import evaluate_language_batches
from wai_r0.experiments.manifest import load_experiment_manifest
from wai_r0.experiments.runner import run_experiment
from wai_r0.experiments.sweep import build_sweep_plan, run_sweep, write_sweep_plan
from wai_r0.hardware import calibrate_model, estimate_training_memory, runtime_capabilities
from wai_r0.inference import SamplingConfig, generate_tokens
from wai_r0.model import ModelOutput, ReasonerCore
from wai_r0.profiler import profile_model
from wai_r0.quality import inspect_release, write_release_report
from wai_r0.registry import RunRegistry, register_report
from wai_r0.reporting import load_report, write_rendered_report
from wai_r0.tokenization import (
    TokenizerTrainingConfig,
    train_bpe_from_conversation_csv,
)
from wai_r0.tokenization.io import load_tokenizer
from wai_r0.training.checkpoint import inspect_checkpoint, load_checkpoint
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
    "release",
    "hardware",
    "tokenizer",
    "infer",
    "eval",
    "runs",
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
        description="WAI-R0 v0.6 ground-truth model research tools.",
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

    train_parser = subparsers.add_parser("train", help="native v0.6 training operations")
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
    experiment_sweep_plan = experiment_subparsers.add_parser(
        "sweep-plan", help="expand a bounded deterministic experiment sweep"
    )
    experiment_sweep_plan.add_argument("path", type=Path)
    experiment_sweep_plan.add_argument("--output-dir", type=Path, required=True)
    experiment_sweep_run = experiment_subparsers.add_parser(
        "sweep-run", help="execute a bounded sweep sequentially"
    )
    experiment_sweep_run.add_argument("path", type=Path)
    experiment_sweep_run.add_argument("--output-dir", type=Path, required=True)
    experiment_sweep_run.add_argument("--repository", type=Path, default=Path.cwd())
    experiment_sweep_run.add_argument("--maximum-trials", type=int)
    experiment_sweep_run.add_argument("--stop-on-failure", action="store_true")

    report_parser = subparsers.add_parser("report", help="research report operations")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_validate = report_subparsers.add_parser("validate", help="validate a research report")
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

    release_parser = subparsers.add_parser("release", help="release-readiness operations")
    release_subparsers = release_parser.add_subparsers(dest="release_command", required=True)
    release_doctor = release_subparsers.add_parser(
        "doctor", help="verify merged-repository release readiness"
    )
    release_doctor.add_argument("--repository", type=Path, default=Path.cwd())
    release_doctor.add_argument("--output", type=Path)

    hardware_parser = subparsers.add_parser("hardware", help="hardware inspection and calibration")
    hardware_subparsers = hardware_parser.add_subparsers(dest="hardware_command", required=True)
    hardware_subparsers.add_parser("inspect", help="show CPU/CUDA runtime capabilities")
    hardware_calibrate = hardware_subparsers.add_parser(
        "calibrate", help="measure safe CUDA batch/sequence profiles"
    )
    hardware_calibrate.add_argument("--config", type=Path, required=True)
    hardware_calibrate.add_argument("--batch-sizes", default="1,2,4,8")
    hardware_calibrate.add_argument("--sequence-lengths", default="64,128,256,512")
    hardware_calibrate.add_argument("--precisions", default="bf16,fp16,none")
    hardware_calibrate.add_argument("--target-memory-fraction", type=float, default=0.90)
    hardware_calibrate.add_argument("--output", type=Path)
    hardware_estimate = hardware_subparsers.add_parser(
        "estimate", help="estimate training memory before allocation"
    )
    hardware_estimate.add_argument("--config", type=Path, required=True)
    hardware_estimate.add_argument("--batch-size", type=int, default=1)
    hardware_estimate.add_argument("--seq-len", type=int, default=128)
    hardware_estimate.add_argument(
        "--mixed-precision", choices=["none", "fp16", "bf16"], default="none"
    )
    hardware_estimate.add_argument("--activation-checkpointing", action="store_true")

    tokenizer_parser = subparsers.add_parser("tokenizer", help="tokenizer training and inspection")
    tokenizer_subparsers = tokenizer_parser.add_subparsers(dest="tokenizer_command", required=True)
    tokenizer_train = tokenizer_subparsers.add_parser(
        "train", help="train deterministic byte-level BPE from conversation CSV"
    )
    tokenizer_train.add_argument("path", type=Path)
    tokenizer_train.add_argument("--output", type=Path, required=True)
    tokenizer_train.add_argument("--vocab-size", type=int, default=4096)
    tokenizer_train.add_argument("--min-frequency", type=int, default=2)
    tokenizer_train.add_argument("--normalization", choices=["none", "nfkc"], default="none")
    tokenizer_train.add_argument("--max-rows", type=int)
    tokenizer_train.add_argument("--max-training-bytes", type=int, default=16_000_000)
    tokenizer_inspect = tokenizer_subparsers.add_parser(
        "inspect", help="validate and inspect a tokenizer artifact"
    )
    tokenizer_inspect.add_argument("path", type=Path)

    data_compile = data_subparsers.add_parser(
        "compile", help="compile CSV into checksum-verified memory-mapped shards"
    )
    data_compile.add_argument("path", type=Path)
    data_compile.add_argument("--output-dir", type=Path, required=True)
    data_compile.add_argument("--tokenizer", type=Path)
    data_compile.add_argument("--max-length", type=int, default=512)
    data_compile.add_argument("--max-rows", type=int)
    data_compile.add_argument("--full-sequence-loss", action="store_true")
    data_compile.add_argument("--overwrite", action="store_true")
    data_compile.add_argument(
        "--allow-cross-split-duplicates",
        action="store_true",
        help="explicitly permit exact content duplicates across assigned splits",
    )
    _split_arguments(data_compile)
    data_verify = data_subparsers.add_parser(
        "verify", help="verify compiled dataset manifest and shard checksums"
    )
    data_verify.add_argument("path", type=Path)

    train_compiled = train_subparsers.add_parser(
        "compiled", help="train or resume from a compiled dataset"
    )
    train_compiled.add_argument("path", type=Path)
    train_compiled.add_argument("--config", type=Path, required=True)
    train_compiled.add_argument("--output-dir", type=Path, required=True)
    compiled_budget = train_compiled.add_mutually_exclusive_group(required=True)
    compiled_budget.add_argument("--max-steps", type=int)
    compiled_budget.add_argument("--max-target-tokens", type=int)
    train_compiled.add_argument("--batch-size", type=int, default=8)
    train_compiled.add_argument("--learning-rate", type=float, default=3e-4)
    train_compiled.add_argument("--weight-decay", type=float, default=0.01)
    train_compiled.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_compiled.add_argument(
        "--mixed-precision", choices=["none", "fp16", "bf16"], default="none"
    )
    train_compiled.add_argument(
        "--schedule", choices=["constant", "linear", "cosine"], default="cosine"
    )
    train_compiled.add_argument("--warmup-steps", type=int, default=0)
    train_compiled.add_argument("--checkpoint-every", type=int, default=0)
    train_compiled.add_argument("--evaluate-every", type=int, default=0)
    train_compiled.add_argument("--validation-batches", type=int, default=8)
    train_compiled.add_argument("--no-shuffle", action="store_true")
    train_compiled.add_argument("--shuffle-seed", type=int, default=1337)
    train_compiled.add_argument("--pack-sequences", action="store_true")
    train_compiled.add_argument("--resume-from", type=Path)
    train_compiled.add_argument("--allow-missing-checkpoint-digest", action="store_true")
    train_compiled.add_argument("--cpu-threads", type=int)
    train_compiled.add_argument("--activation-checkpointing", action="store_true")
    train_compiled.add_argument("--compile-model", action="store_true")
    train_compiled.add_argument("--fused-optimizer", action="store_true")
    train_compiled.add_argument("--training-stage", default="pretraining")
    train_compiled.add_argument("--parent-checkpoint", type=Path)

    infer_parser = subparsers.add_parser("infer", help="native cached generation")
    infer_subparsers = infer_parser.add_subparsers(dest="infer_command", required=True)
    infer_generate = infer_subparsers.add_parser("generate", help="generate text from a checkpoint")
    infer_generate.add_argument("--config", type=Path, required=True)
    infer_generate.add_argument("--checkpoint", type=Path)
    infer_generate.add_argument("--tokenizer", type=Path)
    infer_generate.add_argument("--prompt", required=True)
    infer_generate.add_argument(
        "--system",
        default="",
        help="optional system instruction encoded with the training chat template",
    )
    infer_generate.add_argument("--max-new-tokens", type=int, default=64)
    infer_generate.add_argument("--sample", action="store_true")
    infer_generate.add_argument("--temperature", type=float, default=1.0)
    infer_generate.add_argument("--top-k", type=int)
    infer_generate.add_argument("--top-p", type=float)
    infer_generate.add_argument("--min-p", type=float)
    infer_generate.add_argument("--repetition-penalty", type=float, default=1.0)
    infer_generate.add_argument("--seed", type=int)
    infer_generate.add_argument("--no-cache", action="store_true")
    infer_generate.add_argument(
        "--cpu-threads",
        type=int,
        help="temporary PyTorch intra-op thread count for CPU inference",
    )
    infer_generate.add_argument("--allow-missing-checkpoint-digest", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="v0.6 evaluation operations")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_context = eval_subparsers.add_parser(
        "context", help="evaluate synthetic retrieval or induction behavior"
    )
    eval_context.add_argument("--config", type=Path, required=True)
    eval_context.add_argument("--checkpoint", type=Path)
    eval_context.add_argument("--task", choices=["needle", "induction"], required=True)
    eval_context.add_argument("--cases", type=int, default=64)
    eval_context.add_argument("--distractors", type=int, default=4)
    eval_context.add_argument("--seed", type=int, default=1337)
    eval_context.add_argument(
        "--cpu-threads",
        type=int,
        help="temporary PyTorch intra-op thread count for CPU evaluation",
    )
    eval_context.add_argument("--allow-missing-checkpoint-digest", action="store_true")
    eval_language = eval_subparsers.add_parser(
        "language", help="evaluate loss and tokenization-independent bits per byte"
    )
    eval_language.add_argument("path", type=Path, help="compiled dataset directory")
    eval_language.add_argument("--config", type=Path, required=True)
    eval_language.add_argument("--checkpoint", type=Path)
    eval_language.add_argument("--split", choices=["train", "val", "test"], default="val")
    eval_language.add_argument("--batch-size", type=int, default=8)
    eval_language.add_argument("--max-batches", type=int)
    eval_language.add_argument("--pack-sequences", action="store_true")
    eval_language.add_argument("--cpu-threads", type=int)
    eval_language.add_argument("--allow-missing-checkpoint-digest", action="store_true")
    eval_generation = eval_subparsers.add_parser(
        "generation", help="measure repetition, diversity, EOS, and throughput"
    )
    eval_generation.add_argument("--config", type=Path, required=True)
    eval_generation.add_argument("--checkpoint", type=Path)
    eval_generation.add_argument("--tokenizer", type=Path)
    eval_generation.add_argument("--prompt", required=True)
    eval_generation.add_argument("--system", default="")
    eval_generation.add_argument("--samples", type=int, default=4)
    eval_generation.add_argument("--max-new-tokens", type=int, default=64)
    eval_generation.add_argument("--sample", action="store_true")
    eval_generation.add_argument("--temperature", type=float, default=1.0)
    eval_generation.add_argument("--top-k", type=int)
    eval_generation.add_argument("--top-p", type=float)
    eval_generation.add_argument("--min-p", type=float)
    eval_generation.add_argument("--repetition-penalty", type=float, default=1.0)
    eval_generation.add_argument("--seed", type=int, default=1337)
    eval_generation.add_argument("--no-cache", action="store_true")
    eval_generation.add_argument("--cpu-threads", type=int)
    eval_generation.add_argument("--allow-missing-checkpoint-digest", action="store_true")

    runs_parser = subparsers.add_parser("runs", help="local SQLite run registry")
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_init = runs_subparsers.add_parser("init", help="initialize a run registry")
    runs_init.add_argument("--database", type=Path, required=True)
    runs_list = runs_subparsers.add_parser("list", help="list registered runs")
    runs_list.add_argument("--database", type=Path, required=True)
    runs_list.add_argument("--status")
    runs_list.add_argument("--limit", type=int, default=100)
    runs_show = runs_subparsers.add_parser("show", help="show one registered run")
    runs_show.add_argument("run_id")
    runs_show.add_argument("--database", type=Path, required=True)
    runs_register = runs_subparsers.add_parser("register", help="register a report and artifacts")
    runs_register.add_argument("report", type=Path)
    runs_register.add_argument("--database", type=Path, required=True)
    runs_register.add_argument("--checkpoint", type=Path)
    runs_register.add_argument("--parent-run-id")
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


def _parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("integer list must contain positive values")
    return parsed


def _load_model(
    config_path: Path,
    checkpoint_path: Path | None,
    *,
    require_digest: bool,
) -> ReasonerCore:
    config = ReasonerConfig.from_yaml(config_path)
    model = ReasonerCore(config)
    if checkpoint_path is not None:
        load_checkpoint(
            checkpoint_path,
            model=model,
            map_location=model.device_obj,
            restore_rng=False,
            require_digest=require_digest,
        )
    return model


def _compiled_training_request(args: argparse.Namespace) -> CompiledTrainingRequest:
    return CompiledTrainingRequest(
        model_config=args.config,
        dataset_dir=args.path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_target_tokens=args.max_target_tokens,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        checkpoint_every=args.checkpoint_every,
        evaluate_every=args.evaluate_every,
        validation_batches=args.validation_batches,
        shuffle=not args.no_shuffle,
        shuffle_seed=args.shuffle_seed,
        pack_sequences=args.pack_sequences,
        resume_from=args.resume_from,
        require_checkpoint_digest=not args.allow_missing_checkpoint_digest,
        cpu_threads=args.cpu_threads,
        activation_checkpointing=args.activation_checkpointing,
        compile_model=args.compile_model,
        fused_optimizer=args.fused_optimizer,
        training_stage=args.training_stage,
        parent_checkpoint=args.parent_checkpoint,
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
    if args.command == "experiment" and args.experiment_command == "sweep-plan":
        plan = build_sweep_plan(args.path)
        plan_path = write_sweep_plan(plan, args.output_dir)
        _json_print(
            {
                "sweep_id": plan.spec.id,
                "plan_hash": plan.plan_hash,
                "trial_count": len(plan.trials),
                "plan": str(plan_path),
            }
        )
        return 0
    if args.command == "experiment" and args.experiment_command == "sweep-run":
        _json_print(
            run_sweep(
                args.path,
                output_dir=args.output_dir,
                repository=args.repository,
                maximum_trials=args.maximum_trials,
                stop_on_failure=args.stop_on_failure,
            )
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
    if args.command == "release" and args.release_command == "doctor":
        release_report = inspect_release(args.repository)
        if args.output:
            write_release_report(args.output, release_report)
        _json_print(release_report.to_dict())
        return 0 if release_report.ready else 1
    if args.command == "hardware" and args.hardware_command == "inspect":
        _json_print(runtime_capabilities())
        return 0
    if args.command == "hardware" and args.hardware_command == "calibrate":
        hardware_config = ReasonerConfig.from_yaml(args.config)
        calibration_payload = calibrate_model(
            hardware_config,
            batch_sizes=_parse_int_csv(args.batch_sizes),
            sequence_lengths=_parse_int_csv(args.sequence_lengths),
            precisions=[item.strip() for item in args.precisions.split(",") if item.strip()],
            target_memory_fraction=args.target_memory_fraction,
        ).to_dict()
        if args.output:
            atomic_write_json(args.output, calibration_payload)
        _json_print(calibration_payload)
        return 0
    if args.command == "hardware" and args.hardware_command == "estimate":
        config = ReasonerConfig.from_yaml(args.config)
        model = ReasonerCore(config)
        cache = model.estimate_memory_cost(args.seq_len, args.batch_size)["kv_cache_bytes"]
        estimate = estimate_training_memory(
            model,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            cache_bytes=cache,
            mixed_precision=args.mixed_precision,
            activation_checkpointing=args.activation_checkpointing,
        )
        _json_print(estimate.to_dict())
        return 0
    if args.command == "tokenizer" and args.tokenizer_command == "train":
        tokenizer_config = TokenizerTrainingConfig(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            normalization=args.normalization,
            max_rows=args.max_rows,
            max_training_bytes=args.max_training_bytes,
        )
        tokenizer_result = train_bpe_from_conversation_csv(
            args.path, output=args.output, config=tokenizer_config
        )
        _json_print(tokenizer_result.to_dict())
        return 0
    if args.command == "tokenizer" and args.tokenizer_command == "inspect":
        tokenizer = load_tokenizer(args.path)
        _json_print(tokenizer.manifest())
        return 0
    if args.command == "data" and args.data_command == "compile":
        tokenizer = load_tokenizer(args.tokenizer)
        compiled_manifest_path = compile_conversation_dataset(
            args.path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            split_spec=_split_spec(args),
            max_length=args.max_length,
            assistant_only_loss=not args.full_sequence_loss,
            max_rows=args.max_rows,
            overwrite=args.overwrite,
            allow_cross_split_duplicates=args.allow_cross_split_duplicates,
        )
        _json_print(
            {
                "manifest": str(compiled_manifest_path),
                "verification": verify_compiled_dataset(args.output_dir),
            }
        )
        return 0
    if args.command == "data" and args.data_command == "verify":
        verification = verify_compiled_dataset(args.path)
        _json_print(verification)
        return 0 if verification["valid"] else 1
    if args.command == "train" and args.train_command == "compiled":
        _json_print(run_compiled_training(_compiled_training_request(args)).to_dict())
        return 0
    if args.command == "infer" and args.infer_command == "generate":
        with temporary_torch_threads(args.cpu_threads) as effective_threads:
            tokenizer = load_tokenizer(args.tokenizer)
            model = _load_model(
                args.config,
                args.checkpoint,
                require_digest=not args.allow_missing_checkpoint_digest,
            )
            if model.cfg.vocab_size < tokenizer.vocab_size:
                raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
            prompt_ids = encode_chat_prompt(
                args.prompt,
                system=args.system,
                tokenizer=tokenizer,
                max_length=model.cfg.max_seq_len,
            )
            result = generate_tokens(
                model,
                torch.tensor([prompt_ids], dtype=torch.long),
                max_new_tokens=args.max_new_tokens,
                sampling=SamplingConfig(
                    do_sample=args.sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed,
                ),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.no_cache,
            )
        generated_ids = result.token_ids[0, len(prompt_ids) :].tolist()
        _json_print(
            {
                **result.to_dict(),
                "text": tokenizer.decode(generated_ids),
                "cpu_threads": effective_threads,
            }
        )
        return 0
    if args.command == "eval" and args.eval_command == "context":
        with temporary_torch_threads(args.cpu_threads) as effective_threads:
            model = _load_model(
                args.config,
                args.checkpoint,
                require_digest=not args.allow_missing_checkpoint_digest,
            )
            payload = evaluate_context_task(
                model,
                task=args.task,
                cases=args.cases,
                distractors=args.distractors,
                seed=args.seed,
            ).to_dict()
        _json_print({**payload, "cpu_threads": effective_threads})
        return 0
    if args.command == "eval" and args.eval_command == "language":
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        manifest_path = (
            args.path if args.path.name == "manifest.json" else args.path / "manifest.json"
        )
        compiled_manifest = CompiledDatasetManifest.load(manifest_path)
        summary = compiled_manifest.splits.get(args.split)
        if summary is None or summary.examples < 1:
            raise ValueError(f"compiled dataset split {args.split!r} is empty or missing")
        total_batches = math.ceil(summary.examples / args.batch_size)
        maximum_batches = args.max_batches or total_batches
        full_split = maximum_batches >= total_batches
        stream = StatefulCompiledBatchStream(
            manifest_path.parent,
            split=args.split,
            batch_size=args.batch_size,
            repeat=False,
            shuffle=False,
            pack_sequences=args.pack_sequences,
        )
        try:
            with temporary_torch_threads(args.cpu_threads) as effective_threads:
                model = _load_model(
                    args.config,
                    args.checkpoint,
                    require_digest=not args.allow_missing_checkpoint_digest,
                )
                evaluation = evaluate_language_batches(
                    model,
                    stream,
                    max_batches=maximum_batches,
                    raw_bytes=summary.target_utf8_bytes if full_split else None,
                )
        finally:
            stream.close()
        _json_print(
            {
                **evaluation.to_dict(),
                "split": args.split,
                "full_split": full_split,
                "split_raw_utf8_bytes": summary.raw_utf8_bytes,
                "split_target_utf8_bytes": summary.target_utf8_bytes,
                "cpu_threads": effective_threads,
            }
        )
        return 0
    if args.command == "eval" and args.eval_command == "generation":
        if args.samples < 1:
            raise ValueError("samples must be positive")
        with temporary_torch_threads(args.cpu_threads) as effective_threads:
            tokenizer = load_tokenizer(args.tokenizer)
            model = _load_model(
                args.config,
                args.checkpoint,
                require_digest=not args.allow_missing_checkpoint_digest,
            )
            if model.cfg.vocab_size < tokenizer.vocab_size:
                raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
            prompt_ids = encode_chat_prompt(
                args.prompt,
                system=args.system,
                tokenizer=tokenizer,
                max_length=model.cfg.max_seq_len,
            )
            prompt = torch.tensor([prompt_ids], dtype=torch.long).repeat(args.samples, 1)
            generated = generate_tokens(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                sampling=SamplingConfig(
                    do_sample=args.sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed,
                ),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.no_cache,
            )
            diagnostics = diagnose_generation(
                generated.token_ids,
                prompt_length=len(prompt_ids),
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = [tokenizer.decode(row[len(prompt_ids) :].tolist()) for row in generated.token_ids]
        _json_print(
            {
                "generation": generated.to_dict(),
                "diagnostics": diagnostics.to_dict(),
                "texts": texts,
                "cpu_threads": effective_threads,
            }
        )
        return 0
    if args.command == "runs" and args.runs_command == "init":
        RunRegistry(args.database)
        _json_print({"database": str(args.database), "initialized": True})
        return 0
    if args.command == "runs" and args.runs_command == "list":
        registry = RunRegistry(args.database)
        _json_print(
            {
                "runs": [
                    record.summary_dict()
                    for record in registry.list(status=args.status, limit=args.limit)
                ]
            }
        )
        return 0
    if args.command == "runs" and args.runs_command == "show":
        registry = RunRegistry(args.database)
        record = registry.get(args.run_id)
        _json_print({"run": record.to_dict(), "artifacts": registry.artifacts(args.run_id)})
        return 0
    if args.command == "runs" and args.runs_command == "register":
        registry = RunRegistry(args.database)
        record = register_report(
            registry,
            args.report,
            checkpoint_path=args.checkpoint,
            parent_run_id=args.parent_run_id,
        )
        _json_print(record.to_dict())
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
    if not arguments or arguments[0] in {"-h", "--help"}:
        return False
    if arguments[0] not in _NATIVE_COMMANDS:
        return True
    # Help for a native command always belongs to the stable native parser.
    if any(argument in {"-h", "--help"} for argument in arguments[1:]):
        return False
    # Preserve the historical `wai-r0 train <plan.md>` and
    # `wai-r0 report --input ...` forms while the compatibility layer exists.
    if arguments[0] == "train" and (len(arguments) < 2 or arguments[1] not in {"csv", "compiled"}):
        return True
    return arguments[0] == "report" and (
        len(arguments) < 2 or arguments[1] not in {"validate", "render"}
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not arguments:
        parser.print_help()
        return 0
    if _should_delegate(arguments):
        return _delegate_legacy(arguments)
    try:
        parsed = parser.parse_args(arguments)
        return _run_native(parsed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
