from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from wai_r0.core.reproducibility import (
    RngState,
    _fsync_directory,
    canonical_hash,
    capture_rng_state,
    restore_rng_state,
)
from wai_r0.version import __version__

CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_CHECKPOINT_FORMATS = {1, 2, 3}


@dataclass(slots=True)
class TrainingProgress:
    global_step: int = 0
    micro_step: int = 0
    consumed_tokens: int = 0
    consumed_examples: int = 0
    epoch: int = 0
    data_cursor: int = 0
    elapsed_seconds: float = 0.0
    best_metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        integer_fields = (
            "global_step",
            "micro_step",
            "consumed_tokens",
            "consumed_examples",
            "epoch",
            "data_cursor",
        )
        for name in integer_fields:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")


@dataclass(slots=True)
class RestoredCheckpoint:
    progress: TrainingProgress
    config: dict[str, Any]
    metadata: dict[str, Any]
    data_state: dict[str, Any]
    extra_state: dict[str, Any]
    lineage: dict[str, Any]
    checkpoint_version: int


def _optimizer_to_device(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _model_signature(model: nn.Module) -> str:
    return canonical_hash(
        [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in model.state_dict().items()
        ]
    )


def _write_sidecar_sha256(path: Path) -> Path:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(f"{digest.hexdigest()}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
        _fsync_directory(sidecar.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sidecar


def verify_checkpoint_digest(path: str | Path) -> bool:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        return False
    try:
        fields = sidecar.read_text(encoding="ascii").split()
    except (OSError, UnicodeError):
        return False
    if not fields or len(fields[0]) != 64:
        return False
    expected = fields[0].lower()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    progress: TrainingProgress | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    data_state: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    rng_state: RngState | None = None,
    overwrite: bool = False,
    write_digest: bool = True,
) -> Path:
    """Atomically save all state required for deterministic local resume."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    current_progress = progress or TrainingProgress()
    current_progress.validate()
    resolved_config = dict(config or {})
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "wai_r0_version": __version__,
        "model_signature": _model_signature(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": asdict(current_progress),
        "config": resolved_config,
        "config_hash": canonical_hash(resolved_config),
        "metadata": dict(metadata or {}),
        "data_state": dict(data_state or {}),
        "extra_state": dict(extra_state or {}),
        "lineage": dict(lineage or {}),
        "rng": rng_state or capture_rng_state(),
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        sidecar = destination.with_suffix(destination.suffix + ".sha256")
        if write_digest:
            _write_sidecar_sha256(destination)
        else:
            sidecar.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    # Optimizer and RNG state require Python objects. Only open trusted local files.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 2.2 compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def _progress_from_payload(payload: Mapping[str, Any]) -> TrainingProgress:
    defaults = asdict(TrainingProgress())
    defaults.update(dict(payload))
    progress = TrainingProgress(**defaults)
    progress.validate()
    return progress


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict_model: bool = True,
    require_digest: bool = False,
) -> RestoredCheckpoint:
    """Restore a trusted local checkpoint and validate its structure."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if require_digest and not verify_checkpoint_digest(source):
        raise ValueError("checkpoint digest is missing or invalid")
    payload = _safe_torch_load(source, map_location)
    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"unsupported checkpoint format: {format_version!r}; "
            f"supported={sorted(SUPPORTED_CHECKPOINT_FORMATS)}"
        )
    if not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint is missing model state")
    signature = payload.get("model_signature")
    if signature is not None and signature != _model_signature(model):
        raise ValueError("checkpoint model signature does not match the constructed model")

    model.load_state_dict(payload["model"], strict=strict_model)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if optimizer_state is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
        model_device = next(model.parameters()).device
        _optimizer_to_device(optimizer, model_device)
    if scheduler is not None:
        scheduler_state = payload.get("scheduler")
        if scheduler_state is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler_state = payload.get("scaler")
        if scaler_state is None:
            raise ValueError("checkpoint has no scaler state")
        scaler.load_state_dict(scaler_state)

    progress_payload = payload.get("progress")
    if not isinstance(progress_payload, Mapping):
        raise ValueError("checkpoint is missing training progress")
    progress = _progress_from_payload(progress_payload)

    rng_state = payload.get("rng")
    if restore_rng:
        if not isinstance(rng_state, RngState):
            raise ValueError("checkpoint is missing a valid RNG state")
        restore_rng_state(rng_state)

    config = payload.get("config") or {}
    config_hash = payload.get("config_hash")
    if config_hash is not None and config_hash != canonical_hash(config):
        raise ValueError("checkpoint config hash does not match its config payload")
    metadata = payload.get("metadata") or {}
    data_state = payload.get("data_state") or payload.get("extra_state", {}).get("data_state", {})
    extra_state = payload.get("extra_state") or {}
    lineage = payload.get("lineage") or {}
    for name, value in (
        ("config", config),
        ("metadata", metadata),
        ("data_state", data_state),
        ("extra_state", extra_state),
        ("lineage", lineage),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint {name} must be a mapping")
    return RestoredCheckpoint(
        progress=progress,
        config=config,
        metadata=metadata,
        data_state=data_state,
        extra_state=extra_state,
        lineage=lineage,
        checkpoint_version=int(format_version),
    )


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata. Only inspect trusted local files."""

    source = Path(path)
    payload = _safe_torch_load(source, "cpu")
    model_state = payload.get("model")
    tensor_count = len(model_state) if isinstance(model_state, dict) else 0
    return {
        "format_version": payload.get("format_version"),
        "wai_r0_version": payload.get("wai_r0_version"),
        "model_signature": payload.get("model_signature"),
        "progress": payload.get("progress"),
        "config": payload.get("config"),
        "config_hash": payload.get("config_hash"),
        "metadata": payload.get("metadata"),
        "data_state_keys": sorted((payload.get("data_state") or {}).keys()),
        "extra_state_keys": sorted((payload.get("extra_state") or {}).keys()),
        "lineage": payload.get("lineage") or {},
        "model_tensor_count": tensor_count,
        "has_optimizer": payload.get("optimizer") is not None,
        "has_scheduler": payload.get("scheduler") is not None,
        "has_scaler": payload.get("scaler") is not None,
        "digest_present": source.with_suffix(source.suffix + ".sha256").is_file(),
        "digest_valid": verify_checkpoint_digest(source),
    }
