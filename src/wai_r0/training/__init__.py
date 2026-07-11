from importlib import import_module
from typing import Any

from wai_r0.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    RestoredCheckpoint,
    TrainingProgress,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint_digest,
)
from wai_r0.training.engine import (
    Trainer,
    TrainerConfig,
    TrainingMetrics,
    TrainingOOMError,
    TrainingResult,
)
from wai_r0.training.losses import causal_language_model_loss
from wai_r0.training.optimizer import build_adamw, parameter_groups_for_weight_decay
from wai_r0.training.schedules import build_scheduler, learning_rate_multiplier

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "RestoredCheckpoint",
    "TinyProbeResult",
    "Trainer",
    "TrainerConfig",
    "TrainingMetrics",
    "TrainingOOMError",
    "TrainingProgress",
    "TrainingResult",
    "build_adamw",
    "build_scheduler",
    "causal_language_model_loss",
    "inspect_checkpoint",
    "learning_rate_multiplier",
    "load_checkpoint",
    "parameter_groups_for_weight_decay",
    "run_tiny_probe",
    "save_checkpoint",
    "verify_checkpoint_digest",
]

_LEGACY_EXPORTS = {
    "TinyProbeResult": ("wai_r0.training.probes", "TinyProbeResult"),
    "run_tiny_probe": ("wai_r0.training.probes", "run_tiny_probe"),
}


def __getattr__(name: str) -> Any:
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)
