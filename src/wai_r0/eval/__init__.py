from importlib import import_module
from typing import Any

from wai_r0.eval.algorithmic import (
    DEFAULT_VOCAB_SIZE,
    AlgorithmicBatchStream,
    AlgorithmicEvaluation,
    AlgorithmicExample,
    AlgorithmicTask,
    collate_algorithmic,
    encode_algorithmic_example,
    fixed_algorithmic_examples,
    generate_algorithmic_example,
)
from wai_r0.eval.gates import decide_non_compensatory
from wai_r0.eval.metrics import SequenceMetrics, evaluate_sequence_batches

__all__ = [
    "DEFAULT_VOCAB_SIZE",
    "AlgorithmicBatchStream",
    "AlgorithmicEvaluation",
    "AlgorithmicExample",
    "AlgorithmicTask",
    "GeneratedTaskSpec",
    "LeakageFinding",
    "LeakageGuard",
    "PriorProbe",
    "R0Scorecard",
    "ScoreBand",
    "SequenceMetrics",
    "collate_algorithmic",
    "decide_non_compensatory",
    "encode_algorithmic_example",
    "evaluate_sequence_batches",
    "fixed_algorithmic_examples",
    "generate_algorithmic_example",
    "generate_holdout_tasks",
    "hash_task_file",
    "run_prior_diagnostics",
    "score_from_report",
    "write_holdout_tasks",
]

_LEGACY_EXPORTS = {
    "GeneratedTaskSpec": ("wai_r0.eval.holdout", "GeneratedTaskSpec"),
    "generate_holdout_tasks": ("wai_r0.eval.holdout", "generate_holdout_tasks"),
    "write_holdout_tasks": ("wai_r0.eval.holdout", "write_holdout_tasks"),
    "LeakageGuard": ("wai_r0.eval.leakage_guard", "LeakageGuard"),
    "LeakageFinding": ("wai_r0.eval.leakage_guard", "LeakageFinding"),
    "hash_task_file": ("wai_r0.eval.leakage_guard", "hash_task_file"),
    "PriorProbe": ("wai_r0.eval.prior_diagnostics", "PriorProbe"),
    "run_prior_diagnostics": ("wai_r0.eval.prior_diagnostics", "run_prior_diagnostics"),
    "R0Scorecard": ("wai_r0.eval.scorecard", "R0Scorecard"),
    "ScoreBand": ("wai_r0.eval.scorecard", "ScoreBand"),
    "score_from_report": ("wai_r0.eval.scorecard", "score_from_report"),
}


def __getattr__(name: str) -> Any:
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)
