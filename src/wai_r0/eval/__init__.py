"""Evaluation utilities for WAI-R0."""

from wai_r0.eval.holdout import GeneratedTaskSpec, generate_holdout_tasks, write_holdout_tasks
from wai_r0.eval.leakage_guard import LeakageGuard, LeakageFinding, hash_task_file
from wai_r0.eval.scorecard import R0Scorecard, ScoreBand, score_from_report

__all__ = [
    "GeneratedTaskSpec",
    "LeakageFinding",
    "LeakageGuard",
    "R0Scorecard",
    "ScoreBand",
    "generate_holdout_tasks",
    "hash_task_file",
    "score_from_report",
    "write_holdout_tasks",
]
