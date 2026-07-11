# Experiment Manifest

An experiment manifest freezes the comparison and decision rule before results are inspected.

```yaml
id: recurrent-reverse-ood-001
kind: algorithmic
hypothesis: Fixed latent refinement improves held-out-length reverse accuracy.
candidate: recurrent_steps_4
control: recurrent_steps_0
matching_rule: parameter_matched
evidence_class: learned_algorithmic
datasets: [reverse]
seeds: [1337, 2026, 4096]
primary_metric: ood_token_accuracy
thresholds:
  keep: 0.05
  kill: -0.02
  higher_is_better: true
secondary_metrics: [ood_exact_match, id_token_accuracy]
failure_metrics: [non_finite_loss, parameter_mismatch]
maximum_budget:
  wall_clock_seconds: 1800
  optimizer_steps_per_variant: 40
known_confounds:
  - Tiny generated-task evidence may not transfer to language models.
minimum_successful_seeds: 3
tie_tolerance: 0.000001
final_evaluation: false
execution:
  candidate_config: ../model/nano_recurrent.yaml
  control_config: ../model/nano_recurrent.yaml
  task: reverse
  train_lengths: [4, 6]
  id_length: 6
  ood_length: 12
  train_steps: 40
  candidate_model_mode: think
  candidate_recurrent_steps: 4
  control_model_mode: fast
```

## Kinds

- `profile` — systems-performance comparison using measured model profiles.
- `algorithmic` — paired tiny training/evaluation on generated task families.
- `external_metrics` — gate/statistics evaluation of externally supplied paired values.

## Matching rules

One of these values is mandatory:

- `parameter_matched`;
- `active_parameter_matched`;
- `flop_matched`;
- `wall_clock_matched`;
- `memory_matched`;
- `token_matched`.

The executable runner directly validates parameter and active-parameter equality. Token-matched runs share declared training budgets. FLOP/wall-clock/memory matching must not be claimed through this runner until an executor implements and verifies that rule; unsupported rules fail correctness.

## Evidence classes

- `numerical_diagnostic`;
- `architecture_prior`;
- `learned_algorithmic`;
- `learned_language`;
- `symbolic_solver`;
- `hybrid_system`;
- `systems_performance`.

## Thresholds

For higher-is-better metrics:

- oriented mean difference at or above `keep` → eligible for `keep`;
- at or below `kill` → `kill`;
- between the thresholds → `re_test`.

Eligibility is not the final decision. Correctness, seed-count, matching, contamination, robustness, and provenance gates are non-compensatory.

## Budgets

`wall_clock_seconds` is checked after execution and reported as a correctness failure when exceeded. Algorithmic manifests may additionally declare `optimizer_steps_per_variant`; the runner rejects an execution plan whose `train_steps` exceeds it before training begins.

## Final evaluation

Set `final_evaluation: true` only for a frozen manifest. Reading a final result and then changing architecture, data generator, metric, threshold, or preprocessing requires a new manifest and evaluation generation.
