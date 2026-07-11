# WAI-R0 v0.5 Evaluation Protocol

## Evidence classes

Every metric belongs to one of:

- numerical diagnostic;
- architecture prior;
- learned algorithmic;
- learned language;
- symbolic solver;
- hybrid system;
- systems performance.

Results from different classes are not interchangeable. Symbolic success is not neural reasoning, and static memory estimates are not measured allocation.

## Candidate/control discipline

A manifest must name a matching rule:

- parameter matched;
- active-parameter matched;
- FLOP matched;
- wall-clock matched;
- memory matched;
- token matched.

The current executable runner enforces parameter and active-parameter equality directly and supports token-matched experiments. Unsupported matching rules fail the correctness gate instead of being approximated silently.

## Algorithmic battery

The deterministic generated battery includes:

- copy;
- reverse;
- parity;
- modular addition;
- sorting;
- selective copy;
- associative recall;
- bracket balance;
- finite-state parity.

Each task uses separate training, in-distribution evaluation, and held-out-length evaluation seed ranges. Metrics include token accuracy, sequence exact match, and loss. Held-out-length performance is the preferred screening signal for recurrence experiments.

Generated probes are not a substitute for natural-language evaluation. Generator source and parameters are part of provenance and must be frozen before final evaluation.

## Statistical output

Paired comparisons report:

- candidate/control values by seed;
- oriented differences;
- mean, median, standard deviation, min, and max;
- bootstrap confidence interval;
- standardized effect size where defined;
- wins, losses, and ties;
- successful and failed seed counts.

Failed seeds remain visible. A run cannot meet the successful-seed gate by discarding inconvenient failures.

## Systems profiling

Profiling separately records:

- parameter bytes;
- theoretical cache bytes;
- actual cache tensor allocation;
- prefill median and dispersion;
- decode median and dispersion;
- tokens per second;
- CPU RSS where available;
- CUDA peak allocation where available;
- hardware/software identity.

Warmup and measured run counts are declared. CPU timing is screening evidence; final GPU conclusions require the target GPU.

## Mandatory gates

The decision path is non-compensatory:

1. correctness;
2. successful seed count;
3. contamination/data integrity;
4. matching-rule validity;
5. primary metric threshold;
6. robustness axes;
7. reproducibility/provenance.

`keep` is impossible if a mandatory gate fails. `re_test` must identify the unresolved condition rather than becoming an indefinite holding label.

## Final-evaluation rule

A manifest marked `final_evaluation: true` is immutable. Any architecture, threshold, preprocessing, generator, or metric change after reading that result creates a new manifest and a new final-evaluation generation.
