# Evaluation Protocol

## Evidence classes

Metrics are labeled as numerical diagnostic, architecture prior, learned algorithmic, learned language, systems performance, symbolic solver, or hybrid system evidence.

## Mandatory controls

Architecture comparisons declare parameter, active-parameter, FLOP, wall-clock, token, or memory matching. Correctness, provenance, seed count, and contamination gates are non-compensatory.

## v0.6 suites

- **Algorithmic:** generated train/held-out tasks, exact and token metrics, length and composition shift.
- **Language:** held-out NLL, perplexity, bits per target token, and bits per raw byte.
- **Context:** deterministic needle retrieval and induction-pattern diagnostics.
- **Generation:** repetition, unique-token ratio, longest repeated run, EOS and output-length behavior.
- **Systems:** prefill, decode, throughput, cache bytes, allocator peaks, and backend inventory.

## Statistics

Multi-seed reports include individual rows, failures, paired differences, mean/median/standard deviation, bootstrap confidence intervals, effect size, and wins/losses/ties. A confidence interval crossing zero is inconclusive unless the manifest preregisters a different decision rule.

## Contamination and memorization

Final releases should freeze evaluation generators, scan exact/near overlap, maintain a public-benchmark exposure ledger, and use memorization canaries. Results used to alter architecture are development evidence and cannot remain the untouched final test.

## Claim policy

A loss decrease establishes learning on the measured distribution. Exact generated answers do not by themselves establish generalization. CPU performance does not establish GPU ordering. Hybrid or symbolic success is never reported as pure neural reasoning.
