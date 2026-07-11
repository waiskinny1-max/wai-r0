# Contributing to WAI-R0

## Change discipline

Architecture changes must include:

1. a falsifiable hypothesis;
2. a candidate and control;
3. a declared matching rule;
4. correctness/invariant tests;
5. an experiment manifest with thresholds fixed before results;
6. an honest limitations section.

Do not combine unrelated architecture mechanisms in one comparison unless the experiment is explicitly factorial.

## Engineering checks

```bash
python scripts/check_v05_quality.py
pytest
pytest --cov=wai_r0 --cov-report=term-missing
python -m build
```

Keep public APIs typed, avoid mutable side channels in forward paths, fail closed on corrupted provenance, and preserve compatibility shims only when they do not constrain the new architecture.

## Claims

Use evidence-class terminology. Do not describe random-weight diagnostics as reasoning, symbolic results as neural capability, estimates as measurements, or single-seed gains as established improvements.
