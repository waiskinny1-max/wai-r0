# memory

## Metadata

- Date: 2026-06-17T15:02:33.011572+00:00
- Git commit: unavailable
- Device: cpu
- Dtype: float32
- Seed: 1337
- Result type: architecture-prior diagnostic

## Summary

KV-cache memory estimate. Not a reasoning benchmark.

## Raw metrics

```json
{
  "average_candidate_over_baseline": 0.1875,
  "rows": [
    {
      "baseline_bytes": 4096,
      "candidate_bytes": 768,
      "candidate_over_baseline": 0.1875,
      "saved_bytes": 3328,
      "seq_len": 16
    },
    {
      "baseline_bytes": 8192,
      "candidate_bytes": 1536,
      "candidate_over_baseline": 0.1875,
      "saved_bytes": 6656,
      "seq_len": 32
    },
    {
      "baseline_bytes": 16384,
      "candidate_bytes": 3072,
      "candidate_over_baseline": 0.1875,
      "saved_bytes": 13312,
      "seq_len": 64
    }
  ]
}
```

## Limitations

- Static estimate; profile target hardware.
- MLA-lite is not DeepSeek MLA.

## Recommendation

TINY-TRAIN ONLY — promising but unproven.
