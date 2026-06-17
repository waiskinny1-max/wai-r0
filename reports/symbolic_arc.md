# symbolic_arc

## Metadata

- Date: 2026-06-17T15:02:41.857456+00:00
- Git commit: unavailable
- Device: cpu
- Dtype: n/a
- Seed: 0
- Result type: zero-training symbolic solver result

## Summary

Explicit symbolic program search. Not neural reasoning.

## Raw metrics

```json
{
  "pass_at_1": 1.0,
  "tasks": [
    {
      "candidates_tested": 3,
      "elapsed_s": 0.00015715600000021368,
      "failure": null,
      "predictions": [
        [
          [
            8,
            0,
            7
          ]
        ]
      ],
      "program": "rotate180",
      "solved": true,
      "task_id": "mirror_y_demo"
    },
    {
      "candidates_tested": 2,
      "elapsed_s": 0.00012694899999132758,
      "failure": null,
      "predictions": [
        [
          [
            9,
            7
          ],
          [
            1,
            8
          ]
        ]
      ],
      "program": "rotate90",
      "solved": true,
      "task_id": "rotate90_demo"
    }
  ]
}
```

## Limitations

- Small DSL and demo tasks.
- Avoid public-eval leakage.

## Recommendation

TINY-TRAIN ONLY — promising but unproven.
