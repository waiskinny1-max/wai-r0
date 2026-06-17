# ablation

## Metadata

- Date: 2026-06-17T15:03:23.319469+00:00
- Git commit: unavailable
- Device: cpu
- Dtype: float32
- Seed: 1337
- Result type: mixed architecture diagnostic

## Summary

Ablation over attention, MoE, and recurrence. Diagnostic only.

## Raw metrics

```json
{
  "stable_count": 7,
  "total": 7,
  "variants": [
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.17428527763960036,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.730040550231934,
              5.750452041625977
            ],
            "attention": [
              {
                "attention_entropy": 1.2812066078186035,
                "compression_ratio": null,
                "kv_cache_bytes": 2048
              }
            ],
            "moe": []
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": null
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A0"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.16691296839747916,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.645754814147949,
              5.748652935028076
            ],
            "attention": [
              {
                "attention_entropy": 1.2932419776916504,
                "compression_ratio": null,
                "kv_cache_bytes": 1024
              }
            ],
            "moe": []
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": null
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A1"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.12805068282371698,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.950651168823242,
              5.976316928863525
            ],
            "attention": [
              {
                "attention_entropy": 1.3172687292099,
                "compression_ratio": 0.375,
                "kv_cache_bytes": 384
              }
            ],
            "moe": []
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": null
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A2"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.15844813827948276,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.510364055633545,
              5.522285461425781
            ],
            "attention": [
              {
                "attention_entropy": 1.29697585105896,
                "compression_ratio": null,
                "kv_cache_bytes": 2048
              }
            ],
            "moe": [
              {
                "collapse_warning": false,
                "load_fraction": [
                  0.125,
                  0.75,
                  0.125,
                  0.0
                ],
                "router_entropy": 1.2656155824661255
              }
            ]
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": null
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A3"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.34112529593209423,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.195269584655762,
              5.146566867828369
            ],
            "attention": [
              {
                "attention_entropy": 1.2726836204528809,
                "compression_ratio": null,
                "kv_cache_bytes": 2048
              }
            ],
            "moe": []
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": {
            "depth": 2,
            "drift_by_step": [
              0.6340705156326294,
              0.615553617477417
            ],
            "halted_early": false,
            "norm_by_step": [
              5.653684616088867,
              5.718714237213135
            ]
          }
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A4"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.23483389828470536,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.390571594238281,
              5.462632179260254
            ],
            "attention": [
              {
                "attention_entropy": 1.3152687549591064,
                "compression_ratio": 0.375,
                "kv_cache_bytes": 384
              }
            ],
            "moe": []
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": {
            "depth": 2,
            "drift_by_step": [
              0.5502616763114929,
              0.5264338254928589
            ],
            "halted_early": false,
            "norm_by_step": [
              5.650327682495117,
              5.692564010620117
            ]
          }
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A5"
    },
    {
      "metrics": {
        "finite_gradients": true,
        "finite_logits": true,
        "grad_norm_mean": 0.18421282424016866,
        "inspection": {
          "diagnostics": {
            "activation_norms": [
              5.819457054138184,
              5.857970237731934
            ],
            "attention": [
              {
                "attention_entropy": 1.3113436698913574,
                "compression_ratio": 0.375,
                "kv_cache_bytes": 384
              }
            ],
            "moe": [
              {
                "collapse_warning": false,
                "load_fraction": [
                  0.375,
                  0.25,
                  0.25,
                  0.125
                ],
                "router_entropy": 1.2503588199615479
              }
            ]
          },
          "finite": true,
          "logits_shape": [
            1,
            8,
            64
          ],
          "recurrent": {
            "depth": 2,
            "drift_by_step": [
              0.519006073474884,
              0.5156463980674744
            ],
            "halted_early": false,
            "norm_by_step": [
              5.698239326477051,
              5.782654762268066
            ]
          }
        },
        "r0_stability_score": 0.9999999999999999
      },
      "recommendation": "SCALE CAREFULLY \u2014 worth serious pretraining investigation.",
      "variant": "A6"
    }
  ]
}
```

## Limitations

- Uses zero-neural suite only in v0.1.
- Add memory and tiny-training subruns before scale decisions.

## Recommendation

TINY-TRAIN ONLY — promising but unproven.
