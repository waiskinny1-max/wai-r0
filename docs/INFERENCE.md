# Native Inference

Native inference loads the exact model configuration, checkpoint, tokenizer, and chat-template semantics used in training. Generation supports greedy, temperature, top-k, top-p, min-p, repetition penalty, EOS, stop sequences, deterministic seeds, and KV-cache reuse.

```bash
wai-r0 infer generate --tokenizer tokenizer.json --config model.yaml --checkpoint final.pt --prompt "Hello"
```

Sampling validation covers probability normalization, fixed-seed reproduction, greedy argmax behavior, and cached/full-context agreement. Output quality is evaluated separately from mechanical generation correctness.
