# WAI-R0 v0.5 Tokenization

## Byte-chat control

The built-in tokenizer is a deterministic byte-level control. It reserves explicit tokens for padding, beginning/end, and chat boundaries, then maps UTF-8 bytes into a fixed vocabulary.

Advantages:

- no external tokenizer dependency;
- stable behavior across datasets;
- no unknown token;
- manifest can be reproduced from code and special-token mapping.

Limitations:

- longer sequences than a trained subword tokenizer;
- poorer compute efficiency;
- byte fragmentation is not a claim about natural language representation quality.

## Target construction

The encoded sequence contains role boundaries and content. By default:

- system and user spans are context only;
- assistant role marker is context only;
- assistant content and terminal token are supervised;
- padding and truncated non-target spans use `-100`.

Truncation preserves assistant supervision when possible. Examples with no remaining target tokens are invalid for training.

## Manifest

Tokenizer manifests include tokenizer type/version, vocabulary size, special tokens, encoding policy, and a canonical manifest hash. The hash is stored in stream state; an exact resume with a different tokenizer fails.
