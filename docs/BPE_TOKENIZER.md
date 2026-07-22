# Deterministic BPE Tokenizer

WAI-R0 uses a byte-level base alphabet, then learns ranked adjacent-symbol merges. Byte fallback means every UTF-8 input remains representable.

Training is deterministic for a fixed normalized corpus traversal, vocabulary target, corpus-byte ceiling, and special-token map. The artifact records the corpus hash and actual merge count. The optimized encoder is checked against a reference merge algorithm.

Role tokens are fixed outside the byte range and are part of the tokenizer identity. The same chat-template encoder is used by dataset compilation and inference.

Use bits per raw byte for cross-tokenizer comparison; token perplexity alone can favor a tokenizer merely because it defines larger units.
