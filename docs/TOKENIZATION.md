# Tokenization

## Byte control

The byte tokenizer is deterministic, has complete UTF-8 coverage, and remains the zero-dependency control. It is inefficient for language training because common substrings require many tokens.

## Deterministic byte-level BPE

The v0.6 BPE tokenizer starts from byte symbols and learns ranked merges from a deterministic corpus traversal. It retains byte fallback, fixed role tokens, optional NFKC normalization, and stable serialization.

The artifact includes:

- tokenizer type and format version;
- vocabulary and ranked merges;
- normalization policy;
- fixed special tokens;
- requested/actual vocabulary size;
- corpus hash and byte count;
- artifact/manifest hash.

The optimized encoder is tested against a simple reference implementation of ranked BPE semantics.

## Shared chat template

Training and inference encode the same sequence of BOS, role markers, role content, assistant boundary, supervised assistant tokens, and EOS. System/user spans and role markers are context; assistant content and terminal EOS are targets. Truncation must preserve target supervision or reject the example.

Any tokenizer, normalization, role-token, or template change changes artifact identity and invalidates exact checkpoint resume.

## Cross-tokenizer evaluation

Token loss and perplexity are tokenizer dependent. Comparisons across tokenizers should include bits per raw UTF-8 byte, throughput, sequence lengths, and model/data budgets.
