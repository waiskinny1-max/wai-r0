# Security Policy

## Supported line

Security fixes target the current `0.5.x` line.

## Trusted-artifact boundary

PyTorch checkpoints use Python object deserialization to restore optimizer and RNG state. **Never load a checkpoint from an untrusted source.** SHA-256 sidecars detect corruption/substitution but do not establish publisher identity.

Dataset CSV and YAML inputs are parsed as data. Markdown training plans remain compatibility inputs and must not execute embedded shell or Python instructions.

## Reporting a vulnerability

Report vulnerabilities privately to the repository owner before opening a public issue. Include affected version, reproduction steps, impact, and the smallest safe proof of concept. Do not include secrets or personal data.
