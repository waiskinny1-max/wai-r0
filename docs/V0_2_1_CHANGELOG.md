# WAI-R0 v0.2.1 changelog

## Added

- `python main.py -train training.md` compatibility wrapper.
- `wai-r0 train training.md` explicit CLI subcommand.
- Markdown training-plan parser for tiny-training probes.
- Example plan at `examples/training.md`.
- Tests for Markdown parsing, unsafe-key rejection, legacy `-train` aliasing, and report writing.

## Boundary

The Markdown entrypoint is intentionally declarative. It does not execute shell,
Python, or arbitrary Markdown instructions. In v0.2.1 it only dispatches to the
existing tiny-training architecture probe.
