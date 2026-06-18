# WAI-R0 v0.4.2 — main.py workbench fallback fix

This patch fixes `python main.py` failing in environments where Tkinter cannot open a graphical display.

## Fixed

- `python main.py` no longer crashes with `_tkinter.TclError: couldn't connect to display`.
- `python main.py gui` uses the same fallback behavior.
- Headless/non-interactive environments now print direct command examples instead of hanging.

## Added

- Terminal workbench fallback for SSH, WSL/headless shells, CI containers, and systems without `python3-tk`.
- Shared subprocess environment helper for GUI/TUI commands.
- Tests for non-interactive fallback behavior.

## Notes

The Tkinter GUI is still the preferred local interface when a desktop display is available. The terminal fallback exists so the repo remains usable everywhere.
