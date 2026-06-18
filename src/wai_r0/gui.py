from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any


@dataclass(frozen=True)
class CSVTrainGuiOptions:
    csv_path: str
    config: str = "configs/model/nano.yaml"
    text_column: str = ""
    target_column: str = ""
    steps: int = 500
    batch_size: int = 8
    seq_len: int = 128
    max_rows: int | None = 500_000
    eval_rows: int = 256
    eval_interval: int = 25
    baseline_rows: int = 2048
    lr: float = 3e-4
    checkpoint: str = "reports/csv_probe.pt"
    log: str = "reports/csv_probe_train.jsonl"
    output: str = "reports/csv_language_readiness"


def _nonempty(value: str | None) -> bool:
    return value is not None and str(value).strip() != ""


def command_entrypoint(cwd: str | Path | None = None) -> list[str]:
    """Return a source-tree-safe Python entrypoint.

    Prefer ``main.py`` when running from the repository root because the user
    explicitly wants ``python main.py`` as the normal launch surface. Fall back
    to ``-m wai_r0`` when the package is installed elsewhere.
    """

    root = Path(cwd or os.getcwd())
    if (root / "main.py").exists():
        return [sys.executable, "-u", "main.py"]
    return [sys.executable, "-u", "-m", "wai_r0"]


def build_train_csv_command(options: CSVTrainGuiOptions, cwd: str | Path | None = None) -> list[str]:
    if not _nonempty(options.csv_path):
        raise ValueError("csv_path is required")
    cmd = [
        *command_entrypoint(cwd),
        "train-csv",
        "--csv",
        options.csv_path,
        "--config",
        options.config,
        "--steps",
        str(options.steps),
        "--batch-size",
        str(options.batch_size),
        "--seq-len",
        str(options.seq_len),
        "--eval-rows",
        str(options.eval_rows),
        "--eval-interval",
        str(options.eval_interval),
        "--baseline-rows",
        str(options.baseline_rows),
        "--lr",
        str(options.lr),
        "--checkpoint",
        options.checkpoint,
        "--log",
        options.log,
        "--output",
        options.output,
        "--stream",
    ]
    if _nonempty(options.text_column):
        cmd.extend(["--text-column", options.text_column.strip()])
    if _nonempty(options.target_column):
        cmd.extend(["--target-column", options.target_column.strip()])
    if options.max_rows is not None:
        cmd.extend(["--max-rows", str(options.max_rows)])
    return cmd


def build_audit_csv_command(
    csv_path: str,
    text_column: str = "",
    target_column: str = "",
    max_rows: int | None = 500_000,
    output: str = "reports/gui_csv_audit.json",
    cwd: str | Path | None = None,
) -> list[str]:
    if not _nonempty(csv_path):
        raise ValueError("csv_path is required")
    cmd = [*command_entrypoint(cwd), "audit-csv", "--csv", csv_path, "--output", output]
    if _nonempty(text_column):
        cmd.extend(["--text-column", text_column.strip()])
    if _nonempty(target_column):
        cmd.extend(["--target-column", target_column.strip()])
    if max_rows is not None:
        cmd.extend(["--max-rows", str(max_rows)])
    return cmd


def build_sample_csv_command(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 120,
    cwd: str | Path | None = None,
) -> list[str]:
    if not _nonempty(checkpoint):
        raise ValueError("checkpoint is required")
    return [
        *command_entrypoint(cwd),
        "sample-csv",
        "--checkpoint",
        checkpoint,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--stream",
    ]


def build_benchmark_command(kind: str, config: str = "configs/model/nano.yaml", cwd: str | Path | None = None) -> list[str]:
    base = command_entrypoint(cwd)
    if kind == "zero":
        return [*base, "zero-neural", "--config", config]
    if kind == "prior":
        return [*base, "architecture-priors", "--config", config, "--seq-len", "16", "--recurrent-depths", "1,2,4"]
    if kind == "suite":
        return [*base, "suite", "--config", config, "--suite", "configs/benchmark/suite.yaml"]
    if kind == "symbolic":
        return [*base, "symbolic-arc", "--tasks", "examples/tasks", "--budget", "3s"]
    raise ValueError(f"unknown benchmark kind: {kind}")


def parse_training_event(line: str) -> dict[str, Any] | None:
    prefix = "[train] "
    if not line.startswith(prefix):
        return None
    try:
        payload = json.loads(line[len(prefix) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class ProcessHandle:
        def __init__(self, app: "WaiR0Workbench") -> None:
            self.app = app
            self.process: subprocess.Popen[str] | None = None
            self.reader: threading.Thread | None = None
            self.started_at = 0.0

        def running(self) -> bool:
            return self.process is not None and self.process.poll() is None

        def start(self, cmd: list[str], label: str) -> None:
            if self.running():
                messagebox.showwarning("WAI-R0", "A process is already running. Stop it first.")
                return
            self.app.clear_progress()
            self.app.append_console(f"\n$ {' '.join(cmd)}\n", tag="cmd")
            env = os.environ.copy()
            src = str(Path.cwd() / "src")
            env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            self.started_at = time.perf_counter()
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=0,
                )
            except OSError as exc:
                self.app.append_console(f"failed to start: {exc}\n", tag="error")
                self.process = None
                return
            self.app.set_status(f"running: {label}")
            self.reader = threading.Thread(target=self._read_stdout, daemon=True)
            self.reader.start()

        def _read_stdout(self) -> None:
            assert self.process is not None
            assert self.process.stdout is not None
            while True:
                chunk = self.process.stdout.read(1)
                if chunk == "" and self.process.poll() is not None:
                    break
                if chunk:
                    self.app.queue.put(("output", chunk))
            code = self.process.wait()
            elapsed = time.perf_counter() - self.started_at
            self.app.queue.put(("done", code, elapsed))

        def stop(self) -> None:
            if not self.running():
                return
            assert self.process is not None
            self.app.append_console("\n[gui] terminate requested\n", tag="warn")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.app.append_console("[gui] process did not exit; killing\n", tag="warn")
                self.process.kill()

    class WaiR0Workbench:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
            self.process = ProcessHandle(self)
            self.line_buffer = ""
            self.expected_steps = tk.IntVar(value=500)
            self._build()
            self.root.after(80, self._drain)

        def _build(self) -> None:
            self.root.title("WAI-R0 local workbench")
            self.root.geometry("1180x760")
            self.root.minsize(980, 640)

            self.status = tk.StringVar(value="idle")
            top = ttk.Frame(self.root, padding=(10, 8))
            top.pack(fill="x")
            ttk.Label(top, text="WAI-R0", font=("TkDefaultFont", 15, "bold")).pack(side="left")
            ttk.Label(top, textvariable=self.status).pack(side="left", padx=14)
            ttk.Button(top, text="Stop", command=self.process.stop).pack(side="right")

            panes = ttk.PanedWindow(self.root, orient="vertical")
            panes.pack(fill="both", expand=True)

            notebook = ttk.Notebook(panes)
            panes.add(notebook, weight=3)

            train_tab = ttk.Frame(notebook, padding=12)
            sample_tab = ttk.Frame(notebook, padding=12)
            bench_tab = ttk.Frame(notebook, padding=12)
            notebook.add(train_tab, text="Train CSV")
            notebook.add(sample_tab, text="Talk / sample")
            notebook.add(bench_tab, text="Benchmarks")

            self._build_train_tab(train_tab)
            self._build_sample_tab(sample_tab)
            self._build_bench_tab(bench_tab)

            console_frame = ttk.Frame(panes, padding=(10, 6))
            panes.add(console_frame, weight=2)
            header = ttk.Frame(console_frame)
            header.pack(fill="x")
            ttk.Label(header, text="Live output").pack(side="left")
            ttk.Button(header, text="Clear", command=self._clear_console).pack(side="right")
            self.console = tk.Text(console_frame, height=14, wrap="word", undo=False)
            self.console.pack(fill="both", expand=True, side="left")
            scroll = ttk.Scrollbar(console_frame, command=self.console.yview)
            scroll.pack(fill="y", side="right")
            self.console.configure(yscrollcommand=scroll.set)
            self.console.tag_configure("cmd", foreground="#255a9b")
            self.console.tag_configure("error", foreground="#a32222")
            self.console.tag_configure("warn", foreground="#9a6a00")

        def _row(self, parent: Any, label: str, variable: tk.Variable, row: int, browse: str | None = None) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(parent, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 4))
            if browse:
                ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, browse)).grid(row=row, column=2, sticky="ew")

        def _browse(self, variable: tk.Variable, kind: str) -> None:
            if kind == "csv":
                path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*")])
            elif kind == "checkpoint":
                path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")])
            elif kind == "config":
                path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*")])
            else:
                path = filedialog.askopenfilename()
            if path:
                try:
                    variable.set(str(Path(path).relative_to(Path.cwd())))
                except ValueError:
                    variable.set(path)

        def _build_train_tab(self, tab: ttk.Frame) -> None:
            tab.columnconfigure(1, weight=1)
            self.csv_path = tk.StringVar(value="training/basic_lang_500k.csv")
            self.config_path = tk.StringVar(value="configs/model/nano.yaml")
            self.text_column = tk.StringVar(value="")
            self.target_column = tk.StringVar(value="")
            self.max_rows = tk.StringVar(value="500000")
            self.steps = tk.IntVar(value=500)
            self.batch_size = tk.IntVar(value=8)
            self.seq_len = tk.IntVar(value=128)
            self.eval_rows = tk.IntVar(value=256)
            self.eval_interval = tk.IntVar(value=25)
            self.baseline_rows = tk.IntVar(value=2048)
            self.lr = tk.StringVar(value="0.0003")
            self.checkpoint = tk.StringVar(value="reports/csv_probe.pt")
            self.train_log = tk.StringVar(value="reports/csv_probe_train.jsonl")
            self.train_output = tk.StringVar(value="reports/csv_language_readiness")

            self._row(tab, "CSV", self.csv_path, 0, "csv")
            self._row(tab, "Config", self.config_path, 1, "config")
            self._row(tab, "Text column (blank=auto)", self.text_column, 2)
            self._row(tab, "Target column (blank=auto)", self.target_column, 3)
            self._row(tab, "Max rows", self.max_rows, 4)
            self._row(tab, "Checkpoint", self.checkpoint, 5, "checkpoint")
            self._row(tab, "JSONL log", self.train_log, 6)
            self._row(tab, "Report stem", self.train_output, 7)

            numeric = ttk.LabelFrame(tab, text="Training budget", padding=8)
            numeric.grid(row=8, column=0, columnspan=3, sticky="ew", pady=8)
            for i in range(6):
                numeric.columnconfigure(i, weight=1)
            fields = [
                ("steps", self.steps),
                ("batch", self.batch_size),
                ("seq", self.seq_len),
                ("eval rows", self.eval_rows),
                ("eval every", self.eval_interval),
                ("baseline rows", self.baseline_rows),
            ]
            for col, (label, var) in enumerate(fields):
                ttk.Label(numeric, text=label).grid(row=0, column=col, sticky="w")
                ttk.Spinbox(numeric, textvariable=var, from_=1, to=1_000_000, width=10).grid(row=1, column=col, sticky="ew", padx=2)
            ttk.Label(numeric, text="lr").grid(row=2, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(numeric, textvariable=self.lr, width=10).grid(row=3, column=0, sticky="ew", padx=2)

            self.progress = ttk.Progressbar(tab, maximum=500)
            self.progress.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 2))
            self.progress_label = tk.StringVar(value="no run yet")
            ttk.Label(tab, textvariable=self.progress_label).grid(row=10, column=0, columnspan=3, sticky="w")

            actions = ttk.Frame(tab)
            actions.grid(row=11, column=0, columnspan=3, sticky="ew", pady=8)
            ttk.Button(actions, text="Audit CSV", command=self.run_audit).pack(side="left")
            ttk.Button(actions, text="Start training", command=self.run_training).pack(side="left", padx=8)
            ttk.Button(actions, text="Open GUI from CLI", command=lambda: self.run_command([*command_entrypoint(), "gui"], "gui")).pack(side="left")

        def _build_sample_tab(self, tab: ttk.Frame) -> None:
            tab.columnconfigure(1, weight=1)
            self.sample_checkpoint = tk.StringVar(value="reports/csv_probe.best.pt")
            self.max_tokens = tk.IntVar(value=120)
            self._row(tab, "Checkpoint", self.sample_checkpoint, 0, "checkpoint")
            ttk.Label(tab, text="Max tokens").grid(row=1, column=0, sticky="w", pady=3)
            ttk.Spinbox(tab, textvariable=self.max_tokens, from_=1, to=2048, width=10).grid(row=1, column=1, sticky="w", pady=3, padx=(8, 4))
            ttk.Label(tab, text="Prompt").grid(row=2, column=0, sticky="nw", pady=3)
            self.prompt = tk.Text(tab, height=8, wrap="word")
            self.prompt.insert("1.0", "A noun is")
            self.prompt.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=(8, 0))
            tab.rowconfigure(2, weight=1)
            ttk.Button(tab, text="Stream sample", command=self.run_sample).grid(row=3, column=1, sticky="w", pady=8, padx=(8, 0))
            ttk.Label(
                tab,
                text="This is a byte-level checkpoint inspector. It is not a real chat model yet.",
            ).grid(row=4, column=1, columnspan=2, sticky="w", padx=(8, 0))

        def _build_bench_tab(self, tab: ttk.Frame) -> None:
            tab.columnconfigure(1, weight=1)
            self.bench_config = tk.StringVar(value="configs/model/nano.yaml")
            self._row(tab, "Config", self.bench_config, 0, "config")
            buttons = ttk.Frame(tab)
            buttons.grid(row=1, column=0, columnspan=3, sticky="w", pady=10)
            for label, kind in [
                ("Zero neural", "zero"),
                ("Architecture priors", "prior"),
                ("Symbolic ARC", "symbolic"),
                ("Suite", "suite"),
            ]:
                ttk.Button(buttons, text=label, command=lambda k=kind: self.run_benchmark(k)).pack(side="left", padx=(0, 8))

        def _max_rows_value(self) -> int | None:
            value = self.max_rows.get().strip()
            return int(value) if value else None

        def _train_options(self) -> CSVTrainGuiOptions:
            options = CSVTrainGuiOptions(
                csv_path=self.csv_path.get().strip(),
                config=self.config_path.get().strip(),
                text_column=self.text_column.get().strip(),
                target_column=self.target_column.get().strip(),
                steps=int(self.steps.get()),
                batch_size=int(self.batch_size.get()),
                seq_len=int(self.seq_len.get()),
                max_rows=self._max_rows_value(),
                eval_rows=int(self.eval_rows.get()),
                eval_interval=int(self.eval_interval.get()),
                baseline_rows=int(self.baseline_rows.get()),
                lr=float(self.lr.get()),
                checkpoint=self.checkpoint.get().strip(),
                log=self.train_log.get().strip(),
                output=self.train_output.get().strip(),
            )
            self.expected_steps.set(options.steps)
            self.progress.configure(maximum=options.steps)
            return options

        def run_audit(self) -> None:
            try:
                cmd = build_audit_csv_command(
                    self.csv_path.get().strip(),
                    self.text_column.get().strip(),
                    self.target_column.get().strip(),
                    self._max_rows_value(),
                )
            except Exception as exc:
                messagebox.showerror("WAI-R0", str(exc))
                return
            self.run_command(cmd, "audit csv")

        def run_training(self) -> None:
            try:
                cmd = build_train_csv_command(self._train_options())
            except Exception as exc:
                messagebox.showerror("WAI-R0", str(exc))
                return
            self.run_command(cmd, "csv training")

        def run_sample(self) -> None:
            try:
                cmd = build_sample_csv_command(
                    self.sample_checkpoint.get().strip(),
                    self.prompt.get("1.0", "end").strip(),
                    int(self.max_tokens.get()),
                )
            except Exception as exc:
                messagebox.showerror("WAI-R0", str(exc))
                return
            self.run_command(cmd, "checkpoint sample")

        def run_benchmark(self, kind: str) -> None:
            try:
                cmd = build_benchmark_command(kind, self.bench_config.get().strip())
            except Exception as exc:
                messagebox.showerror("WAI-R0", str(exc))
                return
            self.run_command(cmd, kind)

        def run_command(self, cmd: list[str], label: str) -> None:
            self.process.start(cmd, label)

        def append_console(self, text: str, tag: str | None = None) -> None:
            if tag:
                self.console.insert("end", text, tag)
            else:
                self.console.insert("end", text)
            self.console.see("end")

        def _clear_console(self) -> None:
            self.console.delete("1.0", "end")
            self.line_buffer = ""

        def set_status(self, text: str) -> None:
            self.status.set(text)

        def clear_progress(self) -> None:
            self.progress["value"] = 0
            self.progress_label.set("running")
            self.line_buffer = ""

        def _handle_output(self, chunk: str) -> None:
            self.append_console(chunk)
            self.line_buffer += chunk
            while "\n" in self.line_buffer:
                line, self.line_buffer = self.line_buffer.split("\n", 1)
                event = parse_training_event(line)
                if event is not None:
                    self._update_progress(event)

        def _update_progress(self, event: dict[str, Any]) -> None:
            step = int(event.get("step") or 0)
            self.progress["value"] = step
            elapsed = float(event.get("seconds_elapsed") or 0.0)
            train_loss = event.get("train_loss")
            eval_loss = event.get("eval_loss")
            speed = step / elapsed if elapsed > 0 else 0.0
            pieces = [f"step {step}/{self.expected_steps.get()}", f"{speed:.2f} step/s"]
            if isinstance(train_loss, int | float):
                pieces.append(f"train {float(train_loss):.4f}")
            if isinstance(eval_loss, int | float):
                pieces.append(f"eval {float(eval_loss):.4f}")
            self.progress_label.set(" | ".join(pieces))

        def _drain(self) -> None:
            try:
                while True:
                    item = self.queue.get_nowait()
                    if item[0] == "output":
                        self._handle_output(str(item[1]))
                    elif item[0] == "done":
                        _, code, elapsed = item
                        self.append_console(f"\n[gui] process exited with code {code} in {elapsed:.2f}s\n")
                        self.set_status("idle" if code == 0 else f"failed: {code}")
            except queue.Empty:
                pass
            self.root.after(80, self._drain)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    WaiR0Workbench(root)
    root.mainloop()
