from __future__ import annotations

import json

import yaml

from wai_r0.v05_cli import main
from wai_r0.version import __version__


def test_version_command(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_config_validate_command(tmp_path, capsys) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 32,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 4,
                "d_ff": 32,
                "max_seq_len": 16,
            }
        ),
        encoding="utf-8",
    )
    assert main(["config", "validate", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["config"]["d_model"] == 16


def test_legacy_command_routing_contract(monkeypatch) -> None:
    import importlib.machinery
    import sys
    import types

    from wai_r0 import v05_cli

    calls: list[list[str]] = []
    module = types.ModuleType("wai_r0.cli")
    module.__spec__ = importlib.machinery.ModuleSpec("wai_r0.cli", loader=None)

    def legacy_main() -> int:
        calls.append(list(sys.argv[1:]))
        return 7

    module.main = legacy_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wai_r0.cli", module)

    assert v05_cli._should_delegate([]) is False
    assert v05_cli._should_delegate(["--help"]) is False
    assert v05_cli._should_delegate(["train", "--help"]) is False
    assert v05_cli._should_delegate(["suite"]) is True
    assert v05_cli._should_delegate(["train", "plan.md"]) is True
    assert v05_cli._should_delegate(["train", "csv"]) is False
    assert v05_cli._should_delegate(["report", "validate"]) is False
    assert v05_cli._delegate_legacy(["suite", "--config", "x.yaml"]) == 7
    assert calls == [["suite", "--config", "x.yaml"]]
