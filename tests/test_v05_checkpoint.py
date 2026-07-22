from __future__ import annotations

import random

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore
from wai_r0.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingProgress,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _step(model: ReasonerCore, optimizer: torch.optim.Optimizer, tokens: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = logits.float().square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def test_checkpoint_restores_model_optimizer_progress_and_rng(tmp_path) -> None:
    config = ReasonerConfig(
        vocab_size=23,
        d_model=12,
        n_layers=1,
        n_heads=3,
        n_kv_heads=3,
        d_ff=24,
        max_seq_len=8,
        seed=11,
    )
    model = ReasonerCore(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor([[1, 2, 3, 4]])
    _step(model, optimizer, tokens)

    path = tmp_path / "checkpoint.pt"
    progress = TrainingProgress(global_step=1, consumed_tokens=3, data_cursor=1)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=progress,
        config=config.to_dict(),
        metadata={"test": True},
    )

    expected_python = random.random()
    expected_torch = torch.rand(4)
    expected_loss = _step(model, optimizer, tokens)
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    restored_model = ReasonerCore(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )
    actual_python = random.random()
    actual_torch = torch.rand(4)
    actual_loss = _step(restored_model, restored_optimizer, tokens)

    assert restored.progress == progress
    assert restored.metadata == {"test": True}
    assert actual_python == expected_python
    torch.testing.assert_close(actual_torch, expected_torch, rtol=0, atol=0)
    assert actual_loss == expected_loss
    for actual, expected in zip(restored_model.parameters(), expected_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    summary = inspect_checkpoint(path)
    assert summary["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert summary["has_optimizer"] is True
    assert summary["progress"]["global_step"] == 1


def test_checkpoint_rejects_config_tampering_and_removes_stale_digest(tmp_path) -> None:
    config = ReasonerConfig(
        vocab_size=23,
        d_model=12,
        n_layers=1,
        n_heads=3,
        n_kv_heads=3,
        d_ff=24,
        max_seq_len=8,
    )
    model = ReasonerCore(config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, config={"model": config.to_dict()})
    sidecar = path.with_suffix(".pt.sha256")
    assert sidecar.is_file()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["model"]["seed"] = 999
    torch.save(payload, path)
    # Recreate a valid file digest so load reaches the independent config-hash check.
    import hashlib

    sidecar.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    with pytest.raises(ValueError, match="config hash"):
        load_checkpoint(path, model=ReasonerCore(config), require_digest=True)

    save_checkpoint(path, model=model, config={}, overwrite=True, write_digest=False)
    assert not sidecar.exists()
