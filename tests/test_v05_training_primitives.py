from __future__ import annotations

import pytest
import torch

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore
from wai_r0.training.losses import causal_language_model_loss
from wai_r0.training.optimizer import parameter_groups_for_weight_decay
from wai_r0.training.schedules import learning_rate_multiplier


def test_learning_rate_schedules_cover_boundaries() -> None:
    assert (
        learning_rate_multiplier(
            0, total_steps=10, warmup_steps=2, schedule="cosine", minimum_ratio=0.1
        )
        == 0.5
    )
    assert (
        learning_rate_multiplier(
            2, total_steps=10, warmup_steps=2, schedule="constant", minimum_ratio=0.1
        )
        == 1.0
    )
    assert (
        learning_rate_multiplier(
            10, total_steps=10, warmup_steps=2, schedule="linear", minimum_ratio=0.1
        )
        == 0.1
    )
    assert (
        learning_rate_multiplier(
            10, total_steps=10, warmup_steps=2, schedule="cosine", minimum_ratio=0.1
        )
        == 0.1
    )
    with pytest.raises(ValueError):
        learning_rate_multiplier(
            0, total_steps=0, warmup_steps=0, schedule="constant", minimum_ratio=0.1
        )
    with pytest.raises(ValueError):
        learning_rate_multiplier(
            0, total_steps=2, warmup_steps=2, schedule="constant", minimum_ratio=0.1
        )
    with pytest.raises(ValueError, match="unsupported"):
        learning_rate_multiplier(
            0, total_steps=2, warmup_steps=0, schedule="bad", minimum_ratio=0.1
        )  # type: ignore[arg-type]


def test_optimizer_groups_exclude_norm_bias_and_embeddings() -> None:
    model = ReasonerCore(
        ReasonerConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
        )
    )
    groups = parameter_groups_for_weight_decay(model, weight_decay=0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    grouped = sum(len(group["params"]) for group in groups)
    assert grouped == len(list(model.parameters()))


def test_language_loss_rejects_invalid_and_combines_auxiliary() -> None:
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[-100, -100, 2, 3]])
    total, components = causal_language_model_loss(
        logits,
        labels,
        auxiliary_losses={"router": torch.tensor(0.2)},
    )
    assert torch.isclose(total, components["language_model"] + components["router"])
    with pytest.raises(ValueError, match="shape"):
        causal_language_model_loss(torch.randn(1, 4), labels)
    with pytest.raises(ValueError, match="no supervised"):
        causal_language_model_loss(logits, torch.full_like(labels, -100))
    with pytest.raises(ValueError, match="scalar"):
        causal_language_model_loss(logits, labels, auxiliary_losses={"bad": torch.ones(2)})


def test_trainer_resume_rejects_resume_critical_config_change(tmp_path) -> None:
    from wai_r0.config import ReasonerConfig
    from wai_r0.model import ReasonerCore
    from wai_r0.training.checkpoint import TrainingProgress, save_checkpoint
    from wai_r0.training.engine import Trainer, TrainerConfig

    model_config = ReasonerConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        d_ff=16,
        max_seq_len=8,
    )
    original_model = ReasonerCore(model_config)
    original_trainer = Trainer(
        original_model,
        TrainerConfig(max_steps=2, learning_rate=1e-3, schedule="constant"),
    )
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        model=original_model,
        optimizer=original_trainer.optimizer,
        scheduler=original_trainer.scheduler,
        scaler=original_trainer.scaler,
        progress=TrainingProgress(global_step=1),
        config=original_trainer._resolved_checkpoint_config(),
    )

    changed_model = ReasonerCore(model_config)
    changed_trainer = Trainer(
        changed_model,
        TrainerConfig(max_steps=3, learning_rate=2e-3, schedule="constant"),
    )
    with pytest.raises(ValueError, match="learning_rate"):
        changed_trainer.resume(checkpoint)
