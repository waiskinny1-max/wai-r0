from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from wai_r0.config import ReasonerConfig
from wai_r0.model import ReasonerCore, set_seed


@dataclass(frozen=True)
class TinyProbeResult:
    task: str
    examples: int
    train_len: int
    eval_lens: tuple[int, ...]
    initial_loss: float
    final_loss: float
    loss_delta: float
    eval_token_accuracy: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _target_for(task: str, tokens: torch.Tensor) -> torch.Tensor:
    if task == "copy":
        return tokens.clone()
    if task == "reverse":
        return torch.flip(tokens, dims=[1])
    if task == "parity":
        parity = (tokens % 2).sum(dim=1, keepdim=True) % 2
        return parity.expand_as(tokens).clone()
    raise ValueError(f"unsupported tiny-training task: {task}")


def _batch(cfg: ReasonerConfig, task: str, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    safe_len = min(seq_len, cfg.max_seq_len)
    tokens = torch.randint(1, cfg.vocab_size, (batch_size, safe_len), device=device)
    return tokens, _target_for(task, tokens)


def _token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


def run_tiny_probe(
    cfg: ReasonerConfig,
    task: str = "copy",
    examples: int = 32,
    batch_size: int = 4,
    train_len: int = 8,
    eval_lens: tuple[int, ...] = (8, 16),
    lr: float = 3e-4,
) -> TinyProbeResult:
    """Run a deliberately small supervised algorithmic probe.

    This is a sample-efficiency smoke test. It is not a pretraining run and should
    never be interpreted as learned language reasoning.
    """

    set_seed(cfg.seed)
    core = ReasonerCore(cfg)
    opt = torch.optim.AdamW(core.parameters(), lr=lr)
    steps = max(1, examples // batch_size)
    initial_loss: float | None = None
    final_loss = 0.0

    for _ in range(steps):
        x, y = _batch(cfg, task, batch_size, train_len, core.device_obj)
        opt.zero_grad(set_to_none=True)
        logits = core(x, mode="think" if cfg.recurrent_depth > 1 else "fast")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())

    eval_acc: dict[str, float] = {}
    core.eval()
    with torch.no_grad():
        for length in eval_lens:
            x, y = _batch(cfg, task, batch_size, length, core.device_obj)
            logits = core(x, mode="think" if cfg.recurrent_depth > 1 else "fast")
            eval_acc[str(min(length, cfg.max_seq_len))] = _token_accuracy(logits, y)

    start = float(initial_loss if initial_loss is not None else final_loss)
    return TinyProbeResult(
        task=task,
        examples=examples,
        train_len=min(train_len, cfg.max_seq_len),
        eval_lens=tuple(min(length, cfg.max_seq_len) for length in eval_lens),
        initial_loss=start,
        final_loss=float(final_loss),
        loss_delta=start - float(final_loss),
        eval_token_accuracy=eval_acc,
    )
