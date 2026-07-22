from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wai_r0.inference.sampling import SamplingConfig, sample_next_token
from wai_r0.model import ModelOutput, ReasonerCore


@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: torch.Tensor
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    stopped_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = self.token_ids.tolist()
        return payload


def _ends_with(sequence: torch.Tensor, stop: tuple[int, ...]) -> torch.Tensor:
    if not stop or sequence.shape[1] < len(stop):
        return torch.zeros(sequence.shape[0], dtype=torch.bool, device=sequence.device)
    suffix = sequence[:, -len(stop) :]
    expected = torch.tensor(stop, device=sequence.device, dtype=sequence.dtype)
    return suffix.eq(expected[None]).all(dim=-1)


@torch.no_grad()
def generate_tokens(
    model: ReasonerCore,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    attention_mask: torch.Tensor | None = None,
    sampling: SamplingConfig | None = None,
    eos_token_id: int | None = None,
    stop_sequences: list[tuple[int, ...]] | None = None,
    use_cache: bool = True,
) -> GenerationResult:
    if prompt.ndim != 2 or prompt.shape[1] < 1:
        raise ValueError("prompt must have shape [batch, non-empty time]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt.shape[1] + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("prompt plus generated tokens exceeds model max_seq_len")
    active_sampling = sampling or SamplingConfig()
    active_sampling.validate()
    stops = stop_sequences or []
    if any(not sequence for sequence in stops):
        raise ValueError("stop sequences cannot be empty")
    device = model.device_obj
    output_tokens = prompt.to(device=device, dtype=torch.long).clone()
    mask = (
        attention_mask.to(device=device, dtype=torch.bool)
        if attention_mask is not None
        else torch.ones_like(output_tokens, dtype=torch.bool)
    )
    if mask.shape != output_tokens.shape:
        raise ValueError("attention_mask must match prompt shape")
    generator = None
    if active_sampling.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(active_sampling.seed)
    finished = torch.zeros(output_tokens.shape[0], dtype=torch.bool, device=device)
    generated = 0
    stopped_reason = "max_new_tokens"
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        cache = None
        next_logits: torch.Tensor | None = None
        if max_new_tokens and use_cache:
            prefill = model(
                output_tokens,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )
            if not isinstance(prefill, ModelOutput) or prefill.past_key_values is None:
                raise RuntimeError("cached prefill did not return cache state")
            cache = prefill.past_key_values
            next_logits = prefill.logits[:, -1, :]
        for _ in range(max_new_tokens):
            if use_cache:
                if next_logits is None:
                    raise RuntimeError("cached generation is missing logits")
            else:
                full = model(output_tokens, attention_mask=mask, return_dict=True)
                if not isinstance(full, ModelOutput):
                    raise RuntimeError("generation expected structured model output")
                next_logits = full.logits[:, -1, :]
            next_token = sample_next_token(
                next_logits,
                previous_tokens=output_tokens,
                config=active_sampling,
                generator=generator,
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished[:, None], torch.full_like(next_token, eos_token_id), next_token
                )
            output_tokens = torch.cat((output_tokens, next_token), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token, dtype=torch.bool)), dim=1)
            generated += 1
            if eos_token_id is not None:
                finished |= next_token[:, 0].eq(eos_token_id)
            for stop in stops:
                finished |= _ends_with(output_tokens, stop)
            if bool(finished.all().detach().cpu()):
                stopped_reason = "stop_sequence_or_eos"
                break
            if use_cache:
                decoded = model(
                    next_token,
                    attention_mask=mask,
                    state={
                        "past_key_values": cache,
                        "attention_mask": mask[:, :-1],
                        "batch_size": next_token.shape[0],
                    },
                    use_cache=True,
                    return_dict=True,
                )
                if not isinstance(decoded, ModelOutput) or decoded.past_key_values is None:
                    raise RuntimeError("cached decode did not return cache state")
                cache = decoded.past_key_values
                next_logits = decoded.logits[:, -1, :]
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    return GenerationResult(
        token_ids=output_tokens.detach().cpu(),
        generated_tokens=generated,
        elapsed_seconds=elapsed,
        tokens_per_second=generated * output_tokens.shape[0] / elapsed if elapsed > 0 else 0.0,
        stopped_reason=stopped_reason,
    )


__all__ = ["GenerationResult", "generate_tokens"]
