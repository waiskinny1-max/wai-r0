from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from wai_r0.inference.generate import GenerationResult, generate_tokens
from wai_r0.inference.sampling import SamplingConfig
from wai_r0.model import ReasonerCore


@dataclass(slots=True)
class GenerationSession:
    model: ReasonerCore
    prompt_tokens: list[int] = field(default_factory=list)

    def append(self, token_ids: list[int]) -> None:
        if any(token < 0 or token >= self.model.cfg.vocab_size for token in token_ids):
            raise ValueError("session token is outside the model vocabulary")
        if len(self.prompt_tokens) + len(token_ids) > self.model.cfg.max_seq_len:
            raise ValueError("session exceeds model max_seq_len")
        self.prompt_tokens.extend(token_ids)

    def generate(
        self,
        *,
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        if not self.prompt_tokens:
            raise ValueError("session prompt is empty")
        result: GenerationResult = generate_tokens(
            self.model,
            torch.tensor([self.prompt_tokens], dtype=torch.long),
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            eos_token_id=eos_token_id,
        )
        self.prompt_tokens = [int(token) for token in result.token_ids[0].tolist()]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.prompt_tokens), "length": len(self.prompt_tokens)}


__all__ = ["GenerationSession"]
