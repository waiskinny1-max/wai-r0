from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from wai_r0.config import ReasonerConfig
from wai_r0.modeling.attention import CausalSelfAttention, MLALiteAttention
from wai_r0.modeling.common import RMSNorm, dtype_from_name, set_seed, tensor_to_float_list
from wai_r0.modeling.feedforward import SwiGLU, TopKMoE
from wai_r0.modeling.recurrence import RecurrentRefinement
from wai_r0.modeling.types import LayerKVCache, ModelOutput


class Block(nn.Module):
    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(cfg.d_model, cfg.norm_epsilon)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.norm_epsilon)
        self.attn: CausalSelfAttention
        if cfg.attention_type == "mla_lite":
            self.attn = MLALiteAttention(cfg)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.ff: TopKMoE | SwiGLU
        self.ff = TopKMoE(cfg) if cfg.use_moe else SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None, dict[str, torch.Tensor]]:
        attention_result = self.attn(
            self.attention_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            collect_diagnostics=collect_diagnostics,
        )
        if use_cache:
            if not isinstance(attention_result, tuple):
                raise RuntimeError("attention cache contract was not honored")
            attention_output, present = attention_result
        else:
            if isinstance(attention_result, tuple):
                raise RuntimeError("attention returned a cache when none was requested")
            attention_output = attention_result
            present = None
        hidden = x + attention_output

        normalized = self.ff_norm(hidden)
        if isinstance(self.ff, TopKMoE):
            ff_result = self.ff(normalized, return_aux=True)
            if not isinstance(ff_result, tuple):
                raise RuntimeError("MoE auxiliary-loss contract was not honored")
            ff_output, auxiliary = ff_result
        else:
            ff_output = self.ff(normalized)
            auxiliary = {}
        return hidden + ff_output, present, auxiliary


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_epsilon)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.last_diagnostics: dict[str, Any] = {}
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._initialize_module)
        self._scale_residual_projections()

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initialization_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o.weight.mul_(scale)
                if isinstance(block.ff, SwiGLU):
                    block.ff.down.weight.mul_(scale)
                else:
                    for expert in block.ff.experts:
                        expert.down.weight.mul_(scale)
                    if block.ff.shared is not None:
                        block.ff.shared.down.weight.mul_(scale)

    def _validate_cache(
        self,
        past_key_values: tuple[LayerKVCache, ...] | None,
        *,
        batch_size: int,
    ) -> tuple[LayerKVCache | None, ...]:
        if past_key_values is None:
            return tuple(None for _ in self.blocks)
        if len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values length must equal n_layers")
        lengths = {cache.sequence_length for cache in past_key_values}
        if len(lengths) > 1:
            raise ValueError("all layer caches must have the same sequence length")
        if any(cache.batch_size != batch_size for cache in past_key_values):
            raise ValueError("cache batch size does not match tokens")
        return past_key_values

    def forward(
        self,
        tokens: torch.Tensor,
        return_hidden: bool = False,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[LayerKVCache, ...] | None = None,
        use_cache: bool = False,
        return_dict: bool = False,
        collect_diagnostics: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | ModelOutput:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        if tokens.numel() == 0:
            raise ValueError("tokens cannot be empty")
        tokens = tokens.long()
        batch_size, query_len = tokens.shape
        collect = (
            self.cfg.diagnostics_default if collect_diagnostics is None else collect_diagnostics
        )
        layer_caches = self._validate_cache(past_key_values, batch_size=batch_size)
        past_len = (
            layer_caches[0].sequence_length if layer_caches and layer_caches[0] is not None else 0
        )
        if past_len + query_len > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if position_ids is not None and position_ids.shape != tokens.shape:
            raise ValueError("position_ids must match tokens shape")
        if attention_mask is not None and attention_mask.shape[0] != batch_size:
            raise ValueError("attention_mask batch dimension does not match tokens")

        hidden = self.embed(tokens)
        activation_norms: list[torch.Tensor] = []
        if collect:
            activation_norms.append(hidden.float().norm(dim=-1).mean())
        presents: list[LayerKVCache] = []
        auxiliary_losses: dict[str, torch.Tensor] = {}

        for layer_index, (block, cache) in enumerate(zip(self.blocks, layer_caches, strict=True)):
            hidden, present, block_auxiliary = block(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=use_cache,
                collect_diagnostics=collect,
            )
            if present is not None:
                presents.append(present)
            if collect:
                activation_norms.append(hidden.float().norm(dim=-1).mean())
            for name, loss in block_auxiliary.items():
                auxiliary_losses[f"layer_{layer_index}.{name}"] = loss

        normalized = self.norm(hidden)
        logits = self.head(normalized)
        diagnostics: dict[str, Any] = {}
        if collect:
            diagnostics = {
                "activation_norms": tensor_to_float_list(torch.stack(activation_norms)),
                "attention": [
                    block.attn.last_stats.to_dict()
                    for block in self.blocks
                    if block.attn.last_stats is not None
                ],
                "moe": [
                    block.ff.last_stats.to_dict()
                    for block in self.blocks
                    if isinstance(block.ff, TopKMoE) and block.ff.last_stats is not None
                ],
            }
            self.last_diagnostics = diagnostics
        else:
            self.last_diagnostics = {}

        if return_dict or use_cache:
            return ModelOutput(
                logits=logits,
                hidden_states=normalized if return_hidden else None,
                past_key_values=tuple(presents) if use_cache else None,
                auxiliary_losses=auxiliary_losses,
                diagnostics=diagnostics,
            )
        if return_hidden:
            return logits, normalized
        return logits

    @staticmethod
    def _last_valid_logits(
        logits: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return logits[:, -1, :]
        if attention_mask.ndim != 2 or attention_mask.shape[:2] != logits.shape[:2]:
            raise ValueError("generation attention_mask must match prompt shape")
        mask = attention_mask.to(device=logits.device, dtype=torch.bool)
        if bool((~mask.any(dim=-1)).any().detach().cpu()):
            raise ValueError("every generation prompt must contain at least one valid token")
        positions = torch.arange(mask.shape[1], device=logits.device).expand_as(mask)
        last_indices = positions.masked_fill(~mask, -1).amax(dim=-1)
        indices = last_indices.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(dim=1, index=indices).squeeze(1)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        if temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        scores = logits.float() / temperature
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive")
            top_k = min(top_k, scores.shape[-1])
            threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        probabilities = scores.softmax(dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if prompt.ndim != 2:
            raise ValueError("prompt must have shape [batch, time]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if prompt.shape[1] + max_new_tokens > self.cfg.max_seq_len:
            raise ValueError("prompt plus generated tokens exceeds max_seq_len")
        if eos_token_id is not None and not 0 <= eos_token_id < self.cfg.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
        if max_new_tokens == 0:
            return prompt.clone()

        was_training = self.training
        self.eval()
        try:
            output_tokens = prompt.clone()
            active_mask = (
                attention_mask.to(device=prompt.device, dtype=torch.bool)
                if attention_mask is not None
                else torch.ones_like(prompt, dtype=torch.bool)
            )
            finished = torch.zeros(prompt.shape[0], device=prompt.device, dtype=torch.bool)

            if not use_cache:
                for _ in range(max_new_tokens):
                    logits = self(output_tokens, attention_mask=active_mask)
                    if not isinstance(logits, torch.Tensor):
                        raise RuntimeError("uncached generation expected tensor logits")
                    next_logits = self._last_valid_logits(logits, active_mask)
                    next_token = self._sample_next_token(
                        next_logits,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        generator=generator,
                    )
                    if eos_token_id is not None:
                        next_token = torch.where(
                            finished[:, None],
                            torch.full_like(next_token, eos_token_id),
                            next_token,
                        )
                        finished |= next_token[:, 0].eq(eos_token_id)
                    output_tokens = torch.cat((output_tokens, next_token), dim=1)
                    active_mask = torch.cat(
                        (active_mask, torch.ones_like(next_token, dtype=torch.bool)), dim=1
                    )
                    if eos_token_id is not None and bool(finished.all().detach().cpu()):
                        break
                return output_tokens

            prefill = self(
                prompt,
                attention_mask=active_mask,
                use_cache=True,
                return_dict=True,
            )
            if not isinstance(prefill, ModelOutput) or prefill.past_key_values is None:
                raise RuntimeError("prefill did not return a KV cache")
            cache = prefill.past_key_values
            next_logits = self._last_valid_logits(prefill.logits, active_mask)
            next_token = self._sample_next_token(
                next_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            if eos_token_id is not None:
                finished |= next_token[:, 0].eq(eos_token_id)
            output_tokens = torch.cat((output_tokens, next_token), dim=1)
            active_mask = torch.cat(
                (active_mask, torch.ones_like(next_token, dtype=torch.bool)), dim=1
            )
            if eos_token_id is not None and bool(finished.all().detach().cpu()):
                return output_tokens

            for _ in range(max_new_tokens - 1):
                decoded = self(
                    next_token,
                    attention_mask=active_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                if not isinstance(decoded, ModelOutput) or decoded.past_key_values is None:
                    raise RuntimeError("decode step did not return a KV cache")
                cache = decoded.past_key_values
                next_token = self._sample_next_token(
                    decoded.logits[:, -1, :],
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                )
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished |= next_token[:, 0].eq(eos_token_id)
                output_tokens = torch.cat((output_tokens, next_token), dim=1)
                active_mask = torch.cat(
                    (active_mask, torch.ones_like(next_token, dtype=torch.bool)), dim=1
                )
                if eos_token_id is not None and bool(finished.all().detach().cpu()):
                    break
            return output_tokens
        finally:
            self.train(was_training)


class ReasonerCore(nn.Module):
    def __init__(self, cfg: ReasonerConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        set_seed(cfg.seed, deterministic=cfg.deterministic)
        self.device_obj = torch.device(cfg.device)
        self.dtype_obj = dtype_from_name(cfg.dtype)
        if self.device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device_obj.type == "cpu" and self.dtype_obj == torch.float16:
            raise ValueError(
                "float16 is not supported by the CPU execution profile; use float32 or bfloat16"
            )
        self.transformer = DecoderOnlyTransformer(cfg)
        self.recurrent = RecurrentRefinement(cfg) if cfg.recurrent_depth > 1 else None
        self.to(device=self.device_obj, dtype=self.dtype_obj)

    def init_state(self, batch_size: int) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {"past_key_values": None, "attention_mask": None, "batch_size": batch_size}

    def forward(
        self,
        tokens: torch.Tensor,
        state: dict[str, Any] | None = None,
        mode: str = "fast",
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        recurrent_steps: int | None = None,
        use_cache: bool = False,
        return_dict: bool = False,
        collect_diagnostics: bool | None = None,
    ) -> torch.Tensor | ModelOutput:
        if mode not in {"fast", "think"}:
            raise ValueError("mode must be 'fast' or 'think'")
        tokens = tokens.to(device=self.device_obj, dtype=torch.long)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device_obj)
        if position_ids is not None:
            position_ids = position_ids.to(device=self.device_obj)
        if state is not None and state.get("batch_size") != tokens.shape[0]:
            raise ValueError("state batch_size does not match tokens")
        past_key_values = state.get("past_key_values") if state else None
        if state is not None and attention_mask is None:
            previous_mask = state.get("attention_mask")
            if previous_mask is not None:
                current_mask = torch.ones(
                    tokens.shape,
                    device=self.device_obj,
                    dtype=torch.bool,
                )
                attention_mask = torch.cat((previous_mask, current_mask), dim=1)

        output = self.transformer(
            tokens,
            return_hidden=mode == "think",
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict or use_cache or mode == "think",
            collect_diagnostics=collect_diagnostics,
        )
        if mode == "fast":
            if state is not None and use_cache:
                if not isinstance(output, ModelOutput):
                    raise RuntimeError("cached execution requires a structured model output")
                if output.past_key_values is None:
                    raise RuntimeError("cached execution returned no cache")
                state["past_key_values"] = output.past_key_values
                cache_mask = output.past_key_values[0].key_padding_mask
                state["attention_mask"] = (
                    cache_mask.detach()
                    if cache_mask is not None
                    else torch.ones(
                        (tokens.shape[0], output.past_key_values[0].sequence_length),
                        device=self.device_obj,
                        dtype=torch.bool,
                    )
                )
            if isinstance(output, tuple):
                raise RuntimeError("core fast mode received an unexpected tuple output")
            return output

        if not isinstance(output, ModelOutput):
            raise RuntimeError("internal model output contract violated")
        if self.recurrent is None:
            return output if (return_dict or use_cache) else output.logits
        if use_cache:
            raise ValueError("cached decoding is only supported in fast mode")
        if output.hidden_states is None:
            raise RuntimeError("think mode requires hidden states")

        refined = self.recurrent(
            output.hidden_states,
            steps=recurrent_steps,
            collect_diagnostics=collect_diagnostics,
        )
        logits = self.transformer.head(self.transformer.norm(refined))
        diagnostics = dict(output.diagnostics)
        should_collect = (
            self.cfg.diagnostics_default if collect_diagnostics is None else collect_diagnostics
        )
        if self.recurrent.last_stats is not None and should_collect:
            diagnostics["recurrent"] = self.recurrent.last_stats.to_dict()
        auxiliary_losses = {
            **output.auxiliary_losses,
            **self.recurrent.last_auxiliary_losses,
        }
        result = ModelOutput(
            logits=logits,
            hidden_states=refined,
            auxiliary_losses=auxiliary_losses,
            diagnostics=diagnostics,
        )
        return result if return_dict else result.logits

    @torch.no_grad()
    def think(self, tokens: torch.Tensor, budget: int) -> torch.Tensor:
        if budget <= 0:
            raise ValueError("budget must be positive")
        result = self(tokens, mode="think", recurrent_steps=budget)
        if isinstance(result, ModelOutput):
            return result.logits
        return result

    def estimate_memory_cost(self, seq_len: int, batch_size: int) -> dict[str, int]:
        if seq_len < 1 or batch_size < 1:
            raise ValueError("seq_len and batch_size must be positive")
        payload_total = sum(
            block.attn.estimate_kv_cache_bytes(seq_len, batch_size, self.dtype_obj)
            for block in self.transformer.blocks
        )
        mask_bytes = batch_size * seq_len * len(self.transformer.blocks)
        position_bytes = (
            batch_size * seq_len * 8 * len(self.transformer.blocks)
            if self.cfg.attention_type == "mla_lite"
            else 0
        )
        return {
            "kv_cache_payload_bytes": int(payload_total),
            "kv_cache_metadata_bytes": int(mask_bytes + position_bytes),
            "kv_cache_bytes": int(payload_total + mask_bytes + position_bytes),
            "per_layer_kv_cache_bytes": int(
                (payload_total + mask_bytes + position_bytes)
                // max(1, len(self.transformer.blocks))
            ),
        }

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        embedding = self.transformer.embed.weight.numel()
        active = total
        if self.cfg.use_moe:
            all_expert = 0
            active_expert = 0
            for block in self.transformer.blocks:
                if not isinstance(block.ff, TopKMoE):
                    continue
                per_expert = sum(
                    parameter.numel() for parameter in block.ff.experts[0].parameters()
                )
                all_expert += per_expert * self.cfg.n_experts
                active_expert += per_expert * self.cfg.experts_per_token
            active = total - all_expert + active_expert
        return {
            "total": int(total),
            "trainable": int(trainable),
            "active_per_token_estimate": int(active),
            "embedding": int(embedding),
        }

    @torch.no_grad()
    def inspect_activations(self, tokens: torch.Tensor) -> dict[str, Any]:
        result = self(
            tokens,
            mode="think" if self.recurrent else "fast",
            return_dict=True,
            collect_diagnostics=True,
        )
        if not isinstance(result, ModelOutput):
            raise RuntimeError("inspection requires a structured model output")
        return {
            "logits_shape": list(result.logits.shape),
            "finite": bool(torch.isfinite(result.logits).all().detach().cpu()),
            "effective_dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "diagnostics": result.diagnostics,
            "recurrent": (
                self.recurrent.last_stats.to_dict()
                if self.recurrent is not None and self.recurrent.last_stats is not None
                else None
            ),
        }
