from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    device: str
    device_type: str
    name: str
    total_memory_bytes: int | None
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    fp16_supported: bool
    flash_sdpa_available: bool
    memory_efficient_sdpa_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_device(device: str | torch.device = "cpu") -> DeviceCapabilities:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        return DeviceCapabilities(
            device=f"cuda:{index}",
            device_type="cuda",
            name=properties.name,
            total_memory_bytes=int(properties.total_memory),
            compute_capability=(int(capability[0]), int(capability[1])),
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            fp16_supported=True,
            flash_sdpa_available=hasattr(torch.backends.cuda, "flash_sdp_enabled"),
            memory_efficient_sdpa_available=hasattr(
                torch.backends.cuda, "mem_efficient_sdp_enabled"
            ),
        )
    if resolved.type != "cpu":
        raise ValueError(f"unsupported device type: {resolved.type}")
    return DeviceCapabilities(
        device="cpu",
        device_type="cpu",
        name=platform.processor() or platform.machine() or "cpu",
        total_memory_bytes=None,
        compute_capability=None,
        bf16_supported=False,
        fp16_supported=False,
        flash_sdpa_available=False,
        memory_efficient_sdpa_available=False,
    )


def runtime_capabilities() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "devices": [inspect_device("cpu").to_dict()],
    }
    if torch.cuda.is_available():
        payload["devices"].extend(
            inspect_device(f"cuda:{index}").to_dict() for index in range(torch.cuda.device_count())
        )
    return payload


__all__ = ["DeviceCapabilities", "inspect_device", "runtime_capabilities"]
