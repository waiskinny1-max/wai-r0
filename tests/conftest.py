from __future__ import annotations

import contextlib

import torch


def pytest_sessionstart() -> None:
    # Tiny-model tests are faster and more stable without large BLAS thread pools.
    torch.set_num_threads(1)
    # Another imported plugin may have initialized the inter-op pool already.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
