from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def temporary_torch_threads(threads: int | None) -> Iterator[int]:
    """Temporarily set PyTorch's intra-op CPU thread count and restore it.

    Tiny local models can become dramatically slower when PyTorch fans a small
    matrix operation across every host core. This context manager makes the
    thread policy explicit and reproducible without permanently mutating the
    process-wide setting. Inter-op threads are intentionally left unchanged:
    PyTorch does not permit changing them reliably after parallel work starts.
    """

    previous = torch.get_num_threads()
    if threads is None:
        yield previous
        return
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer or None")
    changed = threads != previous
    if changed:
        torch.set_num_threads(threads)
    try:
        yield threads
    finally:
        if changed:
            torch.set_num_threads(previous)


__all__ = ["temporary_torch_threads"]
