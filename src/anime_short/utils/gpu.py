"""GPU helpers (SPEC §12). The GPU is shared with other services: never assume it is free."""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

log = logging.getLogger(__name__)


class InsufficientVRAM(RuntimeError):
    pass


class Loadable(Protocol):
    name: str

    def load(self) -> None: ...

    def unload(self) -> None: ...


def free_vram_gb(device: int = 0) -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def gpu_session(model: Loadable, min_free_gb: float = 0.0) -> Iterator[Loadable]:
    """Load `model`, yield it, always unload and release memory. Raise InsufficientVRAM first
    if less than `min_free_gb` is free."""
    free = free_vram_gb()
    if free < min_free_gb:
        raise InsufficientVRAM(
            f"{model.name}: needs {min_free_gb:.1f} GB free VRAM, only {free:.1f} GB available"
        )
    log.info("loading %s (free VRAM %.1f GB)", model.name, free)
    model.load()
    try:
        yield model
    finally:
        model.unload()
        release_cuda_memory()
        log.info("unloaded %s (free VRAM %.1f GB)", model.name, free_vram_gb())
