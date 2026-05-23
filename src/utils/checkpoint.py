"""Checkpoint save/load helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a complete training checkpoint.

    The checkpoint contains model weights plus optional optimizer/scheduler
    state so training can resume from the same epoch and selection state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    if config_snapshot is not None:
        state["config_snapshot"] = config_snapshot
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict[str, Any]:
    """Load model and optional training state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state.get("model", state), strict=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
