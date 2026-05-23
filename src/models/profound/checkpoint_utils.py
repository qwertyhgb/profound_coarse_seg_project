"""Checkpoint loading helpers for external ProFound weights."""
from __future__ import annotations
from pathlib import Path
import torch


def load_state_dict_flexible(model, checkpoint_path: str | Path, strict: bool = False) -> None:
    """Load a checkpoint into a model, accepting common wrapper keys."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ProFound checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[ProFound load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[ProFound load] Unexpected keys: {len(unexpected)}")
