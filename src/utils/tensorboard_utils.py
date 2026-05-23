"""TensorBoard utility wrapper."""
from __future__ import annotations
from pathlib import Path


def create_summary_writer(log_dir: str | Path, enabled: bool = True):
    """Create SummaryWriter when enabled and tensorboard is available.

    TensorBoard is useful but not required for training. If the package is not
    installed, return None and keep CSV/stdout logging active.
    """
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("TensorBoard is not installed; continuing without TensorBoard logging.")
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))
