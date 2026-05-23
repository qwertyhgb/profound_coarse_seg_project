"""Configurable early stopping for training metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track one validation metric and request stop after stalled epochs."""

    monitor: str = "val_coarse_score"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0005
    enabled: bool = True
    best_value: float | None = None
    best_epoch: int = 0
    counter: int = 0

    @classmethod
    def from_config(cls, cfg: dict | None) -> "EarlyStopping":
        """Build an early-stopping tracker from config."""
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            monitor=str(cfg.get("monitor", "val_coarse_score")),
            mode=str(cfg.get("mode", "max")).lower(),
            patience=int(cfg.get("patience", 10)),
            min_delta=float(cfg.get("min_delta", 0.0005)),
        )

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """Update with the current metric value.

        Returns:
            improved: whether the monitored metric improved.
            should_stop: whether training should stop.
        """
        if not self.enabled:
            return False, False
        improved = self.best_value is None or self._is_improved(value)
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            return True, False
        self.counter += 1
        return False, self.counter >= self.patience

    def state_dict(self) -> dict:
        """Serialize tracker state into a checkpoint."""
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore tracker state from a checkpoint."""
        if not state:
            return
        self.best_value = state.get("best_value", self.best_value)
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))
        self.counter = int(state.get("counter", self.counter))

    def _is_improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta
