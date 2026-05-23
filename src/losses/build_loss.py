"""Loss factory."""
from __future__ import annotations
from .dice_bce_loss import DiceBCELoss
from .tversky_loss import TverskyLoss, FocalTverskyLoss, DiceFocalTverskyBCELoss


def _cfg_first(cfg: dict, *keys, default=None):
    """Return the first configured value among backward-compatible aliases."""
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def build_loss(cfg: dict):
    """Build the configured loss function."""
    name = cfg.get("name", "dice_bce").lower()
    fp_weight = _cfg_first(cfg, "tversky_fp_weight", "fp_weight", default=0.3)
    fn_weight = _cfg_first(cfg, "tversky_fn_weight", "fn_weight", default=0.7)
    gamma = _cfg_first(cfg, "focal_gamma", "gamma", default=4.0 / 3.0)
    pos_weight = _cfg_first(cfg, "bce_pos_weight", "pos_weight", default=None)

    if name == "dice_bce":
        return DiceBCELoss(
            dice_weight=cfg.get("dice_weight", 1.0),
            bce_weight=cfg.get("bce_weight", 1.0),
            pos_weight=pos_weight,
        )
    if name == "tversky":
        return TverskyLoss(
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            smooth=cfg.get("smooth", 1.0),
        )
    if name == "focal_tversky":
        return FocalTverskyLoss(
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            gamma=gamma,
            smooth=cfg.get("smooth", 1.0),
        )
    if name in {"dice_focal_tversky_bce", "dice_ft_bce"}:
        return DiceFocalTverskyBCELoss(
            dice_weight=cfg.get("dice_weight", 0.5),
            focal_tversky_weight=cfg.get("focal_tversky_weight", 1.0),
            bce_weight=cfg.get("bce_weight", 0.3),
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            gamma=gamma,
            pos_weight=_cfg_first(cfg, "bce_pos_weight", "pos_weight", default=3.0),
        )
    raise ValueError(f"Unsupported loss: {name}")
