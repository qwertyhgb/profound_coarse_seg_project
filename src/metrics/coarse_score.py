"""Recall-oriented scoring for Stage-1 coarse lesion proposals.

Stage 1 is used to generate lesion candidates for downstream prompts, so the
model selection score should reward finding lesions while keeping false
positive components within a usable range.
"""
from __future__ import annotations

from typing import Any


def normalize_coarse_score_config(cfg: dict[str, Any] | None) -> dict[str, float | bool]:
    """Return a stable coarse-score config with backward-compatible aliases."""
    cfg = cfg or {}
    legacy_weights = cfg.get("coarse_score_weights", {}) if isinstance(cfg.get("coarse_score_weights"), dict) else {}
    legacy_fp = cfg.get("fp_component_penalty", {}) if isinstance(cfg.get("fp_component_penalty"), dict) else {}

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "lesion_recall_weight": float(
            cfg.get("lesion_recall_weight", legacy_weights.get("lesion_recall", 0.7))
        ),
        "positive_case_dice_weight": float(
            cfg.get("positive_case_dice_weight", legacy_weights.get("positive_case_dice", 0.3))
        ),
        "fp_penalty_weight": float(cfg.get("fp_penalty_weight", legacy_fp.get("weight", 0.03))),
        "fp_free_margin": float(cfg.get("fp_free_margin", legacy_fp.get("free_per_case", 1.0))),
        "use_precision_penalty": bool(cfg.get("use_precision_penalty", True)),
        "precision_floor": float(cfg.get("precision_floor", 0.10)),
        "precision_penalty_weight": float(cfg.get("precision_penalty_weight", 1.0)),
        "qualified_min_precision": float(
            cfg.get("qualified_min_precision", cfg.get("qualified_lesion_recall", {}).get("min_precision", 0.05))
            if isinstance(cfg.get("qualified_lesion_recall", {}), dict)
            else cfg.get("qualified_min_precision", 0.05)
        ),
        "qualified_min_positive_case_dice": float(
            cfg.get(
                "qualified_min_positive_case_dice",
                cfg.get("qualified_lesion_recall", {}).get("min_positive_case_dice", 0.05),
            )
            if isinstance(cfg.get("qualified_lesion_recall", {}), dict)
            else cfg.get("qualified_min_positive_case_dice", 0.05)
        ),
    }


def add_coarse_score(metrics: dict[str, float], cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Add coarse proposal score and aliases to a metrics dictionary.

    Formula by default:
        0.7 * lesion_recall + 0.3 * positive_case_dice
        - precision_penalty
        - 0.03 * max(0, fp_per_case - 1)
    """
    score_cfg = normalize_coarse_score_config(cfg)
    fp_per_case = float(metrics.get("fp_per_case", metrics.get("fp_components_per_case", 0.0)))
    precision = float(metrics.get("precision", 0.0))
    positive_case_dice = float(metrics.get("positive_case_dice", 0.0))
    lesion_recall = float(metrics.get("lesion_recall", 0.0))

    metrics["fp_per_case"] = fp_per_case
    metrics["fp_components_per_case"] = fp_per_case

    precision_penalty = 0.0
    if bool(score_cfg["use_precision_penalty"]):
        precision_penalty = float(score_cfg["precision_penalty_weight"]) * max(
            0.0, float(score_cfg["precision_floor"]) - precision
        )

    fp_penalty = float(score_cfg["fp_penalty_weight"]) * max(
        0.0, fp_per_case - float(score_cfg["fp_free_margin"])
    )
    metrics["coarse_score"] = (
        float(score_cfg["lesion_recall_weight"]) * lesion_recall
        + float(score_cfg["positive_case_dice_weight"]) * positive_case_dice
        - precision_penalty
        - fp_penalty
    )

    if (
        precision >= float(score_cfg["qualified_min_precision"])
        and positive_case_dice >= float(score_cfg["qualified_min_positive_case_dice"])
    ):
        metrics["qualified_lesion_recall"] = lesion_recall
    else:
        metrics["qualified_lesion_recall"] = 0.0
    return metrics
