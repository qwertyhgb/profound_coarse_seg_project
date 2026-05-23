"""Stage-2 prompt-conditioned refinement head.

The current refinement head predicts a residual correction on top of the coarse
probability logits and optionally predicts proposal objectness. Objectness is
used to rank/filter coarse components at case-level inference, similar in spirit
to mask-quality heads used by promptable segmentation models.
"""
from __future__ import annotations

from .coarse_prompt_refinement_model import CoarsePromptRefinementModel


class Stage2RefinementHead(CoarsePromptRefinementModel):
    """Alias class for the Stage-2 refinement model used in configs."""


__all__ = ["Stage2RefinementHead"]
