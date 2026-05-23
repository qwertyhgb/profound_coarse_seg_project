"""Stage-2 3D prompt-conditioned mask decoder.

This file exposes the decoder used by the current Stage-2 implementation. It is
not a literal SAM transformer decoder; instead it is a compact volumetric decoder
that consumes image patches plus dense 3D prompt priors. The design keeps the
research direction compatible with MedSAM/PCaSAM/SAM-Med3D style prompting while
remaining practical for PI-CAI lesion patches.
"""
from __future__ import annotations

from .coarse_prompt_refinement_model import CoarsePromptRefinementModel


class PromptConditionedMaskDecoder3D(CoarsePromptRefinementModel):
    """Alias class for the trainable Stage-2 prompt-conditioned decoder."""


__all__ = ["PromptConditionedMaskDecoder3D"]
