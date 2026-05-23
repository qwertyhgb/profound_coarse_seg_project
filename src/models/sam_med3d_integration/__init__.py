"""SAM-Med3D integration for PI-CAI prostate cancer refinement.

This package provides:
- Phase 1: SAM-Med3D fine-tune baseline (multi-channel input adapter)
- Phase 2: ProFound × SAM-Med3D (encoder replacement) [TBD]
"""
from .sam_med3d_wrapper import SAMMed3DStage2, build_sam_med3d_stage2

__all__ = ["SAMMed3DStage2", "build_sam_med3d_stage2"]
