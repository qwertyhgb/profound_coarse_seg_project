"""PCaSAM-3D-ProFound: ProFound-Conv encoder + SAM-Med3D prompt-conditioned decoder.

This module implements a unified end-to-end model that:
1. Uses ProFound-Conv (pretrained prostate mpMRI foundation model) as the image encoder
2. Projects multi-scale ProFound features into SAM-Med3D's 384-dim embedding space
3. Generates automatic 3D prompts (box + point) from a coarse segmentation branch
4. Feeds prompts through SAM-Med3D's prompt encoder + mask decoder for refined segmentation

References:
- ProFound: prostate mpMRI foundation model (ConvNeXtV2-Tiny backbone)
- PCaSAM: automatic prompt generation from coarse prostate cancer masks
- SAM-Med3D: volumetric 3D promptable medical segmentation
"""
from .pcasam3d_profound_model import PCaSAM3DProFoundModel, build_pcasam3d_profound
from .feature_bridge import ProFoundToSAMBridge
from .coarse_branch import CoarseBranch
from .auto_prompt_3d import AutoPrompt3DFromCoarse

__all__ = [
    "PCaSAM3DProFoundModel",
    "build_pcasam3d_profound",
    "ProFoundToSAMBridge",
    "CoarseBranch",
    "AutoPrompt3DFromCoarse",
]
