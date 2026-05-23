"""Model factory."""
from __future__ import annotations
from .profound_coarse_seg_model import ProFoundCoarseSegModel
from .refinement.coarse_prompt_refinement_model import CoarsePromptRefinementModel


def build_model(cfg: dict):
    """Build the configured model."""
    name = cfg.get("name", "profound_coarse")
    if name in {"stage2_refinement", "coarse_prompt_refinement"}:
        return CoarsePromptRefinementModel(
            image_channels=cfg.get("image_channels", 3),
            prompt_channels=cfg.get("prompt_channels", 16),
            base_channels=cfg.get("base_channels", 32),
            channel_multipliers=tuple(cfg.get("channel_multipliers", [1, 2, 4])),
            out_channels=cfg.get("out_channels", 1),
            norm=cfg.get("norm", "instance"),
            residual_with_coarse=cfg.get("residual_with_coarse", True),
            use_objectness_head=cfg.get("use_objectness_head", True),
            return_dict=cfg.get("return_dict", False),
        )
    if name != "profound_coarse":
        raise ValueError(f"Unsupported model: {name}")
    return ProFoundCoarseSegModel(
        checkpoint_path=cfg.get("checkpoint_path"),
        profound_repo_path=cfg.get("profound_repo_path"),
        profound_model_import_path=cfg.get("profound_model_import_path"),
        profound_model_kwargs=cfg.get("profound_model_kwargs", {}),
        profound_checkpoint_format=cfg.get("profound_checkpoint_format", "auto"),
        freeze_encoder=cfg.get("freeze_encoder", True),
        strict_load=cfg.get("strict_load", False),
        return_multi_scale_features=cfg.get("return_multi_scale_features", True),
        encoder_channels=cfg.get("encoder_channels", [64, 128, 256, 512]),
        use_lesion_aware_enhancement=cfg.get("use_lesion_aware_enhancement", True),
        use_modality_aware_fusion=cfg.get("use_modality_aware_fusion", False),
        modality_fusion_hidden_channels=cfg.get("modality_fusion_hidden_channels", 16),
        modality_fusion_residual_scale=cfg.get("modality_fusion_residual_scale", 0.1),
        decoder_channels=cfg.get("decoder_channels", [256, 128, 64, 32]),
        out_channels=cfg.get("out_channels", 1),
        return_aux=cfg.get("return_aux", False),
    )
