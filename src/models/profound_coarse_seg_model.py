"""Stage-1 ProFound-Conv coarse lesion segmentation model."""
from __future__ import annotations
import torch
from torch import nn
from .profound.profound_conv_wrapper import ProFoundConvEncoderWrapper
from .fusion.lesion_aware_enhancement_3d import LesionAwareEnhancement3D
from .fusion.modality_aware_fusion_3d import ModalityAwareFusion3D
from .decoders.unetr3d_style_coarse_decoder import UNetR3DStyleCoarseDecoder


class ProFoundCoarseSegModel(nn.Module):
    """ProFound-Conv encoder + lesion-aware enhancement + UNetR3D-style decoder."""
    def __init__(
        self,
        checkpoint_path: str | None,
        profound_repo_path: str | None = None,
        profound_model_import_path: str | None = None,
        profound_model_kwargs: dict | None = None,
        profound_checkpoint_format: str = "auto",
        freeze_encoder: bool = True,
        strict_load: bool = False,
        return_multi_scale_features: bool = True,
        encoder_channels: list[int] | None = None,
        use_lesion_aware_enhancement: bool = True,
        use_modality_aware_fusion: bool = False,
        modality_fusion_hidden_channels: int = 16,
        modality_fusion_residual_scale: float = 0.1,
        decoder_channels: list[int] | None = None,
        out_channels: int = 1,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        self.return_aux = return_aux
        encoder_channels = encoder_channels or [64, 128, 256, 512]
        decoder_channels = decoder_channels or [256, 128, 64, 32]
        self.modality_fusion = (
            ModalityAwareFusion3D(
                in_channels=3,
                hidden_channels=modality_fusion_hidden_channels,
                residual_scale=modality_fusion_residual_scale,
            )
            if use_modality_aware_fusion
            else nn.Identity()
        )
        self.encoder = ProFoundConvEncoderWrapper(
            checkpoint_path=checkpoint_path,
            profound_repo_path=profound_repo_path,
            profound_model_import_path=profound_model_import_path,
            profound_model_kwargs=profound_model_kwargs,
            profound_checkpoint_format=profound_checkpoint_format,
            freeze_encoder=freeze_encoder,
            strict_load=strict_load,
            return_multi_scale_features=return_multi_scale_features,
        )
        self.enhancement = LesionAwareEnhancement3D(encoder_channels[-1]) if use_lesion_aware_enhancement else nn.Identity()
        self.decoder = UNetR3DStyleCoarseDecoder(encoder_channels, decoder_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor):
        fused_input = self.modality_fusion(x)
        features = self.encoder(fused_input)
        deep_key = "stage4" if "stage4" in features else sorted(features.keys())[-1]
        enhanced = self.enhancement(features[deep_key])
        decoder_features = dict(features)
        decoder_features[deep_key] = enhanced
        logits = self.decoder(decoder_features, input_shape=tuple(x.shape[2:]))
        if self.return_aux:
            return {"logits": logits, "features": features, "enhanced_feature": enhanced, "fused_input": fused_input}
        return logits
