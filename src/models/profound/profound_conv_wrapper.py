"""Wrapper around the official ProFound-Conv encoder.

This file deliberately does not implement a fallback CNN. Users must provide the
actual ProFound repository/model import path and checkpoint for real training.
Decoder-only smoke tests are handled separately in scripts/debug_forward.py.
"""
from __future__ import annotations
from pathlib import Path
import importlib
import sys
import torch
from torch import nn
from .checkpoint_utils import load_state_dict_flexible


class ProFoundConvEncoderWrapper(nn.Module):
    """Load and run the official ProFound-Conv model as a 3D encoder."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        profound_repo_path: str | None = None,
        profound_model_import_path: str | None = None,
        profound_model_kwargs: dict | None = None,
        profound_checkpoint_format: str = "auto",
        freeze_encoder: bool = True,
        strict_load: bool = False,
        return_multi_scale_features: bool = True,
    ) -> None:
        super().__init__()
        self.return_multi_scale_features = return_multi_scale_features
        self.freeze_encoder = bool(freeze_encoder)
        self.model = self._build_external_model(
            profound_repo_path,
            profound_model_import_path,
            profound_model_kwargs or {},
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, strict_load, profound_checkpoint_format)
        elif freeze_encoder:
            raise FileNotFoundError(
                "freeze_encoder=True requires a real ProFound-Conv checkpoint. "
                "Set model.checkpoint_path to the official weights."
            )
        if freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self

    def _build_external_model(
        self,
        repo_path: str | None,
        import_path: str | None,
        model_kwargs: dict,
    ) -> nn.Module:
        if repo_path:
            repo = Path(repo_path)
            if not repo.exists():
                raise FileNotFoundError(f"ProFound repo path not found: {repo}")
            sys.path.insert(0, str(repo.resolve()))
        if not import_path:
            raise RuntimeError(
                "ProFound model import path is not configured. Set "
                "model.profound_repo_path and model.profound_model_import_path, e.g. "
                "'some.module:build_profound_conv'. This project will not fake ProFound with a generic CNN."
            )
        module_name, _, attr = import_path.partition(":")
        if not module_name or not attr:
            raise ValueError("profound_model_import_path must use 'module.submodule:factory_or_class' format")
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
        model = factory(**model_kwargs) if callable(factory) else factory
        if not isinstance(model, nn.Module):
            raise TypeError("Configured ProFound factory did not return an nn.Module")
        return model

    def _load_checkpoint(self, checkpoint_path: str, strict: bool, checkpoint_format: str) -> None:
        """Load ProFound weights, using official remapping when available."""
        if checkpoint_format in {"auto", "profound_convnextv2_pretrain"}:
            try:
                convnextv2 = importlib.import_module("models.convnextv2")
                remap = getattr(convnextv2, "remap_checkpoint_keys")
                load_state = getattr(convnextv2, "load_state_dict")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state_dict = remap(checkpoint)
                load_state(self.model, state_dict)
                return
            except Exception:
                if checkpoint_format == "profound_convnextv2_pretrain":
                    raise
        load_state_dict_flexible(self.model, checkpoint_path, strict=strict)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a feature dict from the official ProFound model.

        Official implementations may expose multi-scale features differently. If
        the model returns a tensor, it is treated as stage4 bottleneck. If it
        returns a list/tuple, the last four entries are mapped to stage1..stage4.
        If it returns a dict, stage keys are passed through when present.
        """
        try:
            out = self.model(x, ret_hids=True)
        except TypeError:
            out = self.model(x)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], (list, tuple)):
            return self._coerce_features(out[1])
        if isinstance(out, dict):
            if all(k in out for k in ("stage1", "stage2", "stage3", "stage4")):
                return {k: out[k] for k in ("stage1", "stage2", "stage3", "stage4")}
            if "features" in out:
                return self._coerce_features(out["features"])
            if "stage4" in out:
                return {"stage4": out["stage4"]}
        return self._coerce_features(out)

    @staticmethod
    def _coerce_features(out) -> dict[str, torch.Tensor]:
        if torch.is_tensor(out):
            return {"stage4": out}
        if isinstance(out, (list, tuple)):
            feats = list(out)[-4:]
            names = ["stage1", "stage2", "stage3", "stage4"][-len(feats):]
            return dict(zip(names, feats))
        raise TypeError("Unsupported ProFound output. Expected tensor, list/tuple, or feature dict.")
