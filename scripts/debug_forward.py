#!/usr/bin/env python
"""Debug model or decoder-only forward shapes."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, torch
from src.utils.config_utils import load_config
from src.models.build_model import build_model
from src.models.fusion.lesion_aware_enhancement_3d import LesionAwareEnhancement3D
from src.models.decoders.unetr3d_style_coarse_decoder import UNetR3DStyleCoarseDecoder


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_decoder_only(cfg, shape):
    b, c, d, h, w = shape
    enc_ch = cfg["model"].get("encoder_channels", [64,128,256,512])
    dec_ch = cfg["model"].get("decoder_channels", [256,128,64,32])
    feats = {
        "stage1": torch.randn(b, enc_ch[0], max(d//2,1), max(h//2,1), max(w//2,1)),
        "stage2": torch.randn(b, enc_ch[1], max(d//4,1), max(h//4,1), max(w//4,1)),
        "stage3": torch.randn(b, enc_ch[2], max(d//8,1), max(h//8,1), max(w//8,1)),
        "stage4": torch.randn(b, enc_ch[3], max(d//16,1), max(h//16,1), max(w//16,1)),
    }
    enh = LesionAwareEnhancement3D(enc_ch[-1]) if cfg["model"].get("use_lesion_aware_enhancement", True) else torch.nn.Identity()
    dec = UNetR3DStyleCoarseDecoder(enc_ch, dec_ch, out_channels=1)
    enhanced = enh(feats["stage4"]); feats["stage4"] = enhanced
    logits = dec(feats, input_shape=(d,h,w))
    print("input shape:", shape)
    print("encoder feature shapes:", {k: tuple(v.shape) for k,v in feats.items()})
    print("enhanced feature shape:", tuple(enhanced.shape))
    print("logits shape:", tuple(logits.shape))
    assert tuple(logits.shape) == (b, 1, d, h, w)
    total, trainable = count_params(torch.nn.ModuleList([enh, dec]))
    print("decoder-only params:", total, "trainable:", trainable)


def run_model(cfg, shape):
    model = build_model(cfg["model"])
    x = torch.randn(*shape)
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out
    print("input shape:", tuple(x.shape))
    print("logits shape:", tuple(logits.shape))
    assert tuple(logits.shape) == (shape[0], 1, *shape[2:])
    total, trainable = count_params(model)
    print("total params:", total, "trainable params:", trainable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_profound_coarse.yaml")
    parser.add_argument("--mode", choices=["decoder_only", "model"], default="decoder_only")
    parser.add_argument("--shape", nargs=5, type=int, default=[1,3,64,128,128])
    args = parser.parse_args()
    cfg = load_config(args.config)
    shape = tuple(args.shape)
    if args.mode == "decoder_only":
        run_decoder_only(cfg, shape)
    else:
        run_model(cfg, shape)

if __name__ == "__main__":
    main()
