#!/usr/bin/env python
"""Run single-case inference and save coarse probability maps."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from src.utils.config_utils import load_config
from src.models.build_model import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import save_case_png


def _format_threshold_key(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_single_case.yaml")
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--segmentation-threshold", type=float, default=None)
    parser.add_argument("--prompt-threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for prediction npz files")
    parser.add_argument("--visualization-dir", default=None, help="Directory for optional png visualizations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf = cfg["inference"]
    npz_path = Path(args.npz_path or inf.get("npz_path") or "")
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ path not found: {npz_path}")
    ckpt = args.checkpoint or inf.get("checkpoint_path")
    if not ckpt:
        raise ValueError("Missing checkpoint. Set inference.checkpoint_path or pass --checkpoint.")

    segmentation_threshold = float(
        args.segmentation_threshold
        if args.segmentation_threshold is not None
        else inf.get("segmentation_threshold", inf.get("threshold", 0.5))
    )
    prompt_threshold = float(
        args.prompt_threshold
        if args.prompt_threshold is not None
        else inf.get("prompt_generation_threshold", 0.15)
    )

    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    data = np.load(npz_path, allow_pickle=False)
    if "image" not in data.files:
        raise KeyError(f"NPZ missing required key 'image': {npz_path}")
    image = data["image"].astype(np.float32)
    label = data["label"].astype(np.float32) if "label" in data.files else None
    case_value = data["case_id"] if "case_id" in data.files else npz_path.stem
    case_id = str(case_value.item() if hasattr(case_value, "item") else case_value)

    x = torch.from_numpy(image[None]).to(device)
    with torch.no_grad():
        if inf.get("use_sliding_window", True):
            logits = sliding_window_inference(
                x,
                tuple(inf.get("roi_size", [64, 128, 128])),
                int(inf.get("sw_batch_size", 1)),
                model,
                overlap=float(inf.get("overlap", 0.25)),
            )
        else:
            logits = model(x)
        logits_np = logits.cpu().numpy()[0]
        prob = torch.sigmoid(logits).cpu().numpy()[0]

    seg_mask = (prob >= segmentation_threshold).astype(np.uint8)
    prompt_mask = (prob >= prompt_threshold).astype(np.uint8)
    out_dir = Path(args.output_dir or inf.get("output_dir", "outputs/predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_seg_key = _format_threshold_key("binary_mask_thr", segmentation_threshold)
    exact_prompt_key = _format_threshold_key("binary_mask_prompt_thr", prompt_threshold)
    save_payload = {
        "logits": logits_np,
        "probability": prob,
        exact_seg_key: seg_mask,
        exact_prompt_key: prompt_mask,
        # Sanitized aliases are convenient for Python attribute-like readers.
        exact_seg_key.replace(".", "_"): seg_mask,
        exact_prompt_key.replace(".", "_"): prompt_mask,
        "segmentation_threshold": np.array(segmentation_threshold, dtype=np.float32),
        "prompt_generation_threshold": np.array(prompt_threshold, dtype=np.float32),
        "case_id": np.array(case_id),
    }
    np.savez_compressed(out_dir / f"{case_id}_coarse_pred.npz", **save_payload)

    if inf.get("save_png", True):
        vis_dir = Path(args.visualization_dir or inf.get("visualization_dir", "outputs/visualizations"))
        save_case_png(
            image,
            label,
            prob,
            vis_dir / f"{case_id}_coarse.png",
            case_id,
            segmentation_threshold=segmentation_threshold,
            prompt_threshold=prompt_threshold,
        )
    print(
        f"Saved coarse prediction for {case_id} to {out_dir} "
        f"(seg_thr={segmentation_threshold:.2f}, prompt_thr={prompt_threshold:.2f})"
    )


if __name__ == "__main__":
    main()
