"""Validation and test evaluator."""
from __future__ import annotations
from pathlib import Path
import csv
import torch
from tqdm import tqdm
from src.metrics.segmentation_metrics import SegmentationMetricAccumulator
from src.metrics.lesion_metrics import LesionRecallAccumulator
from src.metrics.coarse_score import add_coarse_score


class Evaluator:
    """Run full-volume/sliding-window evaluation."""

    def __init__(self, model, loss_fn, device, inference_cfg: dict, metrics_cfg: dict) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.inference_cfg = inference_cfg
        self.metrics_cfg = metrics_cfg

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run model inference, using MONAI sliding windows when configured."""
        image = image.to(self.device, non_blocking=True)
        if self.inference_cfg.get("use_sliding_window", True):
            try:
                from monai.inferers import sliding_window_inference
            except ImportError as exc:
                raise ImportError(
                    "MONAI is required for sliding-window inference. Install it with `pip install monai` "
                    "or set inference.use_sliding_window=false for small-volume debugging."
                ) from exc
            return sliding_window_inference(
                image,
                roi_size=tuple(self.inference_cfg.get("roi_size", [64, 128, 128])),
                sw_batch_size=int(self.inference_cfg.get("sw_batch_size", 1)),
                predictor=self.model,
                overlap=float(self.inference_cfg.get("overlap", 0.25)),
            )
        return self.model(image)

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        save_csv: str | Path | None = None,
        desc: str = "Validate",
        return_details: bool = False,
    ):
        """Evaluate one loader and optionally return per-threshold sweep rows."""
        self.model.eval()
        base_threshold = float(self.metrics_cfg.get("threshold", 0.5))
        min_pred_component_size = int(self.metrics_cfg.get("min_pred_component_size", 0))
        compute_lesion_recall = bool(self.metrics_cfg.get("compute_lesion_recall", True))

        seg = SegmentationMetricAccumulator(threshold=base_threshold)
        lesion = (
            LesionRecallAccumulator(threshold=base_threshold, min_pred_component_size=min_pred_component_size)
            if compute_lesion_recall
            else None
        )
        sweep = {
            t: (
                SegmentationMetricAccumulator(threshold=t),
                LesionRecallAccumulator(threshold=t, min_pred_component_size=min_pred_component_size)
                if compute_lesion_recall
                else None,
            )
            for t in self._get_sweep_thresholds()
        }

        total_loss = 0.0
        n = 0
        rows = []
        progress = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
        for batch in progress:
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)
            logits = self.predict(image)
            loss = self.loss_fn(logits, label)
            loss_value = float(loss.item())
            total_loss += loss_value
            n += 1
            progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total_loss / max(n, 1):.4f}")

            seg.update(logits, label)
            if lesion is not None:
                lesion.update(logits, label)
            for sweep_seg, sweep_lesion in sweep.values():
                sweep_seg.update(logits, label)
                if sweep_lesion is not None:
                    sweep_lesion.update(logits, label)

            case_id = batch["case_id"][0] if isinstance(batch["case_id"], list) else batch["case_id"]
            rows.append({"case_id": case_id, "loss": loss_value})

        metrics = {"loss": total_loss / max(n, 1), **seg.compute()}
        if lesion is not None:
            metrics.update(lesion.compute())
        add_coarse_score(metrics, self._coarse_score_cfg())

        sweep_rows: list[dict[str, float]] = []
        if sweep:
            best_by_coarse = None
            best_by_dice = None
            for threshold, (sweep_seg, sweep_lesion) in sweep.items():
                threshold_metrics = sweep_seg.compute()
                if sweep_lesion is not None:
                    threshold_metrics.update(sweep_lesion.compute())
                add_coarse_score(threshold_metrics, self._coarse_score_cfg())
                candidate = {"threshold": float(threshold), **threshold_metrics}
                sweep_rows.append(candidate)
                if best_by_coarse is None or candidate.get("coarse_score", -1.0) > best_by_coarse.get("coarse_score", -1.0):
                    best_by_coarse = candidate
                if best_by_dice is None or candidate.get("dice", -1.0) > best_by_dice.get("dice", -1.0):
                    best_by_dice = candidate

            if best_by_coarse is not None:
                metrics["best_threshold_by_coarse_score"] = best_by_coarse["threshold"]
                metrics["threshold_sweep_best_coarse_score"] = best_by_coarse.get("coarse_score", 0.0)
                metrics["threshold_sweep_best_lesion_recall"] = best_by_coarse.get("lesion_recall", 0.0)
                metrics["threshold_sweep_best_positive_case_dice"] = best_by_coarse.get("positive_case_dice", 0.0)
                metrics["threshold_sweep_best_fp_per_case"] = best_by_coarse.get("fp_per_case", 0.0)
                # Backward-compatible aliases for existing logs/configs.
                metrics["best_threshold"] = metrics["best_threshold_by_coarse_score"]
                metrics["best_threshold_coarse_score"] = metrics["threshold_sweep_best_coarse_score"]
            if best_by_dice is not None:
                metrics["best_threshold_by_dice"] = best_by_dice["threshold"]
                metrics["threshold_sweep_best_dice"] = best_by_dice.get("dice", 0.0)

        if save_csv:
            save_csv = Path(save_csv)
            save_csv.parent.mkdir(parents=True, exist_ok=True)
            with save_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["case_id", "loss"])
                writer.writeheader()
                writer.writerows(rows)
        if return_details:
            return metrics, sweep_rows
        return metrics

    def _coarse_score_cfg(self) -> dict:
        """Merge old metrics config and new top-level coarse_score config shape."""
        cfg = dict(self.metrics_cfg)
        if isinstance(self.metrics_cfg.get("coarse_score"), dict):
            cfg.update(self.metrics_cfg["coarse_score"])
        return cfg

    def _get_sweep_thresholds(self) -> list[float]:
        """Return configured threshold sweep values, supporting old and new config formats."""
        sweep_cfg = self.metrics_cfg.get("threshold_sweep", [])
        if isinstance(sweep_cfg, dict):
            if not bool(sweep_cfg.get("enabled", True)):
                return []
            thresholds = sweep_cfg.get("thresholds", [])
        else:
            thresholds = sweep_cfg
        return [float(t) for t in thresholds]
