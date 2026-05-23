"""Training loop for Stage-1 coarse lesion segmentation."""
from __future__ import annotations
from pathlib import Path
import copy
import csv
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.picai_npz_dataset import PICAINPZDataset
from src.datasets.collate import picai_collate_fn
from src.models.build_model import build_model
from src.losses.build_loss import build_loss
from src.trainers.evaluator import Evaluator
from src.trainers.early_stopping import EarlyStopping
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.tensorboard_utils import create_summary_writer


class Trainer:
    """Config-driven trainer with AMP, clipping, checkpointing, CSV and TensorBoard logs."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.output_root = Path(cfg.get("logging", {}).get("output_root", "outputs"))
        self.ckpt_dir = self.output_root / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model(cfg["model"]).to(self.device)
        self.loss_fn = build_loss(cfg["loss"]).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(cfg["training"].get("amp", True)) and self.device.type == "cuda"
        )
        self.writer = create_summary_writer(self.output_root / "tensorboard", cfg.get("logging", {}).get("tensorboard", True))
        self.metrics_cfg = self._build_metrics_cfg()
        self.best_values: dict[str, float] = {}
        self.early_stopping = EarlyStopping.from_config(cfg.get("early_stopping"))
        self.start_epoch = 1
        self._resume_if_configured()

    def _build_optimizer(self):
        opt_cfg = self.cfg["optimizer"]
        enc_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        groups = []
        if enc_params:
            groups.append({"params": enc_params, "lr": float(opt_cfg.get("encoder_lr", 1e-5))})
        if head_params:
            groups.append({"params": head_params, "lr": float(opt_cfg.get("head_lr", 1e-4))})
        if not groups:
            raise ValueError("No trainable parameters found. Check freeze_encoder and model heads.")
        return torch.optim.AdamW(groups, weight_decay=float(opt_cfg.get("weight_decay", 1e-4)))

    def _build_scheduler(self):
        sched = self.cfg.get("scheduler", {})
        name = str(sched.get("name", "cosine")).lower()
        if name == "none":
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=10)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg["training"].get("epochs", 200)),
            eta_min=float(sched.get("min_lr", 1e-6)),
        )

    def _build_metrics_cfg(self) -> dict:
        metrics_cfg = dict(self.cfg.get("metrics", {}))
        if isinstance(self.cfg.get("coarse_score"), dict):
            metrics_cfg["coarse_score"] = dict(self.cfg["coarse_score"])
        if "threshold_sweep" in self.cfg:
            metrics_cfg["threshold_sweep"] = self.cfg["threshold_sweep"]
        return metrics_cfg

    def build_loaders(self):
        data = self.cfg["data"]
        train_ds = PICAINPZDataset(
            data["processed_root"], data["train_split"], mode="train",
            train_patch_size=data.get("train_patch_size"),
            use_lesion_aware_sampling=data.get("use_lesion_aware_sampling", True),
            pos_patch_ratio=data.get("pos_patch_ratio", 0.7),
            positive_case_ratio=data.get("positive_case_ratio", 0.6),
            max_cases=data.get("max_train_cases"), seed=self.cfg.get("project", {}).get("seed", 42),
        )
        val_ds = PICAINPZDataset(data["processed_root"], data["val_split"], mode="val", max_cases=data.get("max_val_cases"))
        num_workers = int(data.get("num_workers", 4))
        loader_kwargs = {"pin_memory": bool(data.get("pin_memory", True))}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(data.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
        train_loader = DataLoader(
            train_ds,
            batch_size=int(data.get("batch_size", 1)),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=picai_collate_fn,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(data.get("pin_memory", True)),
            collate_fn=picai_collate_fn,
        )
        return train_loader, val_loader

    def fit(self) -> None:
        train_loader, val_loader = self.build_loaders()
        evaluator = Evaluator(self.model, self.loss_fn, self.device, self.cfg["inference"], self.metrics_cfg)
        csv_path = self.log_dir / "train_log.csv"
        sweep_csv_path = self.log_dir / "threshold_sweep_log.csv"
        rows = self._read_csv(csv_path) if self.start_epoch > 1 else []
        sweep_rows = self._read_csv(sweep_csv_path) if self.start_epoch > 1 else []
        final_epoch = self.start_epoch - 1
        metrics: dict[str, float] = {}

        for epoch in range(self.start_epoch, int(self.cfg["training"].get("epochs", 200)) + 1):
            final_epoch = epoch
            train_loss = self.train_one_epoch(train_loader, epoch)
            metrics = self._empty_metrics()
            learning_rate = self._current_lr()
            should_stop = False

            if epoch % int(self.cfg["training"].get("validate_every", 1)) == 0:
                metrics, threshold_rows = evaluator.evaluate(val_loader, desc=f"Val   epoch {epoch}", return_details=True)
                metrics["learning_rate"] = learning_rate
                for threshold_row in threshold_rows:
                    log_row = {"epoch": epoch, **threshold_row}
                    sweep_rows.append(log_row)
                    self._log_threshold_sweep(epoch, threshold_row)
                if threshold_rows:
                    self._write_csv(sweep_csv_path, sweep_rows)

                self._save_monitored_checkpoints(epoch, metrics)
                should_stop = self._update_early_stopping(epoch, metrics)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        monitor = self.cfg.get("scheduler", {}).get("monitor", "coarse_score")
                        self.scheduler.step(float(metrics.get(monitor, metrics.get("dice", 0.0))))
                    else:
                        self.scheduler.step()
            metrics.setdefault("learning_rate", learning_rate)
            metrics.setdefault("early_stopping_counter", self.early_stopping.counter)
            metrics.setdefault("early_stopping_best", self.early_stopping.best_value if self.early_stopping.best_value is not None else float("nan"))
            metrics.setdefault("early_stopping_best_epoch", self.early_stopping.best_epoch)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "early_stopping_counter": self.early_stopping.counter,
                "early_stopping_best": self.early_stopping.best_value if self.early_stopping.best_value is not None else float("nan"),
                **{f"val_{k}": v for k, v in self._select_log_metrics(metrics).items()},
            }
            rows.append(row)
            self._log_epoch(row)
            if epoch % int(self.cfg["training"].get("save_every", 10)) == 0:
                self._save_training_checkpoint(self.ckpt_dir / "last.pth", epoch, metrics)
            self._write_csv(csv_path, rows)

            if should_stop:
                print(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"best {self.early_stopping.monitor}={self.early_stopping.best_value:.4f} "
                    f"at epoch {self.early_stopping.best_epoch}, "
                    f"patience={self.early_stopping.counter}/{self.early_stopping.patience}."
                )
                break

        self._save_training_checkpoint(self.ckpt_dir / "last.pth", final_epoch, metrics)

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "loss": float("nan"),
            "dice": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "positive_case_dice": float("nan"),
            "lesion_recall": float("nan"),
            "fp_per_case": float("nan"),
            "coarse_score": float("nan"),
            "best_threshold_by_coarse_score": float("nan"),
            "threshold_sweep_best_coarse_score": float("nan"),
        }

    def _select_log_metrics(self, metrics: dict) -> dict:
        """Keep train_log.csv readable while checkpoints still store all metrics."""
        keys = self.cfg.get("logging", {}).get("main_metrics") or [
            "loss",
            "dice",
            "precision",
            "recall",
            "positive_case_dice",
            "lesion_recall",
            "fp_per_case",
            "coarse_score",
            "best_threshold_by_coarse_score",
            "threshold_sweep_best_coarse_score",
        ]
        return {k: metrics[k] for k in keys if k in metrics}

    def _save_monitored_checkpoints(self, epoch: int, metrics: dict) -> None:
        """Save best checkpoints for segmentation and coarse-proposal metrics."""
        monitors = self.cfg.get("training", {}).get("checkpoint_monitors") or [
            {"metric": "coarse_score", "filename": "best_by_val_coarse_score.pth", "mode": "max"},
            {"metric": "threshold_sweep_best_coarse_score", "filename": "best_by_val_threshold_sweep_coarse_score.pth", "mode": "max"},
            {"metric": "lesion_recall", "filename": "best_by_val_lesion_recall.pth", "mode": "max"},
            {"metric": "positive_case_dice", "filename": "best_by_val_positive_case_dice.pth", "mode": "max"},
            {"metric": "dice", "filename": "best_by_val_dice.pth", "mode": "max"},
        ]
        for monitor in monitors:
            metric = str(monitor["metric"])
            value = self._metric_value(metrics, metric)
            if value is None or not math.isfinite(value):
                continue
            mode = str(monitor.get("mode", "max")).lower()
            best = self.best_values.get(metric)
            improved = best is None or (value > best if mode == "max" else value < best)
            if improved:
                self.best_values[metric] = value
                self._save_training_checkpoint(self.ckpt_dir / monitor["filename"], epoch, metrics)

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        n = 0
        amp = bool(self.cfg["training"].get("amp", True)) and self.device.type == "cuda"
        progress = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True, leave=False)
        for batch in progress:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = self.model(image)
                loss = self.loss_fn(logits, label)
            self.scaler.scale(loss).backward()
            clip = self.cfg["training"].get("grad_clip_norm", None)
            if clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_value = float(loss.item())
            total += loss_value
            n += 1
            progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total / max(n, 1):.4f}")
        return total / max(n, 1)

    def _update_early_stopping(self, epoch: int, metrics: dict) -> bool:
        if not self.early_stopping.enabled:
            metrics["early_stopping_counter"] = 0
            return False
        value = self._metric_value(metrics, self.early_stopping.monitor)
        if value is None or not math.isfinite(value):
            print(f"Early stopping monitor {self.early_stopping.monitor} unavailable at epoch {epoch}; continuing.")
            metrics["early_stopping_counter"] = self.early_stopping.counter
            return False
        improved, should_stop = self.early_stopping.step(value, epoch)
        metrics["early_stopping_counter"] = self.early_stopping.counter
        metrics["early_stopping_best"] = self.early_stopping.best_value if self.early_stopping.best_value is not None else float("nan")
        metrics["early_stopping_best_epoch"] = self.early_stopping.best_epoch
        return should_stop

    def _save_training_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        metadata = self._checkpoint_metadata(epoch, metrics)
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            metrics,
            metadata=metadata,
            config_snapshot=copy.deepcopy(self.cfg),
            extra_state={
                "best_values": copy.deepcopy(self.best_values),
                "early_stopping": self.early_stopping.state_dict(),
            },
        )

    def _checkpoint_metadata(self, epoch: int, metrics: dict) -> dict:
        return {
            "epoch": int(epoch),
            "threshold": float(metrics.get("best_threshold_by_coarse_score", self.metrics_cfg.get("threshold", 0.5))),
            "val_loss": float(metrics.get("loss", float("nan"))),
            "val_dice": float(metrics.get("dice", float("nan"))),
            "positive_case_dice": float(metrics.get("positive_case_dice", float("nan"))),
            "lesion_recall": float(metrics.get("lesion_recall", float("nan"))),
            "fp_per_case": float(metrics.get("fp_per_case", metrics.get("fp_components_per_case", float("nan")))),
            "coarse_score": float(metrics.get("coarse_score", float("nan"))),
            "learning_rate": self._current_lr(),
            "config_snapshot": copy.deepcopy(self.cfg),
        }

    def _resume_if_configured(self) -> None:
        resume_from = self.cfg.get("training", {}).get("resume_from") or self.cfg.get("resume_from")
        if not resume_from:
            return
        state = load_checkpoint(resume_from, self.model, self.optimizer, self.scheduler, map_location=self.device)
        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.best_values = {k: float(v) for k, v in state.get("best_values", {}).items()}
        self.early_stopping.load_state_dict(state.get("early_stopping", {}))
        print(f"Resumed training from {resume_from}; next epoch is {self.start_epoch}.")

    def _log_threshold_sweep(self, epoch: int, row: dict) -> None:
        if not self.writer:
            return
        threshold = float(row.get("threshold", 0.0))
        tag_prefix = f"threshold_sweep/thr_{threshold:.2f}"
        for key in ["dice", "lesion_recall", "positive_case_dice", "fp_per_case", "coarse_score"]:
            if key in row:
                self.writer.add_scalar(f"{tag_prefix}/{key}", float(row[key]), epoch)

    def _log_epoch(self, row: dict) -> None:
        print(self._format_epoch_summary(row, self.early_stopping))
        if self.writer:
            epoch = row["epoch"]
            for k, v in row.items():
                if k != "epoch" and isinstance(v, (float, int)) and math.isfinite(float(v)):
                    self.writer.add_scalar(k, v, epoch)

    @staticmethod
    def _format_epoch_summary(row: dict, early_stopping: EarlyStopping | None = None) -> str:
        """Compact human-readable epoch summary for the terminal."""
        def fmt(key: str, default: str = "nan") -> str:
            value = row.get(key)
            if value is None:
                return default
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        es_text = ""
        if early_stopping is not None and early_stopping.enabled:
            best = early_stopping.best_value
            best_text = "nan" if best is None else f"{best:.4f}"
            es_text = f" ES={early_stopping.counter}/{early_stopping.patience} best={best_text}@{early_stopping.best_epoch}"
        return (
            f"E{int(row.get('epoch', 0)):03d} "
            f"train={fmt('train_loss')} "
            f"val={fmt('val_loss')} "
            f"dice={fmt('val_dice')} "
            f"posDice={fmt('val_positive_case_dice')} "
            f"lesionR={fmt('val_lesion_recall')} "
            f"fp/case={fmt('val_fp_per_case')} "
            f"score={fmt('val_coarse_score')} "
            f"sweepScore={fmt('val_threshold_sweep_best_coarse_score')} "
            f"thr={fmt('val_best_threshold_by_coarse_score')} "
            f"lr={fmt('learning_rate')}"
            f"{es_text}"
        )

    def _metric_value(self, metrics: dict, metric: str) -> float | None:
        key = metric[4:] if metric.startswith("val_") else metric
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)

    def _current_lr(self) -> float:
        return max(float(group.get("lr", 0.0)) for group in self.optimizer.param_groups)

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames: list[str] = []
        preferred = [
            "epoch",
            "train_loss",
            "learning_rate",
            "early_stopping_counter",
            "early_stopping_best",
            "threshold",
            "dice",
            "lesion_recall",
            "positive_case_dice",
            "fp_per_case",
            "coarse_score",
        ]
        for key in preferred:
            if any(key in row for row in rows):
                fieldnames.append(key)
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
