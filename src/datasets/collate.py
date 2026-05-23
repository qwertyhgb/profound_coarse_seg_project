"""Custom collate functions for PI-CAI batches."""
from __future__ import annotations
import torch


def picai_collate_fn(batch: list[dict]) -> dict:
    """Collate image/label tensors while keeping metadata as Python lists.

    PyTorch's default collate recursively stacks dictionaries. PI-CAI metadata
    can contain variable-length arrays/lists, so batch_size > 1 can fail unless
    metadata stays unstacked.
    """
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "case_id": [item["case_id"] for item in batch],
        "metadata": [item.get("metadata", {}) for item in batch],
        "npz_path": [item.get("npz_path", "") for item in batch],
    }



def stage2_prompt_collate_fn(batch: list[dict]) -> dict:
    """Collate Stage-2 prompt refinement samples."""
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "coarse_prob": torch.stack([item["coarse_prob"] for item in batch], dim=0),
        "box_prior": torch.stack([item["box_prior"] for item in batch], dim=0),
        "point_prior": torch.stack([item["point_prior"] for item in batch], dim=0),
        "case_id": [item["case_id"] for item in batch],
        "proposal_rank": torch.tensor([item["proposal_rank"] for item in batch], dtype=torch.long),
        "component_id": torch.tensor([item["component_id"] for item in batch], dtype=torch.long),
        "bbox_zyxzyx": torch.stack([item["bbox_zyxzyx"] for item in batch], dim=0),
        "center_zyx": torch.stack([item["center_zyx"] for item in batch], dim=0),
        "objectness_label": torch.stack([item["objectness_label"] for item in batch], dim=0),
        "crop_start_zyx": torch.stack([item["crop_start_zyx"] for item in batch], dim=0),
        "crop_end_zyx": torch.stack([item["crop_end_zyx"] for item in batch], dim=0),
        "patch_valid_start_zyx": torch.stack([item["patch_valid_start_zyx"] for item in batch], dim=0),
        "patch_valid_end_zyx": torch.stack([item["patch_valid_end_zyx"] for item in batch], dim=0),
        "original_shape_zyx": torch.stack([item["original_shape_zyx"] for item in batch], dim=0),
        "metadata": [item.get("metadata", {}) for item in batch],
    }
