"""PCaSAM-style automatic 3D prompt generation from coarse segmentation.

Given a coarse probability map, this module generates:
1. 3D bounding box prompts (from connected component analysis)
2. 3D point prompts (component centroids or max-probability points)
3. Dense mask prior (low-res coarse probability for SAM mask input)

Training uses GPU top-k peak points plus a differentiable dense mask prior,
with an optional hard connected-component branch for prompt-distribution matching.
Point/box selection is intentionally non-differentiable; mask prior and coarse
loss carry dense supervision. Inference uses hard thresholding + connected
components to produce discrete prompts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoPrompt3DFromCoarse(nn.Module):
    """Generate point, box, and dense mask prompts from coarse logits."""

    def __init__(
        self,
        coarse_threshold: float = 0.3,
        max_proposals: int = 5,
        min_component_voxels: int = 20,
        point_type: str = "centroid",
        embed_dim: int = 384,
        image_embedding_size: tuple[int, int, int] = (8, 8, 8),
        no_prompt_if_empty: bool = True,
        no_prompt_threshold: float = 0.05,
        box_margin_voxels: int = 4,
        soft_box_std_scale: float = 2.0,
        training_point_mode: str = "topk_peaks",
        training_nms_kernel: int = 9,
        train_hard_prompt_prob: float = 0.20,
        training_use_soft_box: bool = False,
    ) -> None:
        super().__init__()
        self.coarse_threshold = coarse_threshold
        self.max_proposals = max_proposals
        self.min_component_voxels = min_component_voxels
        self.point_type = point_type
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.no_prompt_if_empty = bool(no_prompt_if_empty)
        self.no_prompt_threshold = float(no_prompt_threshold)
        self.box_margin_voxels = int(box_margin_voxels)
        self.soft_box_std_scale = float(soft_box_std_scale)
        self.training_point_mode = str(training_point_mode)
        self.training_nms_kernel = int(training_nms_kernel)
        self.train_hard_prompt_prob = float(train_hard_prompt_prob)
        self.training_use_soft_box = bool(training_use_soft_box)

        self.coarse_point_embedding = nn.Embedding(1, embed_dim)
        self.coarse_not_a_point_embedding = nn.Embedding(1, embed_dim)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        input_shape: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        """Generate prompts from coarse logits."""
        coarse_prob = torch.sigmoid(coarse_logits)

        use_train_hard = (
            self.training
            and self.train_hard_prompt_prob > 0
            and torch.rand((), device=coarse_prob.device).item() < self.train_hard_prompt_prob
        )
        if self.training and not use_train_hard:
            point_coords, point_labels, box_coords, box_valid = self._soft_prompts(coarse_prob, input_shape)
            prompt_mode = "soft"
        else:
            point_coords, point_labels, box_coords, box_valid = self._hard_prompts(coarse_prob, input_shape)
            prompt_mode = "hard"

        mask_prior = F.interpolate(
            coarse_prob, size=self.image_embedding_size, mode="trilinear", align_corners=False
        )

        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box_coords": box_coords,
            "box_valid": box_valid,
            "mask_prior": mask_prior,
            "coarse_prob": coarse_prob,
            "prompt_mode": prompt_mode,
        }

    def _soft_prompts(
        self, coarse_prob: torch.Tensor, input_shape: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training soft path: top-k points plus optional loose global box."""
        B = coarse_prob.shape[0]
        D, H, W = input_shape
        device = coarse_prob.device

        z_coords = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)

        prob = coarse_prob
        if prob.shape[2:] != (D, H, W):
            prob = F.interpolate(prob, size=(D, H, W), mode="trilinear", align_corners=False)

        prob_sum = prob.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        soft_z = (prob * z_coords).sum(dim=(2, 3, 4), keepdim=True) / prob_sum
        soft_y = (prob * y_coords).sum(dim=(2, 3, 4), keepdim=True) / prob_sum
        soft_x = (prob * x_coords).sum(dim=(2, 3, 4), keepdim=True) / prob_sum

        soft_centroid = torch.cat([
            soft_z.view(B, 1, 1),
            soft_y.view(B, 1, 1),
            soft_x.view(B, 1, 1),
        ], dim=2)

        max_prob = prob.flatten(2).amax(dim=2).squeeze(1)
        box_valid = max_prob > self.no_prompt_threshold
        if self.training_point_mode == "topk_peaks":
            point_coords, point_labels = self._topk_peak_points(prob, input_shape, box_valid)
        else:
            point_coords = soft_centroid
            point_labels = torch.where(
                box_valid,
                torch.ones(B, dtype=torch.long, device=device),
                -torch.ones(B, dtype=torch.long, device=device),
            ).view(B, 1)

        var_z = (prob * (z_coords - soft_z).pow(2)).sum(dim=(2, 3, 4), keepdim=True) / prob_sum
        var_y = (prob * (y_coords - soft_y).pow(2)).sum(dim=(2, 3, 4), keepdim=True) / prob_sum
        var_x = (prob * (x_coords - soft_x).pow(2)).sum(dim=(2, 3, 4), keepdim=True) / prob_sum
        std = torch.cat([
            var_z.sqrt().view(B, 1),
            var_y.sqrt().view(B, 1),
            var_x.sqrt().view(B, 1),
        ], dim=1)
        center = soft_centroid[:, 0, :]
        margin = torch.tensor(
            [
                self.box_margin_voxels / max(D - 1, 1),
                self.box_margin_voxels / max(H - 1, 1),
                self.box_margin_voxels / max(W - 1, 1),
            ],
            dtype=prob.dtype,
            device=device,
        ).view(1, 3)
        half_extent = (self.soft_box_std_scale * std + margin).clamp(min=0.02, max=0.5)
        lo = (center - half_extent).clamp(0.0, 1.0)
        hi = (center + half_extent).clamp(0.0, 1.0)
        box_coords = torch.stack([lo, hi], dim=1)
        if not self.training_use_soft_box:
            box_valid = torch.zeros_like(box_valid)

        return point_coords, point_labels, box_coords, box_valid

    def _topk_peak_points(
        self,
        prob: torch.Tensor,
        input_shape: tuple[int, int, int],
        box_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training-time multi-point prompts from local probability peaks.

        This avoids collapsing multi-lesion cases into a single global centroid.
        Peak selection is intentionally detached from gradient flow; the coarse
        branch still receives dense supervision through coarse loss and mask prior.
        """
        B = prob.shape[0]
        D, H, W = input_shape
        device = prob.device
        k = max(int(self.max_proposals), 1)
        kernel = max(int(self.training_nms_kernel), 1)
        if kernel % 2 == 0:
            kernel += 1
        with torch.no_grad():
            score = prob.detach()
            pooled = F.max_pool3d(score, kernel_size=kernel, stride=1, padding=kernel // 2)
            peaks = torch.where(score >= pooled, score, torch.zeros_like(score))
            flat = peaks.flatten(2).squeeze(1)
            values, indices = torch.topk(flat, k=k, dim=1)
            z = indices // (H * W)
            y = (indices % (H * W)) // W
            x = indices % W
            coords = torch.stack([
                z.float() / max(D - 1, 1),
                y.float() / max(H - 1, 1),
                x.float() / max(W - 1, 1),
            ], dim=2).to(device=device, dtype=prob.dtype)
            valid = (values > self.no_prompt_threshold) & box_valid.view(B, 1)
            labels = torch.where(
                valid,
                torch.ones((B, k), dtype=torch.long, device=device),
                -torch.ones((B, k), dtype=torch.long, device=device),
            )
            empty = ~valid.any(dim=1)
            if bool(empty.any().item()):
                coords[empty] = 0.5
        return coords, labels

    @torch.no_grad()
    def _hard_prompts(
        self, coarse_prob: torch.Tensor, input_shape: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Non-differentiable hard prompts via connected components."""
        B = coarse_prob.shape[0]
        D, H, W = input_shape
        device = coarse_prob.device

        prob = coarse_prob
        if prob.shape[2:] != (D, H, W):
            prob = F.interpolate(prob, size=(D, H, W), mode="trilinear", align_corners=False)

        all_coords = []
        all_labels = []
        all_boxes = []
        all_valid = []

        for b in range(B):
            prob_np = prob[b, 0].cpu().numpy()
            binary = prob_np > self.coarse_threshold
            components = self._extract_components(binary, prob_np)

            coords_b = [c["point"] for c in components]
            labels_b = [1] * len(coords_b)
            box_b = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
            valid_b = len(components) > 0
            if valid_b:
                box_b = components[0]["box"]
            elif len(coords_b) == 0:
                max_prob = float(prob[b, 0].max().item())
                if (not self.no_prompt_if_empty) or max_prob > self.no_prompt_threshold:
                    flat_idx = prob[b, 0].flatten().argmax().item()
                    z = flat_idx // (H * W)
                    y = (flat_idx % (H * W)) // W
                    x = flat_idx % W
                    coords_b = [[z / max(D - 1, 1), y / max(H - 1, 1), x / max(W - 1, 1)]]
                    labels_b = [1]
                    box_b = self._point_box(z, y, x, D, H, W)
                    valid_b = True
                else:
                    coords_b = [[0.5, 0.5, 0.5]]
                    labels_b = [-1]

            while len(coords_b) < self.max_proposals:
                coords_b.append([0.5, 0.5, 0.5])
                labels_b.append(-1)

            all_coords.append(torch.tensor(coords_b[: self.max_proposals], dtype=torch.float32, device=device))
            all_labels.append(torch.tensor(labels_b[: self.max_proposals], dtype=torch.long, device=device))
            all_boxes.append(torch.tensor(box_b, dtype=torch.float32, device=device))
            all_valid.append(bool(valid_b))

        point_coords = torch.stack(all_coords, dim=0)
        point_labels = torch.stack(all_labels, dim=0)
        box_coords = torch.stack(all_boxes, dim=0)
        box_valid = torch.tensor(all_valid, dtype=torch.bool, device=device)

        return point_coords, point_labels, box_coords, box_valid

    def _extract_components(self, binary: "np.ndarray", prob_map: "np.ndarray") -> list[dict]:
        """Extract ranked component points and boxes."""
        import numpy as np
        from scipy import ndimage

        labeled, n_components = ndimage.label(binary)
        if n_components == 0:
            return []

        D, H, W = binary.shape
        components = []

        for comp_id in range(1, n_components + 1):
            mask = labeled == comp_id
            if mask.sum() < self.min_component_voxels:
                continue

            coords = np.argwhere(mask)
            if self.point_type == "max_prob":
                comp_prob = prob_map * mask
                flat_idx = comp_prob.argmax()
                z, y, x = np.unravel_index(flat_idx, (D, H, W))
            else:
                z, y, x = coords.mean(axis=0)

            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0)
            margin = self.box_margin_voxels
            z0 = max(int(z0) - margin, 0)
            y0 = max(int(y0) - margin, 0)
            x0 = max(int(x0) - margin, 0)
            z1 = min(int(z1) + margin, D - 1)
            y1 = min(int(y1) + margin, H - 1)
            x1 = min(int(x1) + margin, W - 1)

            score = float(prob_map[mask].max())
            components.append({
                "point": [
                    float(z) / max(D - 1, 1),
                    float(y) / max(H - 1, 1),
                    float(x) / max(W - 1, 1),
                ],
                "box": [
                    [z0 / max(D - 1, 1), y0 / max(H - 1, 1), x0 / max(W - 1, 1)],
                    [z1 / max(D - 1, 1), y1 / max(H - 1, 1), x1 / max(W - 1, 1)],
                ],
                "score": score,
            })

        components.sort(key=lambda item: -item["score"])
        return components[: self.max_proposals]

    def _point_box(self, z: int, y: int, x: int, D: int, H: int, W: int) -> list[list[float]]:
        margin = self.box_margin_voxels
        z0, z1 = max(z - margin, 0), min(z + margin, D - 1)
        y0, y1 = max(y - margin, 0), min(y + margin, H - 1)
        x0, x1 = max(x - margin, 0), min(x + margin, W - 1)
        return [
            [z0 / max(D - 1, 1), y0 / max(H - 1, 1), x0 / max(W - 1, 1)],
            [z1 / max(D - 1, 1), y1 / max(H - 1, 1), x1 / max(W - 1, 1)],
        ]
