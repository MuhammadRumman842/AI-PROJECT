"""Losses and candidate-to-target matching for the VLMOD scaffold.

IMPORTANT:
    This module is an engineering baseline, not the authors' exact CyclopsNet
    training objective. The released/verified materials do not expose the full
    original loss or matching implementation.

Design used here:
    1. Match each ground-truth object to at most one candidate using a simple
       greedy cost based on 2D box distance + class mismatch + 3D location/
       dimension distance.
    2. Train matched candidates on class, 2D box, dimensions, location and
       rotation.
    3. Train objectness for matched vs unmatched candidates.
    4. Encourage query-object similarity for matched candidates with a margin
       against unmatched candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MatchResult:
    pred_indices: torch.Tensor
    target_indices: torch.Tensor


def _pairwise_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, None, :] - b[None, :, :]).abs().mean(dim=-1)


def match_candidates(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    image_size: Tuple[int, int] = (1920, 1080),
) -> MatchResult:
    """Greedily match predicted slots to real objects.

    Prediction boxes are normalized [cx, cy, w, h]. Dataset boxes are expected
    in pixel [x1, y1, x2, y2] format.
    """
    n_pred = prediction["boxes_2d"].shape[0]
    n_tgt = target["labels"].shape[0]
    if n_pred == 0 or n_tgt == 0:
        device = prediction["boxes_2d"].device
        return MatchResult(
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    width, height = float(image_size[0]), float(image_size[1])
    boxes = target["boxes_2d"].to(prediction["boxes_2d"].device)
    # [x1,y1,x2,y2] pixels -> normalized [cx,cy,w,h].
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    tgt_boxes = torch.stack([
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        (x2 - x1).clamp_min(0) / width,
        (y2 - y1).clamp_min(0) / height,
    ], dim=-1)

    pred_boxes = prediction["boxes_2d"]
    box_cost = _pairwise_l1(pred_boxes, tgt_boxes)

    pred_classes = prediction["class_logits"].argmax(dim=-1)
    tgt_classes = target["labels"].to(pred_classes.device)
    class_cost = (pred_classes[:, None] != tgt_classes[None, :]).float()

    pred_loc = prediction["location"]
    tgt_loc = target["location"].to(pred_loc.device)
    loc_cost = _pairwise_l1(pred_loc, tgt_loc)

    pred_dim = prediction["dimensions"]
    tgt_dim = target["dimensions"].to(pred_dim.device)
    dim_cost = _pairwise_l1(pred_dim, tgt_dim)

    cost = 2.0 * box_cost + 1.0 * class_cost + 0.25 * loc_cost + 0.25 * dim_cost

    # Simple deterministic one-to-one greedy assignment. This is intentionally
    # not claimed to be the paper's matcher.
    flat_order = torch.argsort(cost.flatten())
    used_pred = set()
    used_tgt = set()
    pairs: List[Tuple[int, int]] = []
    for flat_idx in flat_order.tolist():
        p = flat_idx // n_tgt
        t = flat_idx % n_tgt
        if p in used_pred or t in used_tgt:
            continue
        used_pred.add(p)
        used_tgt.add(t)
        pairs.append((p, t))
        if len(pairs) == min(n_pred, n_tgt):
            break

    pairs.sort(key=lambda x: x[0])
    pred_idx = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=cost.device)
    tgt_idx = torch.tensor([t for _, t in pairs], dtype=torch.long, device=cost.device)
    return MatchResult(pred_idx, tgt_idx)


def _safe_mean(value: torch.Tensor) -> torch.Tensor:
    return value.mean() if value.numel() else value.new_zeros(())


def compute_loss(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    image_size: Tuple[int, int] = (1920, 1080),
    similarity_margin: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], MatchResult]:
    """Compute the baseline multi-task loss for one query example."""
    device = prediction["objectness"].device
    target = {k: v.to(device) for k, v in target.items()}
    matches = match_candidates(prediction, target, image_size=image_size)

    n_pred = prediction["objectness"].shape[0]
    objectness_target = torch.zeros(n_pred, device=device)
    objectness_target[matches.pred_indices] = 1.0
    loss_objectness = F.binary_cross_entropy_with_logits(
        prediction["objectness"], objectness_target
    )

    if len(matches.pred_indices):
        p = matches.pred_indices
        t = matches.target_indices
        loss_class = F.cross_entropy(
            prediction["class_logits"][p], target["labels"][t]
        )

        width, height = float(image_size[0]), float(image_size[1])
        boxes = target["boxes_2d"][t]
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        tgt_boxes = torch.stack([
            ((x1 + x2) * 0.5) / width,
            ((y1 + y2) * 0.5) / height,
            (x2 - x1).clamp_min(0) / width,
            (y2 - y1).clamp_min(0) / height,
        ], dim=-1)
        loss_box = F.smooth_l1_loss(prediction["boxes_2d"][p], tgt_boxes)
        loss_dimensions = F.smooth_l1_loss(
            prediction["dimensions"][p], target["dimensions"][t]
        )
        loss_location = F.smooth_l1_loss(
            prediction["location"][p], target["location"][t]
        )
        loss_rotation = F.smooth_l1_loss(
            prediction["rotation"][p, 0], target["rotation"][t]
        )

        # Matched candidates should score above unmatched candidates for the
        # current language query. Cosine similarity is already bounded [-1, 1].
        sim = prediction["similarity"]
        unmatched = torch.ones(n_pred, dtype=torch.bool, device=device)
        unmatched[p] = False
        if unmatched.any():
            positive = sim[p][:, None]
            negative = sim[unmatched][None, :]
            loss_similarity = F.relu(similarity_margin - positive + negative).mean()
        else:
            loss_similarity = sim.new_zeros(())
    else:
        loss_class = prediction["class_logits"].sum() * 0.0
        loss_box = prediction["boxes_2d"].sum() * 0.0
        loss_dimensions = prediction["dimensions"].sum() * 0.0
        loss_location = prediction["location"].sum() * 0.0
        loss_rotation = prediction["rotation"].sum() * 0.0
        loss_similarity = prediction["similarity"].sum() * 0.0

    # These weights are explicit engineering choices, not paper hyperparameters.
    components = {
        "objectness": loss_objectness,
        "class": loss_class,
        "box_2d": loss_box,
        "dimensions": loss_dimensions,
        "location": loss_location,
        "rotation": loss_rotation,
        "similarity": loss_similarity,
    }
    total = (
        1.0 * loss_objectness
        + 1.0 * loss_class
        + 2.0 * loss_box
        + 1.0 * loss_dimensions
        + 1.0 * loss_location
        + 0.5 * loss_rotation
        + 1.0 * loss_similarity
    )
    return total, components, matches
