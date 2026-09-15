"""CyclopsNet-inspired engineering scaffold for VLMOD Track B.

IMPORTANT:
    This is NOT a reproduction of the authors' exact CyclopsNet implementation.
    The supplied repository is partial, so only architecture concepts verified
    from the CVPR 2025 paper are represented here. Exact SPVE/DAF equations,
    backbones, losses, and hyperparameters remain unverified.

Verified concepts represented in this scaffold:
    1. Image -> candidate 3D objects (stage 1).
    2. State-aware visual representation (SPVE-inspired).
    3. Text + object probabilistic representations.
    4. Uncertainty-aware multimodal fusion (DAF-inspired).
    5. Object-text matching (stage 2).

Reconstructed engineering choices:
    - ResNet18-like image encoder with randomly initialized weights.
    - Fixed K candidate slots instead of the authors' exact detector.
    - GRU text encoder over a small deterministic hashed vocabulary.
    - Gaussian mean/log-variance vectors for object/text/fused features.
    - Precision-weighted Gaussian fusion as a simple DAF-inspired operation.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18
except Exception as exc:  # pragma: no cover
    resnet18 = None
    _TORCHVISION_ERROR = exc
else:
    _TORCHVISION_ERROR = None


class HashTextEncoder(nn.Module):
    """Small dependency-free text encoder for shape testing and prototyping.

    It is intentionally not claimed to match the paper's tokenizer/text model.
    """

    def __init__(self, vocab_size: int = 4096, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def _token_ids(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        token_lists: List[List[int]] = []
        for text in texts:
            words = text.lower().split()
            ids = []
            for word in words[:64]:
                digest = hashlib.sha1(word.encode("utf-8")).digest()
                value = int.from_bytes(digest[:4], "little")
                ids.append(1 + value % (self.vocab_size - 1))
            token_lists.append(ids or [1])
        max_len = max(len(x) for x in token_lists)
        out = torch.zeros(len(token_lists), max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(token_lists):
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return out

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        ids = self._token_ids(texts, device)
        x = self.embedding(ids)
        _, h = self.gru(x)
        return h[-1]


class GaussianProjection(nn.Module):
    """Map features to Gaussian mean and log-variance parameters."""

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mean = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        return self.mean(x), self.logvar(x).clamp(-8.0, 8.0)


def precision_fuse(
    mean_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mean_b: torch.Tensor,
    logvar_b: torch.Tensor,
):
    """Simple product-of-Gaussians fusion used only as an engineering baseline."""
    var_a = logvar_a.exp().clamp_min(1e-6)
    var_b = logvar_b.exp().clamp_min(1e-6)
    precision_a = var_a.reciprocal()
    precision_b = var_b.reciprocal()
    fused_var = (precision_a + precision_b).reciprocal()
    fused_mean = fused_var * (precision_a * mean_a + precision_b * mean_b)
    return fused_mean, fused_var.log()


class CyclopsNetInspired(nn.Module):
    """Two-stage CyclopsNet-inspired model scaffold.

    Input:
        images: [B, 3, H, W]
        queries: list[str] of length B

    Output:
        candidate 3D object predictions and object/query similarity scores.
    """

    def __init__(self, num_candidates: int = 16, latent_dim: int = 256, num_classes: int = 4):
        super().__init__()
        if resnet18 is None:
            raise RuntimeError(f"torchvision is unavailable: {_TORCHVISION_ERROR}")

        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Reconstructed engineering backbone; exact paper backbone is unverified.
        backbone = resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_dim = 512

        # Stage-1 candidate detector: class + 2D box + 3D state.
        self.candidate_head = nn.Linear(self.image_dim, num_candidates * 5)
        # 5 = objectness + 4 class logits. Bounding boxes use a separate head.
        self.bbox_head = nn.Linear(self.image_dim, num_candidates * 4)
        self.state_head = nn.Linear(self.image_dim, num_candidates * 8)
        # 8 state values = dimensions(3), location(3), rotation(1), depth(1).

        # SPVE-inspired state prompt: encode predicted physical state into feature space.
        self.state_prompt = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_dim),
        )

        # Candidate object Gaussian representation.
        self.object_gaussian = GaussianProjection(self.image_dim * 2, latent_dim)

        # Text side; exact paper text encoder is unverified.
        self.text_encoder = HashTextEncoder(hidden_dim=256)
        self.text_gaussian = GaussianProjection(256, latent_dim)

        # DAF-inspired projection after uncertainty-aware fusion.
        self.fused_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _detect_candidates(self, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.image_pool(image_features).flatten(1)
        b = pooled.shape[0]
        raw = self.candidate_head(pooled).view(b, self.num_candidates, 5)
        state = self.state_head(pooled).view(b, self.num_candidates, 8)

        objectness = raw[..., 0]
        class_logits = raw[..., 1:5]
        # Normalize bbox to [0,1] through sigmoid: cx, cy, w, h.
        bbox_head = self._bbox_head(pooled).view(b, self.num_candidates, 4)
        boxes_2d = bbox_head.sigmoid()

        state = state.clone()
        dimensions = F.softplus(state[..., 0:3])
        location = state[..., 3:6]
        rotation = state[..., 6:7]
        depth = F.softplus(state[..., 7:8])

        return {
            "objectness": objectness,
            "class_logits": class_logits,
            "boxes_2d": boxes_2d,
            "dimensions": dimensions,
            "location": location,
            "rotation": rotation,
            "depth": depth,
            "pooled": pooled,
        }

    def _bbox_head(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.bbox_head(pooled)

    def forward(self, images: torch.Tensor, queries: Sequence[str]) -> Dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        if len(queries) != images.shape[0]:
            raise ValueError("Number of queries must equal image batch size")

        feature_map = self.image_encoder(images)
        candidates = self._detect_candidates(feature_map)

        # State-prompt visual encoding (SPVE-inspired).
        state_vector = torch.cat([
            candidates["dimensions"],
            candidates["location"],
            candidates["rotation"],
            candidates["depth"],
        ], dim=-1)
        state_prompt = self.state_prompt(state_vector)
        pooled = candidates["pooled"].unsqueeze(1).expand(-1, self.num_candidates, -1)
        visual_state = pooled + state_prompt
        object_mean, object_logvar = self.object_gaussian(torch.cat([visual_state, state_prompt], dim=-1))

        text_feature = self.text_encoder(queries, images.device)
        text_mean, text_logvar = self.text_gaussian(text_feature)
        text_mean = text_mean.unsqueeze(1).expand(-1, self.num_candidates, -1)
        text_logvar = text_logvar.unsqueeze(1).expand(-1, self.num_candidates, -1)

        # DAF-inspired uncertainty-aware fusion.
        fused_mean, fused_logvar = precision_fuse(
            object_mean, object_logvar, text_mean, text_logvar
        )
        fused = self.fused_projection(fused_mean)

        # Similarity is cosine similarity between the fused multimodal representation
        # and each candidate object representation.
        similarity = F.cosine_similarity(fused, object_mean, dim=-1)

        return {
            **{k: v for k, v in candidates.items() if k != "pooled"},
            "object_mean": object_mean,
            "object_logvar": object_logvar,
            "text_mean": text_mean,
            "text_logvar": text_logvar,
            "fused_mean": fused_mean,
            "fused_logvar": fused_logvar,
            "similarity": similarity,
        }

