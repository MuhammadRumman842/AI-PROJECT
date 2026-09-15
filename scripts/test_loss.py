"""Unit/smoke test for the baseline VLMOD loss on a real sample."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset import VLMODDataset
from src.preprocessing import build_image_transform, extract_query_examples
from src.model import CyclopsNetInspired
from src.losses import compute_loss

root = Path(__file__).resolve().parents[1]
data_root = root.parent / "IMG"
json_root = root.parent / "FILES"
if not data_root.exists():
    data_root = root.parent.parent / "IMG"
    json_root = root.parent.parent / "FILES"

torch.manual_seed(0)
ds = VLMODDataset(
    data_root / "train",
    json_root / "train",
    transform=build_image_transform(224),
)
sample = ds[0]
example = extract_query_examples(sample)[0]

model = CyclopsNetInspired(num_candidates=16, latent_dim=64)
model.train()

prediction = model(sample["image"].unsqueeze(0), [example["query"]])
# Remove batch dimension for the single-example loss API.
prediction = {k: v[0] for k, v in prediction.items()}

total, components, matches = compute_loss(
    prediction,
    example["target"],
    image_size=(1920, 1080),
)

assert torch.isfinite(total), "total loss is not finite"
for name, value in components.items():
    assert torch.isfinite(value), f"{name} loss is not finite"
assert len(matches.pred_indices) == len(example["target"]["labels"])
assert len(set(matches.pred_indices.tolist())) == len(matches.pred_indices)
assert len(set(matches.target_indices.tolist())) == len(matches.target_indices)

# Backward pass verifies the loss is connected to model parameters.
total.backward()
grad_ok = any(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.parameters()
    if p.requires_grad
)
assert grad_ok, "no finite gradients reached model parameters"

print("dataset_pairs", len(ds))
print("target_objects", len(example["target"]["labels"]))
print("matched_pairs", len(matches.pred_indices))
print("matched_pred_indices", matches.pred_indices.tolist())
for name, value in components.items():
    print(f"loss_{name:12s} {value.item():.6f}")
print(f"loss_total       {total.item():.6f}")
print("LOSS + BACKWARD: OK")
