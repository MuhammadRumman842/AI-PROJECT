"""Forward-pass smoke test on one real supplied VLMOD sample."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset import VLMODDataset
from src.preprocessing import build_image_transform, extract_query_examples
from src.model import CyclopsNetInspired

root = Path(__file__).resolve().parents[1]
data_root = root.parent / "IMG"
json_root = root.parent / "FILES"
if not data_root.exists():
    data_root = root.parent.parent / "IMG"
    json_root = root.parent.parent / "FILES"

# Keep this smoke test small and deterministic; it is not training.
torch.manual_seed(0)
ds = VLMODDataset(
    data_root / "train",
    json_root / "train",
    transform=build_image_transform(224),
)
sample = ds[0]
examples = extract_query_examples(sample)
assert examples, "Sample must contain at least one query"

query = examples[0]["query"]
model = CyclopsNetInspired(num_candidates=16, latent_dim=64)
model.eval()

with torch.no_grad():
    out = model(sample["image"].unsqueeze(0), [query])

expected = {
    "objectness": (1, 16),
    "class_logits": (1, 16, 4),
    "boxes_2d": (1, 16, 4),
    "dimensions": (1, 16, 3),
    "location": (1, 16, 3),
    "rotation": (1, 16, 1),
    "depth": (1, 16, 1),
    "object_mean": (1, 16, 64),
    "object_logvar": (1, 16, 64),
    "text_mean": (1, 16, 64),
    "text_logvar": (1, 16, 64),
    "fused_mean": (1, 16, 64),
    "fused_logvar": (1, 16, 64),
    "similarity": (1, 16),
}

for key, shape in expected.items():
    got = tuple(out[key].shape)
    assert got == shape, f"{key}: expected {shape}, got {got}"
    assert torch.isfinite(out[key]).all(), f"{key} contains non-finite values"

print("dataset_pairs", len(ds))
print("query", repr(query[:120]))
for key, shape in expected.items():
    print(f"{key:16s} {shape}")
print("FORWARD PASS: OK")
