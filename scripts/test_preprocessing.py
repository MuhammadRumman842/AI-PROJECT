from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import VLMODDataset
from src.preprocessing import build_image_transform, extract_query_examples

root = Path(__file__).resolve().parents[1]
data_root = root.parent / "IMG"
json_root = root.parent / "FILES"
# Fall back to paths if project was extracted with sibling data folders.
if not data_root.exists():
    data_root = root.parent.parent / "IMG"
    json_root = root.parent.parent / "FILES"

ds = VLMODDataset(data_root / "train", json_root / "train", transform=build_image_transform(224))
sample = ds[0]
examples = extract_query_examples(sample)
print("pairs", len(ds))
print("image", tuple(sample["image"].shape))
print("queries", len(examples))
for i, ex in enumerate(examples):
    print(i, repr(ex["query"][:100]), "objects", len(ex["target"]["labels"]))
    for k, v in ex["target"].items(): print(" ", k, tuple(v.shape))
