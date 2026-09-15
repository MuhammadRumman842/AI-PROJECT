"""Stage-12 inference smoke test on one real supplied sample."""
from pathlib import Path
import sys, torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.dataset import VLMODDataset
from src.preprocessing import build_image_transform, extract_query_examples
from src.model import CyclopsNetInspired
from scripts.infer import load_model, run


def main():
    base = VLMODDataset(ROOT.parent / "IMG" / "train", ROOT.parent / "FILES" / "train", transform=build_image_transform(224), require_exact_pair=True)
    sample = base[0]
    q = extract_query_examples(sample)[0]
    ckpt = ROOT / "checkpoints/stage11_controlled/best_val.pt"
    dev = torch.device("cpu")
    model, _ = load_model(ckpt, dev)
    result = run(Path(sample["image_path"]), q["query"], model, dev, 0.30, 5)
    assert result["predictions"], "Inference returned no candidates"
    for p in result["predictions"]:
        for key in ("score", "objectness", "class_confidence", "query_similarity", "rotation", "depth"):
            assert torch.isfinite(torch.tensor(p[key])), f"non-finite {key}"
        assert all(torch.isfinite(torch.tensor(p[k])).all() for k in ("dimensions", "location", "bbox_2d_normalized_xyxy"))
    print("image:", sample["image_path"])
    print("query:", q["query"][:140])
    print("predictions:", len(result["predictions"]))
    print("top_prediction:", result["predictions"][0])
    print("STAGE 12 INFERENCE: OK")

if __name__ == "__main__": main()
