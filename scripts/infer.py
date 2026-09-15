"""Inference for the CyclopsNet-inspired VLMOD engineering baseline.

This script reports predictions only. Thresholds are engineering choices and are
not claimed to be official CyclopsNet settings.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import CyclopsNetInspired
from src.preprocessing import build_image_transform
from src.dataset import load_vlmod_json

ID_TO_CLASS = {0: "car", 1: "van", 2: "truck", 3: "bus"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/stage11_controlled/best_val.pt")
    p.add_argument("--threshold", type=float, default=0.30, help="Engineering combined score threshold")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def device(name):
    if name == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    if name == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    model = CyclopsNetInspired(
        num_candidates=int(cfg.get("num_candidates", 16)),
        num_classes=int(cfg.get("num_classes", 4)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run(image_path, query, model, dev, threshold, top_k):
    transform = build_image_transform(224)
    with Image.open(image_path) as im:
        image = transform(im.convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(image, [query])
    objectness = torch.sigmoid(out["objectness"][0])
    similarity = out["similarity"][0]
    cls_prob = out["class_logits"][0].softmax(-1)
    cls_score, cls_id = cls_prob.max(-1)
    # Simple transparent ranking: average of objectness, class confidence,
    # and normalized similarity. This is an engineering baseline only.
    sim_norm = (similarity + 1.0) / 2.0
    score = (objectness + cls_score + sim_norm) / 3.0
    keep = score >= threshold
    indices = torch.where(keep)[0]
    indices = indices[torch.argsort(score[indices], descending=True)] if len(indices) else torch.argsort(score, descending=True)[:top_k]
    indices = indices[:top_k]
    preds = []
    for i in indices.tolist():
        box = out["boxes_2d"][0, i].detach().cpu().tolist()
        # Model box format is normalized cx,cy,w,h.
        cx, cy, w, h = box
        box_xyxy = [max(0.0, cx-w/2), max(0.0, cy-h/2), min(1.0, cx+w/2), min(1.0, cy+h/2)]
        preds.append({
            "candidate_index": i,
            "score": float(score[i]),
            "objectness": float(objectness[i]),
            "class": ID_TO_CLASS.get(int(cls_id[i]), str(int(cls_id[i]))),
            "class_confidence": float(cls_score[i]),
            "query_similarity": float(similarity[i]),
            "bbox_2d_normalized_xyxy": box_xyxy,
            "dimensions": out["dimensions"][0, i].cpu().tolist(),
            "location": out["location"][0, i].cpu().tolist(),
            "rotation": float(out["rotation"][0, i, 0]),
            "depth": float(out["depth"][0, i, 0]),
        })
    return {"image": str(image_path), "query": query, "threshold": threshold, "predictions": preds}


def main():
    args = parse_args(); dev = device(args.device)
    if not args.image.is_file(): raise FileNotFoundError(args.image)
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    model, ckpt = load_model(args.checkpoint, dev)
    result = run(args.image, args.query, model, dev, args.threshold, args.top_k)
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_epoch"] = ckpt.get("epoch")
    result["device"] = str(dev)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved_json: {args.output}")

if __name__ == "__main__": main()
