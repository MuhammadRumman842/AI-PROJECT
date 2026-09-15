"""Conservative training/validation pipeline for the VLMOD engineering baseline.

IMPORTANT: This trains the reconstructed CyclopsNet-inspired scaffold, not the
authors' exact CyclopsNet implementation. Metrics are validation diagnostics
on the supplied exact-pair subset; they are not official challenge test scores.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import VLMODDataset
from src.losses import compute_loss
from src.model import CyclopsNetInspired
from src.preprocessing import build_image_transform, extract_query_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-train-examples", type=int, default=0)
    p.add_argument("--max-val-examples", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


class QueryExampleDataset(Dataset):
    def __init__(self, base: VLMODDataset, indices: List[int]):
        self.examples: List[Dict[str, Any]] = []
        for idx in indices:
            sample = base[idx]
            for q in extract_query_examples(sample):
                self.examples.append({
                    "image": sample["image"],
                    "query": q["query"],
                    "target": q["target"],
                    "image_path": sample["image_path"],
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([x["image"] for x in batch]),
        "queries": [x["query"] for x in batch],
        "targets": [x["target"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
    }


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices(n: int, val_ratio: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    return indices[n_val:], indices[:n_val]


def update_metrics(store: Dict[str, float], count: int, components: Dict[str, torch.Tensor], matches, prediction, target):
    for name, value in components.items():
        store[f"loss_{name}"] = store.get(f"loss_{name}", 0.0) + float(value.detach().cpu())
    store["total_loss"] = store.get("total_loss", 0.0) + float(sum(components.values()).detach().cpu())
    p = matches.pred_indices
    t = matches.target_indices
    if len(p):
        pred_cls = prediction["class_logits"][p].argmax(-1)
        store["matched_class_correct"] = store.get("matched_class_correct", 0.0) + float((pred_cls == target["labels"][t].to(pred_cls.device)).sum().item())
        store["matched_count"] = store.get("matched_count", 0.0) + float(len(p))
        store["location_mae"] = store.get("location_mae", 0.0) + float((prediction["location"][p] - target["location"][t].to(prediction["location"].device)).abs().mean().item())
        store["dimension_mae"] = store.get("dimension_mae", 0.0) + float((prediction["dimensions"][p] - target["dimensions"][t].to(prediction["dimensions"].device)).abs().mean().item())
        store["rotation_mae"] = store.get("rotation_mae", 0.0) + float((prediction["rotation"][p, 0] - target["rotation"][t].to(prediction["rotation"].device)).abs().mean().item())
        store["box_mae"] = store.get("box_mae", 0.0) + float((prediction["boxes_2d"][p] - _target_boxes(target["boxes_2d"][t], prediction["boxes_2d"].device)).abs().mean().item())
    store["steps"] = store.get("steps", 0.0) + count


def _target_boxes(boxes: torch.Tensor, device: torch.device) -> torch.Tensor:
    boxes = boxes.to(device)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([((x1+x2)*0.5)/1920.0, ((y1+y2)*0.5)/1080.0, (x2-x1).clamp_min(0)/1920.0, (y2-y1).clamp_min(0)/1080.0], -1)


def finalize_metrics(store: Dict[str, float]) -> Dict[str, float]:
    steps = max(1.0, store.get("steps", 1.0))
    out = {k: v / steps for k, v in store.items() if k not in {"matched_count", "matched_class_correct", "steps"}}
    matched = store.get("matched_count", 0.0)
    out["matched_class_accuracy"] = store.get("matched_class_correct", 0.0) / max(1.0, matched)
    out["matched_objects"] = matched
    return out


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    store: Dict[str, float] = {}
    for batch in loader:
        images = batch["images"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(images, batch["queries"])
            total_batch = images.new_zeros(())
            batch_components = {}
            for i, target in enumerate(batch["targets"]):
                pred_one = {k: v[i] for k, v in prediction.items() if isinstance(v, torch.Tensor) and v.ndim >= 1}
                total, components, matches = compute_loss(pred_one, target)
                total_batch = total_batch + total
                for k, v in components.items():
                    batch_components[k] = batch_components.get(k, images.new_zeros(())) + v
                update_metrics(store, 1, components, matches, pred_one, target)
            total_batch = total_batch / len(batch["targets"])
            if training:
                if not torch.isfinite(total_batch):
                    raise RuntimeError("Non-finite training loss")
                total_batch.backward()
                optimizer.step()
    return finalize_metrics(store)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    data_root = ROOT.parent
    base = VLMODDataset(data_root / "IMG" / "train", data_root / "FILES" / "train", transform=build_image_transform(args.image_size), require_exact_pair=True)
    train_idx, val_idx = split_indices(len(base), args.val_ratio, args.seed)
    train_ds = QueryExampleDataset(base, train_idx)
    val_ds = QueryExampleDataset(base, val_idx)
    if args.max_train_examples:
        train_ds.examples = train_ds.examples[:args.max_train_examples]
    if args.max_val_examples:
        val_ds.examples = val_ds.examples[:args.max_val_examples]
    if not train_ds or not val_ds:
        raise RuntimeError(f"Need non-empty train/validation sets: train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = CyclopsNetInspired(num_candidates=args.num_candidates, num_classes=args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"resumed_from: {args.resume} epoch={start_epoch}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)
    config["resume"] = str(args.resume) if args.resume else None
    print(json.dumps({"device": str(device), "train_pairs": len(train_idx), "val_pairs": len(val_idx), "train_queries": len(train_ds), "val_queries": len(val_ds), "config": config}, indent=2, default=str))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        print(f"epoch {epoch + 1}: train_total_loss={train_metrics['total_loss']:.6f} val_total_loss={val_metrics['total_loss']:.6f} val_cls_acc={val_metrics['matched_class_accuracy']:.4f} val_loc_mae={val_metrics.get('location_mae', float('nan')):.6f}")

        state = {
            "stage": 10,
            "epoch": epoch,
            "best_val_loss": min(best_val, val_metrics["total_loss"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "note": "Engineering baseline; not authors' exact CyclopsNet and not official challenge test performance.",
        }
        torch.save(state, args.checkpoint_dir / "latest.pt")
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            state["best_val_loss"] = best_val
            torch.save(state, args.checkpoint_dir / "best_val.pt")
            print(f"best_checkpoint: {args.checkpoint_dir / 'best_val.pt'}")
    print("STAGE 10 TRAINING PIPELINE: OK")


if __name__ == "__main__":
    main()
