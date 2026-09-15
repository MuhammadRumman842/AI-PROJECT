"""Tiny Stage-9 training sanity check for the VLMOD scaffold.

IMPORTANT: This is NOT full training and does not establish model accuracy.
It verifies only that the engineering baseline can perform optimizer steps,
change parameters, save a checkpoint, and reload it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import VLMODDataset
from src.model import CyclopsNetInspired
from src.preprocessing import build_image_transform, extract_query_examples
from src.losses import compute_loss


def main() -> None:
    data_root = ROOT.parent
    image_dir = data_root / "IMG" / "train"
    json_dir = data_root / "FILES" / "train"
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "stage9_sanity.pt"

    dataset = VLMODDataset(
        image_dir=image_dir,
        json_dir=json_dir,
        transform=build_image_transform(224),
        require_exact_pair=True,
    )

    # The raw dataset item contains multiple queries. Flatten only enough data
    # for this tiny sanity check; no large-scale training is attempted.
    examples = []
    for idx in range(min(len(dataset), 2)):
        sample = dataset[idx]
        for query_example in extract_query_examples(sample):
            examples.append({
                "image": sample["image"],
                "query": query_example["query"],
                "target": query_example["target"],
            })
            if len(examples) >= 2:
                break
        if len(examples) >= 2:
            break

    if not examples:
        raise RuntimeError("No query examples available for sanity training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyclopsNetInspired(num_candidates=16, num_classes=4).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Keep this intentionally tiny: exactly one or two optimizer steps.
    max_steps = min(2, len(examples))
    print(f"device: {device}")
    print(f"dataset_pairs: {len(dataset)}")
    print(f"sanity_examples: {len(examples)}")

    before = next(p for p in model.parameters() if p.requires_grad).detach().clone()

    for step in range(max_steps):
        item = examples[step]
        image = item["image"].unsqueeze(0).to(device)
        query = [item["query"]]
        target = item["target"]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(image, query)
        prediction_one = {k: v[0] for k, v in prediction.items() if isinstance(v, torch.Tensor) and v.ndim >= 1}
        total, components, matches = compute_loss(prediction_one, target)
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at step {step}: {total.item()}")

        total_before = float(total.detach().cpu())
        component_values = {k: float(v.detach().cpu()) for k, v in components.items()}
        total.backward()

        finite_grad = all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.requires_grad
        )
        if not finite_grad:
            raise RuntimeError(f"Non-finite gradients at step {step}")

        optimizer.step()
        print(f"step {step + 1}: total_loss={total_before:.6f} components={component_values} matches={len(matches.pred_indices)}")

    after = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    parameter_delta = float((after - before).abs().sum().cpu())
    if parameter_delta <= 0.0:
        raise RuntimeError("No trainable parameter changed after optimizer steps.")
    print(f"parameter_delta_l1: {parameter_delta:.8f}")

    checkpoint = {
        "stage": 9,
        "note": "Tiny sanity checkpoint; not a trained model.",
        "config": {
            "num_candidates": 16,
            "num_classes": 4,
            "image_size": 224,
            "learning_rate": 1e-4,
            "steps": max_steps,
            "device": str(device),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model2 = CyclopsNetInspired(num_candidates=16, num_classes=4).to(device)
    optimizer2 = Adam(model2.parameters(), lr=1e-4)
    model2.load_state_dict(loaded["model_state_dict"])
    optimizer2.load_state_dict(loaded["optimizer_state_dict"])
    print(f"checkpoint: {checkpoint_path}")
    print("checkpoint reload: OK")
    print("STAGE 9 SANITY TRAINING: OK")
    print("This test does not measure accuracy and does not constitute full training.")


if __name__ == "__main__":
    main()
