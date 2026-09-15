"""Preprocessing and target formatting for the reconstructed VLMOD baseline.

This file does not define a neural network. It converts raw dataset records into
model-ready image tensors, query strings, and explicit variable-length targets.
"""
from __future__ import annotations
from typing import Any, Dict, List, Sequence
import torch
from torchvision import transforms

# These are a baseline representation, not claimed to be the professor's model.
CLASS_TO_ID = {"car": 0, "van": 1, "truck": 2, "bus": 3}


def build_image_transform(image_size: int = 224):
    """Resize RGB images and normalize them for a pretrained ImageNet encoder."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def extract_query_examples(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand one image record into one example per natural-language query."""
    examples: List[Dict[str, Any]] = []
    for ann in sample["query_annotations"]:
        labels = ann["labels"]
        boxes = torch.tensor([x["bbox_2d"] for x in labels], dtype=torch.float32)
        classes = torch.tensor([CLASS_TO_ID[x["class_name"]] for x in labels], dtype=torch.long)
        dimensions = torch.tensor([x["dimensions"] for x in labels], dtype=torch.float32)
        location = torch.tensor([x["location"] for x in labels], dtype=torch.float32)
        rotation = torch.tensor([x["rotation"] for x in labels], dtype=torch.float32)
        instance_ids = torch.tensor([x["instance_id"] for x in labels], dtype=torch.long)

        examples.append({
            "query": ann["public_description"],
            "properties": ann["public_properties"],
            "target": {
                "labels": classes,
                "boxes_2d": boxes,
                "dimensions": dimensions,
                "location": location,
                "rotation": rotation,
                "instance_ids": instance_ids,
            },
        })
    return examples


def vlmod_query_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep variable-length targets as a list while stacking image tensors."""
    images = torch.stack([item["image"] for item in batch])
    return {
        "images": images,
        "queries": [item["query"] for item in batch],
        "properties": [item["properties"] for item in batch],
        "targets": [item["target"] for item in batch],
        "calibration": [item.get("calibration") for item in batch],
        "denorm": [item.get("denorm") for item in batch],
    }
