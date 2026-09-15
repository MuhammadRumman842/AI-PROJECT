"""PyTorch dataset for the VLMOD Track B data.

The released dataset stores one JSON file per image/query record.  Each JSON
contains a list of referring-expression annotations plus calibration data.
This module deliberately uses exact filename-stem matching; it never guesses
an image for a JSON record.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


OBJECT_CLASSES = {"car", "van", "truck", "bus"}


def parse_label_3(value: str) -> Dict[str, Any]:
    """Parse one label_3 string safely with ast.literal_eval()."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) < 16:
        raise ValueError(f"Unexpected label_3 record: {value!r}")

    return {
        "class_name": parsed[0],
        "instance_id": int(parsed[1]),
        "group_id": int(parsed[2]),
        "depth": float(parsed[3]),
        "bbox_2d": [float(v) for v in parsed[4:8]],
        "dimensions": [float(v) for v in parsed[8:11]],
        "location": [float(v) for v in parsed[11:14]],
        "rotation": float(parsed[14]),
        "appearance": parsed[15],
    }


def load_vlmod_json(path: Path) -> Dict[str, Any]:
    """Load and normalize one VLMOD JSON file."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"Unexpected VLMOD JSON structure: {path}")

    annotations_raw = raw[0]
    metadata_raw = raw[1]
    if not isinstance(annotations_raw, list) or not isinstance(metadata_raw, list):
        raise ValueError(f"Unexpected VLMOD JSON structure: {path}")

    annotations: List[Dict[str, Any]] = []
    for ann in annotations_raw:
        if not isinstance(ann, dict):
            raise ValueError(f"Invalid annotation in {path}")
        labels = [parse_label_3(x) for x in ann.get("label_3", [])]
        annotations.append({
            "ann_id": int(ann.get("ann_id", len(annotations))),
            "public_properties": ann.get("public_properties", []),
            "public_description": ann.get("public_description", ""),
            "labels": labels,
        })

    metadata = metadata_raw[0] if metadata_raw else {}
    calib = metadata.get("calib")
    denorm = metadata.get("denorm")

    calibration = None
    if calib:
        values = [float(x) for x in calib.split(",")]
        if len(values) != 12:
            raise ValueError(f"Calibration matrix must contain 12 values: {path}")
        calibration = torch.tensor(values, dtype=torch.float32).reshape(3, 4)

    denorm_tensor = None
    if denorm:
        values = [float(x) for x in denorm.split(",")]
        if len(values) != 4:
            raise ValueError(f"Denorm must contain 4 values: {path}")
        denorm_tensor = torch.tensor(values, dtype=torch.float32)

    return {
        "annotations": annotations,
        "calibration": calibration,
        "denorm": denorm_tensor,
    }


def exact_stem(path: Path) -> str:
    return path.stem


class VLMODDataset(Dataset):
    """One item per JSON/image pair.

    Args:
        image_dir: Directory containing RGB images.
        json_dir: Directory containing VLMOD JSON files.
        transform: Optional image transform. If omitted, the PIL image is
            returned unchanged.
        require_exact_pair: If True, JSONs without an image of the same stem
            are excluded. If False, initialization raises an error instead.
    """

    def __init__(
        self,
        image_dir: str | Path,
        json_dir: str | Path,
        transform=None,
        require_exact_pair: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        self.transform = transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.json_dir.is_dir():
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")

        image_map = {
            exact_stem(p): p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        json_paths = sorted(self.json_dir.glob("*.json"))

        self.missing_images = [
            p for p in json_paths if exact_stem(p) not in image_map
        ]
        self.extra_images = [
            p for stem, p in image_map.items()
            if not (self.json_dir / f"{stem}.json").exists()
        ]

        if self.missing_images and not require_exact_pair:
            names = ", ".join(p.name for p in self.missing_images[:5])
            raise FileNotFoundError(
                f"{len(self.missing_images)} JSON files have no exact image pair. "
                f"Examples: {names}"
            )

        self.records: List[Tuple[Path, Path]] = [
            (json_path, image_map[exact_stem(json_path)])
            for json_path in json_paths
            if exact_stem(json_path) in image_map
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        json_path, image_path = self.records[index]
        data = load_vlmod_json(json_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "query_annotations": data["annotations"],
            "calibration": data["calibration"],
            "denorm": data["denorm"],
        }


def vlmod_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length annotations without destroying their structure."""
    images = [item["image"] for item in batch]
    if all(isinstance(x, torch.Tensor) for x in images):
        images = torch.stack(list(images), dim=0)

    return {
        "images": images,
        "image_paths": [item["image_path"] for item in batch],
        "json_paths": [item["json_path"] for item in batch],
        "query_annotations": [item["query_annotations"] for item in batch],
        "calibration": [item["calibration"] for item in batch],
        "denorm": [item["denorm"] for item in batch],
    }
