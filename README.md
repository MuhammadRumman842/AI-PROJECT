# VLMOD / MonoMulti-3DVG — Engineering Baseline

This project is a learning/reproduction implementation built around the **VLMOD / MonoMulti-3DVG** task: given a monocular RGB image and a natural-language description, predict the referred objects' 3D properties.

The official CVPR 2025 paper introduces **MonoMulti-3DVG**, the **MonoMulti3D-ROPE** benchmark, and **CyclopsNet**, which uses State-Prompt Visual Encoder (SPVE), Denoising Alignment Fusion (DAF), probabilistic/Gaussian representations, and a two-stage detection-to-language grounding procedure. See the official paper for the authors' method. 

> **Important:** the code in this repository is a **CyclopsNet-inspired engineering scaffold**, not a claim of the authors' exact implementation. Several internal equations, detector details, training hyperparameters, and other implementation details were not available in the supplied partial repository, so those parts were reconstructed explicitly as engineering choices.

## 1. Task

Input:
- one monocular RGB image
- one complex natural-language query

Target object properties represented by this project include:
- object class
- 2D bounding box
- 3D dimensions `(width, height, depth)`
- 3D location `(x, y, z)`
- rotation/orientation
- query-object similarity

The supplied labels contain vehicle classes such as `car`, `van`, `truck`, and `bus`.

## 2. Supplied dataset structure

The working data is expected one directory above this repository:

```text
VLMOD/
├── IMG/
│   ├── train/
│   └── test/
├── FILES/
│   ├── train/
│   └── test/
└── MonoMulti-3DVG-main/
```

During development, the supplied archive contained:
- 125 train images + 125 train JSON files
- 125 test images + 125 test JSON files
- 76 exact train image/JSON filename-stem pairs
- 93 exact test image/JSON filename-stem pairs
- 913 parsed train object labels
- 1,699 test candidate records

**Important data rule:** unmatched image/JSON files were not guessed or force-matched. The training experiments use only exact filename-stem pairs. The supplied test candidate records are treated as **unlabeled for evaluation**; no official test accuracy is fabricated.

## 3. Environment setup

Use Python 3.10+ (the development environment used Python 3.13).

```bash
cd MonoMulti-3DVG-main
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Quick verification

Run the lightweight preprocessing/model/loss/inference/visualization checks:

```bash
python scripts/test_preprocessing.py
python scripts/test_model.py
python scripts/test_loss.py
python scripts/test_infer.py
```

The visualization test can be reproduced with the real matched sample used during development:

```bash
python scripts/visualize.py \
  --image ../IMG/train/62540_fa2sd4a16East154_420_1625808117_1625808861_282_obstacle.jpg \
  --query "Vehicles, which are white in appearance and located on the left side of the scene, have a width ranging from approximately 2.0 to 2.4 meters" \
  --checkpoint checkpoints/stage11_controlled/best_val.pt \
  --gt-json ../FILES/train/62540_fa2sd4a16East154_420_1625808117_1625808861_282_obstacle.json \
  --output outputs/visualization.png \
  --device cpu
```

## 5. Training

### Tiny sanity training

This verifies forward pass, loss, backward pass, optimizer update, and checkpoint save/reload:

```bash
python scripts/train_sanity.py
```

### Controlled training

For a small reproducible experiment on the exact-pair subset:

```bash
python scripts/train.py \
  --epochs 2 \
  --batch-size 2 \
  --lr 1e-4 \
  --seed 42 \
  --max-train-examples 20 \
  --max-val-examples 8 \
  --device cpu \
  --checkpoint-dir checkpoints/stage11_controlled
```

The training script splits at the image-pair level before query expansion, then trains the engineering baseline and reports validation diagnostics.

## 6. Inference

Run one image + query through a saved checkpoint:

```bash
python scripts/infer.py \
  --image ../IMG/train/62540_fa2sd4a16East154_420_1625808117_1625808861_282_obstacle.jpg \
  --query "Vehicles, which are white in appearance and located on the left side of the scene, have a width ranging from approximately 2.0 to 2.4 meters" \
  --checkpoint checkpoints/stage11_controlled/best_val.pt \
  --top-k 5 \
  --threshold 0.30 \
  --output outputs/predictions.json \
  --device cpu
```

The `0.30` threshold and combined ranking score are transparent engineering choices, **not official paper settings**.

## 7. Visualization

`visualize.py` draws predicted 2D boxes on the original image. When `--gt-json` is supplied, ground-truth boxes are drawn separately for qualitative comparison.

The development visualization is stored at:

```text
outputs/stage13_visualization.png
outputs/stage13_visualization.json
```

## 8. Checkpoints

Development checkpoints were generated during the staged experiments. They are intentionally not required for the source-code archive because they are large model-state files.

Important development checkpoint locations when available:

```text
checkpoints/stage9_sanity.pt
checkpoints/stage11_controlled/best_val.pt
checkpoints/stage11_controlled/latest.pt
```

## 9. What is confirmed vs reconstructed

### Confirmed from the official paper
- Task: MonoMulti-3DVG
- Dataset: MonoMulti3D-ROPE
- Model name: CyclopsNet
- SPVE and DAF modules
- two-stage inference: 3D object detection followed by language-object matching
- probabilistic/Gaussian representations for object, text, and multimodal representations
- RoBERTa-base and KAN are described in the paper
- KL-divergence-based probabilistic alignment is described in the paper

### Reconstructed in this student implementation
- lightweight image encoder choice
- fixed candidate-slot representation
- text encoding implementation
- Gaussian parameterization/fusion implementation
- candidate-to-target matching and loss weights
- prediction heads and engineering score threshold
- training hyperparameters used for sanity experiments

These reconstructed choices are useful for learning and for building a working baseline, but they must not be presented in an interview as the authors' exact CyclopsNet implementation.

## 10. Development results

The staged pipeline was verified on real supplied data:
- model forward pass: **OK**
- loss + backward pass: **OK**
- optimizer/checkpoint sanity: **OK**
- controlled 2-epoch training experiment: **OK**
- inference: **OK**
- visualization: **OK**

The controlled experiment used only a very small subset and produced validation diagnostics. These numbers are not official challenge accuracy and should not be used as evidence of final model performance.

## 11. Project documentation

- `PROJECT_PROGRESS.md` — stage-by-stage verified progress and honesty notes
- `NEXT_CLAUDE_PROMPT.md` — continuation instructions; future work should read progress first
- `MODEL_ARCHITECTURE_SPEC.md` — confirmed/reconstructed architecture specification
- `STAGE6_ARCHITECTURE_VERIFICATION.md` — architecture verification record
- `src/model.py` — CyclopsNet-inspired model scaffold
- `src/losses.py` — engineering baseline matching/losses
- `src/dataset.py` — exact-pair dataset loader
- `src/preprocessing.py` — image/query preprocessing

## 12. Academic note

This repository is intended for academic learning and experimentation. When presenting the project, clearly distinguish:
1. what the official paper reports,
2. what was verified from the supplied partial repository/data, and
3. what was reconstructed as an engineering baseline.
