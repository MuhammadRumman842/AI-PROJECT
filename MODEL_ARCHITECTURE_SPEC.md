# VLMOD / CyclopsNet Architecture Specification

## Status
Stage 5 design specification. This document separates facts confirmed by the CVPR 2025 paper from implementation decisions that still need verification against the paper figures/equations or the authors' partial repository.

## 1. Task
Input:
- one monocular RGB image
- one natural-language referring description

Output for every referred object:
- 3D position (x, y, z)
- 3D size (width, height, depth)
- orientation / rotation angle

The supplied README describes this as VLMOD Challenge Track B and explicitly says the repository is partial code.

## 2. Official model identity
The CVPR 2025 paper proposes **CyclopsNet** for MonoMulti-3DVG. The paper states that CyclopsNet integrates:
- State-Prompt Visual Encoder (SPVE)
- Denoising Alignment Fusion (DAF)

The paper also describes object, textual, and multi-modal representations as Gaussian distributions and describes a two-stage inference procedure for object-text matching.

## 3. High-level pipeline

RGB image
   |
   v
Visual representation
   |
   +--> SPVE (state/prompt-conditioned visual encoding)
   |
   +------------------------------+
                                  |
Natural-language query --> text representation
                                  |
                                  v
                         DAF multimodal fusion
                                  |
                                  v
                     joint visual-language representation
                                  |
                                  v
                    object / query grounding stage
                                  |
                                  v
                  referred-object 3D property prediction
                                  |
             +--------------------+-------------------+
             |                    |                   |
          position             size             rotation
          (x,y,z)            (w,h,d)              angle

## 4. What is confirmed vs. what is not

### Confirmed by the paper
- Task is monocular RGB multi-object 3D visual grounding.
- Model name is CyclopsNet.
- SPVE and DAF are core modules.
- The method uses probabilistic/Gaussian representations for object, text, and multimodal features.
- The paper describes a two-stage inference procedure for object-text matching.

### Not yet safe to claim as exact implementation details
- Exact backbone names and layer counts.
- Exact tokenizer/text encoder.
- Exact dimensions of latent vectors.
- Exact SPVE equations and prompt/state construction.
- Exact DAF equations and denoising schedule.
- Exact Gaussian parameterization and divergence/loss formula.
- Exact prediction-head architecture.
- Exact training hyperparameters.

These must be extracted from the paper/released source before coding them as "the original" implementation.

## 5. Implementation decision for this project
We will NOT jump directly to a DETR/Hungarian architecture. That would be a separate reconstruction rather than a faithful CyclopsNet implementation.

The next coding stage should implement only components whose behavior is supported by:
1. the supplied repository,
2. the paper, or
3. an explicitly documented engineering baseline.

Any engineering baseline will be labeled as such in the code and README.

## 6. Data interface to the future model
The Stage 4 preprocessing interface supplies:
- RGB image tensor: [C,H,W]
- normalized query text
- variable-length target object records
- 2D box [x1,y1,x2,y2]
- 3D dimensions [w,h,d]
- 3D location [x,y,z]
- rotation angle
- class / instance identifiers
- camera calibration and denormalization metadata

## 7. Recommended next step
Before implementing `src/model.py`, inspect the paper's method section, figures, equations, and the released `libs.py` for evidence of the exact probabilistic modules. Then write a small architecture test that checks tensor shapes and forward-pass contracts before training.
