# Stage 6 — CyclopsNet Architecture Verification

## Goal
Verify what can be stated confidently about the authors' CyclopsNet architecture before implementing model code.

## Sources checked
- Official CVPR 2025 paper: `Beyond Human Perception: Understanding Multi-Object World from Monocular View`.
- Supplied repository and its existing `libs.py`.
- Existing project dataset/preprocessing findings.

## Confirmed architecture
The official paper identifies the proposed model as **CyclopsNet**. It is a **two-stage** architecture:

1. **Stage 1 — 3D detection:** a 3D detector identifies objects in the monocular scene.
2. **Stage 2 — language grounding:** detected 3D objects are matched with the natural-language description.

The paper introduces:
- **State-Prompt Visual Encoder (SPVE):** uses object state information (orientation, dimensions, position) to create a richer visual representation.
- **Denoising Alignment Fusion (DAF):** reduces uncertainty/noise during multimodal alignment and fusion.
- **Probabilistic representations:** object, text, and multimodal representations are modeled as Gaussian distributions.
- **Object-text matching:** the final system selects objects matching the description using a similarity threshold.

These points are directly supported by the official CVPR paper.

## What is NOT verified yet
The supplied partial repository does not contain the complete CyclopsNet implementation. The available `libs.py` contains KAN and VAE-style helper components, but it does not prove their exact location or role in CyclopsNet.

The following details remain unverified and MUST NOT be presented as the authors' exact implementation:
- exact image/3D detector backbone;
- exact text encoder/tokenizer;
- exact SPVE layer structure and equations;
- exact DAF layer structure and equations;
- exact Gaussian parameterization (`mu`, covariance/log-variance, etc.);
- exact probabilistic distance/divergence used for matching;
- exact loss terms and weights;
- exact 3D detector prediction heads;
- exact training hyperparameters;
- exact similarity threshold value;
- exact KAN/VAE helper usage.

## Engineering decision
Do **not** invent missing equations or claim a reconstructed module is the professor/authors' implementation.

The next implementation should therefore use a clearly labeled **CyclopsNet-inspired engineering scaffold** only after the forward-pass contracts are agreed. Every reconstructed component must be marked as such in code/documentation.

## Important distinction
The paper confirms the *architecture idea and module names*, but the supplied release is partial. Therefore, matching the paper at the conceptual level is possible, while reproducing the authors' exact implementation is not yet justified from the available source.
