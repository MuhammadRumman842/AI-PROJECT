# VLMOD Project Progress

## Goal
Build a reproducible implementation for VLMOD Track B: multi-object 3D visual grounding from a monocular RGB image and natural-language descriptions.

## Completed
- Inspected the supplied `VLMOD.zip`.
- Confirmed the archive contains `IMG/`, `FILES/`, and `MonoMulti-3DVG-main/`.
- Confirmed the released repository is explicitly **partial code** and contains `libs.py` plus `README.md`; a bundled Windows `venv/` is also present and is not treated as project source.
- Parsed the VLMOD JSON format safely with `ast.literal_eval()` for `label_3` records.
- Verified available image dimensions are 1920x1080 RGB JPEGs.
- Implemented `src/dataset.py` using exact filename-stem matching only; no guessed pairing is performed.
- Implemented a variable-length annotation collate function.
- Smoke-tested the dataset loader and DataLoader on the supplied archive.

## Dataset facts verified from the supplied archive
- Train: 125 JSON files, 125 images, 76 exact stem pairs.
- Test: 125 JSON files, 125 images, 93 exact stem pairs.
- The remaining JSON/image files do not have an exact same-stem counterpart in the supplied archive.
- Train annotations include natural-language descriptions, `public_properties`, `label_3`, calibration, and denormalization data.
- `label_3` contains class, instance/group IDs, depth, 2D bounding box, dimensions, 3D location, rotation, and appearance.
- Observed object classes: car, van, truck, bus.

## Important limitation
The archive is not the complete 1 GB dataset. The current loader therefore exposes only exact pairs present in the supplied archive. It must not invent matches for missing pairs.

## Files created
- `src/dataset.py`
- Existing/previous validation artifacts should be retained if present: `scripts/inspect_dataset.py`, `scripts/validate_dataset.py`, `DATASET_VALIDATION_REPORT.md`.

## Test result
`src/dataset.py` compiles successfully. A train Dataset instantiated successfully with 76 records; one sample loaded as a 1920x1080 RGB PIL image with 3 query annotations, a 3x4 calibration tensor, and a 4-value denorm tensor. DataLoader batching also succeeded with the custom collate function.

## Stage 4 completed
- Added `src/preprocessing.py` for image/query/target preparation.
- Added `scripts/test_preprocessing.py` and verified preprocessing on the supplied exact-pair subset.
- Verified image tensor shape `(3, 224, 224)` and variable-length target fields.

## Stage 5 architecture decision
- Verified from the CVPR 2025 paper that the proposed model is **CyclopsNet**, not a DETR model.
- Confirmed that CyclopsNet integrates SPVE (State-Prompt Visual Encoder) and DAF (Denoising Alignment Fusion).
- Confirmed that the paper describes object, textual, and multimodal representations as Gaussian distributions and a two-stage inference procedure for object-text matching.
- Created `MODEL_ARCHITECTURE_SPEC.md` separating confirmed paper facts from details that still require verification.
- Decision: do not claim an invented DETR/Hungarian architecture as the professor/authors' model.

## Stage 6 completed
- Re-checked the official CVPR 2025 paper and the supplied partial repository before writing model code.
- Confirmed CyclopsNet is a two-stage system: 3D detection first, then object-text matching.
- Confirmed SPVE, DAF, Gaussian/probabilistic representations, and similarity-threshold matching.
- Confirmed that the supplied repository does not provide enough source to justify the authors' exact SPVE/DAF equations, backbones, probabilistic distance, losses, or hyperparameters.
- Added `STAGE6_ARCHITECTURE_VERIFICATION.md` documenting confirmed facts vs. unresolved details.
- Decision: do not fabricate missing equations or claim a reconstructed implementation is the professor/authors' exact model.

## Current stage
Stage 6: Architecture verification complete; model implementation is the next stage.

## Next stage
Build a **clearly labeled CyclopsNet-inspired engineering scaffold** with explicit forward-pass contracts and shape tests. The scaffold must separate confirmed architecture concepts from reconstructed implementation choices. Do not start large-scale training until the forward pass works on real supplied samples.

## Stage 4
- Added `src/preprocessing.py`.
- Defined a reconstructed-baseline input representation: normalized 224x224 RGB tensor plus raw query text.
- Expanded each image record into one example per referring-expression query.
- Defined variable-length targets containing class IDs, KITTI 2D boxes, dimensions, location, rotation, and instance IDs.
- This target format is our baseline engineering decision, not a claim about the professor's original architecture.
- Next stage: explain and choose the model architecture before implementing it.

## Stage 7 completed
- Added `src/model.py` as a clearly labeled **CyclopsNet-inspired engineering scaffold**, not the authors' exact implementation.
- Implemented the verified high-level flow: image -> candidate 3D objects -> state-aware visual representation -> probabilistic object/text representations -> uncertainty-aware fusion -> object-text similarity.
- Used fixed candidate slots only as an engineering scaffold; this is not claimed to be the paper's exact detector.
- Added `scripts/test_model.py` for a real-sample forward-pass smoke test.

## Stage 7 test result
- Real supplied train sample was loaded successfully.
- Model forward pass completed without errors.
- Verified finite output tensors and expected shapes for objectness, class logits, 2D boxes, dimensions, location, rotation, depth, Gaussian parameters, fused representations, and similarity.
- Test output ended with `FORWARD PASS: OK`.
- No large-scale training was started.

## Stage 7 architecture honesty note
The following are reconstructed engineering choices and must not be presented in an interview as the authors' exact implementation: ResNet18 backbone choice, fixed candidate-slot detector, hashed-token GRU text encoder, latent dimension, precision-weighted Gaussian fusion formula, and cosine similarity implementation. The paper only verifies the higher-level CyclopsNet/SPVE/DAF/probabilistic/two-stage concepts used as the scaffold basis.

## Stage 8 completed
- Added `src/losses.py` with a clearly labeled engineering-baseline candidate-to-target matcher and multi-task loss.
- Matching uses one-to-one greedy assignment from 2D box cost + class mismatch + 3D location/dimension costs.
- Losses cover objectness, class, 2D box, dimensions, 3D location, rotation, and query-object similarity margin.
- Explicit loss weights are engineering choices and are NOT claimed to be the authors' CyclopsNet hyperparameters.
- Added `scripts/test_loss.py`.

## Stage 8 test result
- Ran on one real supplied train sample.
- Sample contained 1 referred target object; 1 candidate was matched.
- All loss components and total loss were finite.
- Backward pass completed and finite gradients reached model parameters.
- Test ended with `LOSS + BACKWARD: OK`.
- No full training was started.

## Stage 8 architecture honesty note
The greedy matcher, loss formulas, loss weights, and similarity-margin objective are reconstructed engineering choices. They are not verified as the authors' exact CyclopsNet training objective.

## Current stage
Stage 8: loss + target matching complete.

## Next stage
Build a **small training/sanity-check pipeline**: one or two batches only, optimizer step, loss decrease/parameter update check, checkpoint save, and clear logging. Do not run large-scale training yet.

## Stage 9 completed
- Added `scripts/train_sanity.py` for a deliberately tiny optimizer/checkpoint sanity test.
- Used the real supplied exact-pair train subset: 76 matched image/JSON records.
- Ran exactly 2 optimizer steps on 2 real query examples.
- Verified forward -> loss -> backward -> optimizer.step().
- Verified finite losses/gradients and that trainable parameters actually changed.
- Step losses observed: 18.735327 then 38.874126. These values are only sanity-check diagnostics; they are NOT an accuracy result and the second-step increase is not evidence of model quality.
- Verified checkpoint save and reload, including model and optimizer state.
- Saved `checkpoints/stage9_sanity.pt`.
- No large-scale training was started.

## Stage 9 honesty note
The sanity run only proves that the reconstructed engineering scaffold can execute an optimizer step and checkpoint round-trip. It does not prove convergence, accuracy, generalization, or equivalence to the authors' CyclopsNet implementation.

## Current stage
Stage 9: tiny training/checkpoint sanity check complete.

## Next stage
Build a proper training/validation pipeline using the labeled exact-pair train subset, with configuration, epoch/batch logging, validation metrics, checkpointing, and resume support. Before full training, explain what each metric means and explicitly distinguish validation metrics from challenge test performance.

## Stage 10 completed
- Added `scripts/train.py` with configurable epochs, batch size, learning rate, validation split, seed, device, example limits, checkpoint directory, and resume support.
- Training/validation split is performed at the exact image/JSON-pair level before query expansion to reduce image-level leakage between train and validation.
- Added validation diagnostics: total/component losses, matched-object classification accuracy, normalized 2D box MAE, 3D location MAE, dimensions MAE, and rotation MAE.
- Added `latest.pt` and `best_val.pt` checkpointing with model state, optimizer state, epoch, configuration, and metrics.
- Added resume support from a saved checkpoint.
- Ran a deliberately tiny Stage-10 sanity run on CPU: 1 epoch, 4 train query examples, 2 validation query examples.
- Stage-10 sanity result: `STAGE 10 TRAINING PIPELINE: OK`.
- Tiny-run diagnostics: train total loss 35.430672; validation total loss 43.639097; validation matched-class accuracy 1.0000; validation location MAE 39.632475. These are tiny-run diagnostics only and must not be presented as final model performance.
- Saved `checkpoints/latest.pt` and `checkpoints/best_val.pt` from the tiny run.

## Stage 10 honesty note
The training/validation pipeline trains the reconstructed CyclopsNet-inspired engineering scaffold. It is not the authors' exact implementation. Validation metrics are computed only on the supplied exact-pair subset and are not official challenge test scores. The official test candidate records are not treated as labeled ground truth.

## Current stage
Stage 10: conservative training/validation pipeline complete.

## Next stage
Run a larger but still controlled training experiment on the available exact-pair training subset, compare train/validation curves and checkpoint behavior, then build an inference/evaluation script that reports predictions without inventing official test accuracy. After that, add visualization and final project cleanup/documentation.

## Stage 11 completed
- Ran a controlled training experiment using the existing training/validation pipeline.
- Used CPU, 2 epochs, batch size 2, learning rate 1e-4, seed 42.
- Used 20 training query examples and 8 validation query examples from the supplied exact-pair subset.
- Train/validation image-pair split: 61/15 pairs.
- Epoch 1: train loss 35.664664; validation loss 31.480422; validation matched-class accuracy 1.0000; validation location MAE 27.654158.
- Epoch 2: train loss 33.882098; validation loss 30.627240; validation matched-class accuracy 1.0000; validation location MAE 27.394683.
- Validation loss and location MAE improved between the two epochs; this is an encouraging pipeline/model-sanity signal, not evidence of final accuracy.
- Best checkpoint saved at `checkpoints/stage11_controlled/best_val.pt`.
- This remains a reconstructed CyclopsNet-inspired engineering scaffold, not the authors' exact implementation.

## Stage 11 honesty note
The controlled experiment is too small to support claims about generalization or challenge performance. The reported metrics are validation diagnostics on the supplied exact-pair subset only. Official test accuracy is not available from these labels and must not be fabricated.

## Current stage
Stage 11: controlled training experiment complete.

## Next stage
Build the inference/evaluation script. It should load a checkpoint, accept a real image + query, produce object candidates/predictions, report confidence/similarity and 2D/3D outputs, and clearly distinguish local validation evaluation from unlabeled challenge-test inference.

## Stage 12 completed
- Added `scripts/infer.py` for checkpoint-based inference from one image + natural-language query.
- Inference reports candidate objectness, class/confidence, query similarity, normalized 2D box, dimensions, 3D location, rotation, and depth.
- Added transparent engineering score: average of objectness, class confidence, and normalized similarity; default threshold is 0.30. This threshold is NOT claimed to be from the paper.
- Added optional JSON output saving.
- Added `scripts/test_infer.py` and ran it on one real supplied exact-pair training sample.
- Test returned 5 ranked predictions and all checked numeric outputs were finite.
- Test ended with `STAGE 12 INFERENCE: OK`.
- No additional training was started.

## Stage 12 honesty note
Inference results are predictions from the reconstructed CyclopsNet-inspired engineering scaffold. They are not official CyclopsNet results and are not challenge accuracy. The supplied test candidate records remain unlabeled for evaluation purposes.

## Current stage
Stage 12: inference/evaluation interface complete.

## Next stage
Build visualization: draw predicted 2D boxes and labels on a real image, optionally compare with ground-truth boxes for a labeled validation sample, and save the visualization. Keep prediction-vs-ground-truth comparison clearly separate from official challenge metrics.

## Stage 13 completed
- Added `scripts/visualize.py` for demo visualization of model predictions on real VLMOD images.
- Visualization draws predicted normalized 2D boxes in image coordinates with class, score, and query similarity labels.
- Optional ground-truth 2D boxes from a matched training JSON are drawn separately.
- Ground truth parsing uses `ast.literal_eval()` for the string-encoded `label_3` records.
- Tested on a real exact-pair training sample.
- Result: `STAGE 13 VISUALIZATION: OK`.
- Saved visualization to `outputs/stage13_visualization.png` and prediction sidecar to `outputs/stage13_visualization.json`.
- Test produced 5 predictions and 3 parsed ground-truth boxes.
- Visual inspection confirmed the image/box overlay renders correctly.

## Stage 13 honesty note
The visualization is a qualitative demonstration of the reconstructed CyclopsNet-inspired engineering baseline. Red predicted boxes and blue ground-truth boxes are not an official challenge evaluation, and the small controlled model should not be interpreted as a trained production detector.

## Current stage
Stage 13: visualization complete.

## Next stage
Final project cleanup and documentation: audit files, remove generated caches, ensure README explains the reconstructed-vs-confirmed architecture distinction, add reproducible commands, update progress/next prompt, and create a clean project archive if packaging succeeds.

## Stage 14 completed
- Audited the final repository structure and Python imports.
- Updated `README.md` with task description, dataset layout, exact-pair limitation, setup, verification, training, inference, visualization, checkpoint locations, and confirmed-vs-reconstructed architecture notes.
- Added `requirements.txt` for the Python dependencies used by the project.
- Re-ran Python compilation checks for `src`, `scripts`, and `libs.py`: `STAGE 14 COMPILE: OK`.
- Removed generated `__pycache__` directories from the clean project tree.
- Kept large training checkpoints out of the clean source archive to avoid packaging a ~700+ MB artifact; checkpoints remain in the working project directory.
- No model redesign or large-scale training was performed in Stage 14.

## Stage 14 honesty note
The final source tree documents the implementation as a CyclopsNet-inspired engineering scaffold. It does not claim that the reconstructed model is the authors' exact CyclopsNet implementation. No official challenge/test accuracy is reported.

## Current stage
Stage 14: final cleanup and documentation complete.

## Next stage
No further implementation stage is required for the baseline. If continuing, the next useful step is interview preparation: understand each file, tensor shape, loss term, training step, and the confirmed-vs-reconstructed architecture distinction before making any additional code changes.
