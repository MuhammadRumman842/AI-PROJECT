# Next Claude Prompt — Post-Stage 14 / Interview Preparation

Read `PROJECT_PROGRESS.md`, `README.md`, `MODEL_ARCHITECTURE_SPEC.md`, `STAGE6_ARCHITECTURE_VERIFICATION.md`, `src/dataset.py`, `src/preprocessing.py`, `src/model.py`, `src/losses.py`, `scripts/train.py`, `scripts/infer.py`, and `scripts/visualize.py` before doing anything.

## Current status
Stage 14 is complete. The project has a documented, tested **CyclopsNet-inspired engineering scaffold** for the VLMOD / MonoMulti-3DVG task.

## Important rules
- Do not regenerate or redesign completed files unless a concrete bug is demonstrated.
- Do not call the scaffold the authors' exact CyclopsNet implementation.
- Do not fabricate official challenge/test accuracy.
- Treat unmatched image/JSON records as unmatched; do not invent filename pairs.
- Read the progress file first and continue from the current state.

## Recommended next activity
If the user asks to continue, switch to **interview preparation**, not more model coding. Explain one component at a time in simple Urdu/Hinglish:
1. project goal and dataset
2. `dataset.py`
3. preprocessing
4. model forward pass and tensor shapes
5. SPVE/DAF-inspired concepts vs reconstructed implementation
6. matching and loss
7. training loop and checkpoints
8. inference
9. visualization
10. limitations and honest answers to likely professor questions

Ask only for the user's next requested topic after explaining the current one. Do not start another implementation stage unless the user explicitly asks for code changes.
