# GS-Fuse Public Release Cleanup Design

## Goal

Prepare `GS-Fuse` for upload to a new public GitHub repository by removing local/generated clutter, dropping legacy scripts that are not needed for training or testing, and refining the README around the unified `train.py` workflow.

## Current State

`GS-Fuse` is already a git repository on `main`, but it has no configured remote. The working tree contains a README update, new unified model/training files, tests, an `environment.yml`, and modified legacy compatibility files. Local cache directories such as `.pytest_cache/` and `__pycache__/` are present in the project folder but are ignored by `.gitignore`.

## Cleanup Scope

Keep only scripts that support core training or testing:

- Remove generated/local artifacts from the working folder, especially `.pytest_cache/` and `__pycache__/`.
- Delete `train3.py` and `train_kronos.py` because they are legacy wrappers with machine-specific default paths.
- Delete `model/CAMEF4P19L.py` and `model/CAMEF4P19K.py` because they are compatibility shims, not training or testing entrypoints.
- Keep `test_pretrained_model.py` and `tests/` because they support checkpoint evaluation and lightweight automated tests.
- Delete `test_financial.py` and `test_gate_openness.py` because they are optional analysis scripts, not required for core training or tests, and contain local-machine defaults.
- Keep `moment.yml` and `phi.yml` as legacy environment references unless a later review confirms they are obsolete.

## README Refinement

The README should present the project as a clean public research codebase:

- Explain the GS-FUSE goal and supported backbones.
- Show `train.py` as the single training entrypoint.
- Document external model/data/checkpoint requirements without exposing local paths as defaults.
- Describe checkpoint evaluation, tests, data layout, ignored large assets, and migration notes.
- Remove references that imply legacy wrappers or CAMEF shim files are available entrypoints.

## Verification

Run lightweight checks that do not require large local checkpoints or datasets:

- `python -m py_compile train.py model/gs_fuse.py model/ts_encoders.py model/text_encoders.py test_pretrained_model.py`
- `python -m pytest tests/test_encoder_shapes.py -q`
- `python -m pytest tests/test_pretrained_eval_metrics.py -q`

If dependency-heavy checks fail because the active environment lacks project dependencies, record that clearly and avoid claiming the test suite passes.

## GitHub Upload

After cleanup and verification, commit the release-prep changes, create a new public GitHub repository named `GS-Fuse`, add it as `origin`, and push `main`.
