# GS-Fuse

Active multimodal financial forecasting experiments split from `Multi-Model-Project`.

## Scope

This repo contains the current GS/FS-Fuse training and evaluation line:

- `train3.py`: LLaMA + MOMENT three-stage training using `model/CAMEF4P19L.py`.
- `train_kronos.py`: LLaMA + Kronos three-stage training using `model/CAMEF4P19K.py`.
- `test_financial.py`: financial evaluation utilities for trained checkpoints.
- `test_gate_openness.py`: gate-openness analysis for multimodal fusion behavior.

## Layout

```text
GS-Fuse/
├── train3.py
├── train_kronos.py
├── test_financial.py
├── test_gate_openness.py
├── model/
└── data/
```

## Local Dependencies

The current scripts still contain local paths for model checkpoints such as LLaMA, MOMENT, and Kronos. Before running, update those paths in the training scripts or refactor them into CLI/config files.

Required data layout by default:

```text
data/event/
data/series/
```

Large data, model weights, checkpoints, and logs are intentionally ignored by git.

## Quick Checks

```bash
python -m py_compile train3.py train_kronos.py test_financial.py test_gate_openness.py
```
