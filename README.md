# GS-FUSE

GS-FUSE is a multimodal financial forecasting project that fuses text events and time-series windows through a shared cross-attention / gating stack to predict future market trajectories.

The public project is centered on one model class (`GSFuse`), one training entrypoint (`train.py`), and a small set of checkpoint/testing utilities.

## Supported Backbones

| Modality | Options | CLI flag |
|----------|---------|----------|
| Time series | MOMENT, Kronos | `--ts-backbone moment|kronos` |
| Text | LLaMA-family, Phi-family | `--text-backbone llama|phi` |

Any time-series and text pairing is valid, for example MOMENT + Phi or Kronos + LLaMA.

## Architecture

```text
Text events  -> text encoder (LLaMA / Phi) -> text tokens --+
                                                            +-> GS-FUSE fusion -> decoder -> forecast
TS windows   -> ts encoder (MOMENT / Kronos) -> ts tokens --+
```

Training runs in three stages:

1. Stage 1: TS-only pretraining with the sliding-window loader.
2. Stage 2: text-only pretraining with the event loader.
3. Stage 3: full multimodal training.

## Project Layout

```text
GS-Fuse/
|-- train.py                    # primary training entrypoint
|-- test_pretrained_model.py    # checkpoint evaluation utility
|-- environment.yml             # unified conda environment
|-- moment.yml                  # legacy MOMENT env export, reference only
|-- phi.yml                     # legacy Phi env export, reference only
|-- model/
|   |-- gs_fuse.py              # GSFuse model and training loop
|   |-- ts_encoders.py          # MOMENT and Kronos TS encoders
|   |-- text_encoders.py        # LLaMA and Phi text encoders
|   |-- kronos.py               # vendored Kronos implementation
|   `-- ...                     # attention, masking, decoder modules
|-- data/
|   |-- dataloader.py
|   `-- sliding_window_dataloader.py
`-- tests/
    |-- test_encoder_shapes.py
    `-- test_pretrained_eval_metrics.py
```

## Environment

Use the unified conda environment:

```bash
conda env create -f environment.yml
conda activate gs-fuse
```

`environment.yml` unifies the MOMENT and Phi exports with the deployed Kronos runtime needs. It keeps the stable MOMENT-compatible pins (`torch==2.3.0`, `transformers==4.43.1`, `momentfm==0.1.4`) and adds Kronos dependencies such as `einops`.

## External Assets

Before training or evaluation, provide local paths or Hugging Face IDs for the required external assets:

- Text encoder weights: LLaMA or Phi.
- MOMENT checkpoint when using `--ts-backbone moment`.
- Kronos model and tokenizer when using `--ts-backbone kronos`.
- Event text data under `data/event/` or a custom `--event-dir`.
- Time-series CSV data under `data/series/` or a custom `--series-dir`.

Large datasets, pretrained weights, checkpoints, outputs, and logs are intentionally ignored by git.

## Training

### MOMENT + LLaMA

```bash
python train.py \
  --ts-backbone moment \
  --text-backbone llama \
  --moment-path /path/to/MOMETN-1-large \
  --text-model-path /path/to/llama3-2-3B \
  --series-id SP500 \
  --seq-len 35 \
  --pred-len 140 \
  --batch-size 32 \
  --output-path /path/to/output
```

### Kronos + Phi

```bash
python train.py \
  --ts-backbone kronos \
  --text-backbone phi \
  --kronos-model-path /path/to/Kronos-base \
  --kronos-tokenizer-path /path/to/Kronos-Tokenizer-base \
  --text-model-path /path/to/Phi-3.5-mini-instruct \
  --series-id SP500 \
  --output-path /path/to/output
```

### Useful Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--event-dir` | `data/event` | Event text data root |
| `--series-dir` | `data/series` | Time-series data root |
| `--stage1-epochs` | `2` | TS-only stage epochs |
| `--stage2-epochs` | `2` | Text-only stage epochs |
| `--num-epochs` | `2` | Multimodal stage epochs |
| `--finetune-ts-last-block` | off | Partially unfreeze MOMENT/Kronos last block in Stage 1 |
| `--finetune-text-last-block` | model default | Partially unfreeze LLaMA/Phi last block |

## Checkpoint Evaluation

Use `test_pretrained_model.py` for pretrained GS-FUSE checkpoint evaluation. It reports forecasting metrics and long/short financial strategy summaries at configurable forecast horizons.

```bash
python test_pretrained_model.py \
  --model-path /path/to/checkpoint/lastest_model.pth \
  --ts-backbone moment \
  --text-backbone phi \
  --moment-path /path/to/MOMETN-1-large \
  --text-model-path /path/to/Phi-3.5-mini-instruct \
  --series-id NASDAQ \
  --seq-len 35 \
  --pred-len 140 \
  --eval-horizons 35 70 140 \
  --output-json /path/to/metrics.json
```

For Kronos checkpoints, replace `--moment-path` with `--kronos-model-path` and `--kronos-tokenizer-path`. Add `--inverse-transform` if endpoint DHR/Sharpe should be computed in original scaler space instead of scaled space.

## Data Layout

```text
data/event/     # event text files by type/date
data/series/    # per-asset time-series CSVs
```

## Checkpoints

GS-FUSE saves combined checkpoints via `GSFuse.save_model_combined()`.

| Key | Contents |
|-----|----------|
| `ts_encoder` | MOMENT or Kronos backbone weights |
| `text_encoder` | Text encoder module state |
| `llm` | Legacy-compatible LLM weights saved for older checkpoints |
| `decoder`, `llm_proj`, fusion layers | Fusion and prediction head |

Loading accepts legacy checkpoint keys (`moment`, `llm`) from older runs. New checkpoints use `ts_encoder` and `text_encoder`.

## Python API

```python
from model.gs_fuse import GSFuse, train_three_stage

model = GSFuse(
    text_model_path="/path/to/llama-or-phi",
    ts_backbone="moment",
    text_backbone="llama",
    moment_path="/path/to/MOMENT",
    seq_len=35,
    pred_len=140,
    d=4,
)
```

## Quick Checks

```bash
python -m py_compile \
  train.py model/gs_fuse.py model/ts_encoders.py model/text_encoders.py \
  test_pretrained_model.py

python -m pytest tests/test_encoder_shapes.py -q
python -m pytest tests/test_pretrained_eval_metrics.py -q
```

These checks require the `gs-fuse` conda environment. Full training and checkpoint evaluation also require the external assets listed above.

## Migration Notes

Legacy training wrappers, optional analysis scripts, and compatibility shim modules have been removed from the public release. Use `train.py` for all training runs, `test_pretrained_model.py` for checkpoint evaluation, and import `GSFuse` directly from `model.gs_fuse` or `model`.
