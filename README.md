# GS-FUSE

**GS-FUSE: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting**

GS-FUSE is a multimodal event-driven financial forecasting framework that learns when macro-financial text provides predictive value beyond historical prices, then fuses that signal with time-series representations for multi-horizon market forecasting.

- Paper: [arXiv:2605.28520](https://arxiv.org/abs/2605.28520)
- Proceedings paper: Coming Soon
- Trained GS-FUSE model checkpoints: [Alipan](https://www.alipan.com/s/NBvf1wWskh4)
- Trained model mirror: [Google Drive for Trained Models](https://drive.google.com/drive/folders/1-Mx2VeBnB8YKGrDCaZSNkbc_7bj6y5Gj?usp=sharing)
- Dataset: [Google Drive for Dataset](https://drive.google.com/file/d/1fH436rkOHVYIG2JPROrFTXXemnY7JPdB/view?usp=sharing)

## Paper Overview

Accurately forecasting the market impact of salient financial events requires deciding when event text is actually informative beyond historical price patterns. GS-FUSE addresses this with a Granger-supervised, causal-aware gated fusion module that opens toward event text only when it provides incremental predictive value, plus a multi-granularity alignment mechanism that aligns both event-level representations and fine-grained textual cues with future market trajectories.

The paper is available on arXiv: [GS-FUSE: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting](https://arxiv.org/abs/2605.28520).

## Model Architecture

GS-FUSE is designed as a plug-and-play adapter on top of off-the-shelf time-series foundation models and large language models.

```text
Event text
  -> LLaMA / Phi text encoder
  -> fine-grained text tokens
  -> multi-granularity alignment -----+
                                      |
Historical price window              +-> Granger-supervised gated fusion -> decoder -> multi-horizon forecast
  -> MOMENT / Kronos TS encoder       |
  -> time-series tokens --------------+
```

The core components are:

- Pluggable text encoder: LLaMA-family or Phi-family models encode event summaries and counterfactual event text.
- Pluggable time-series encoder: MOMENT or Kronos encodes historical market windows.
- Granger-supervised gated fusion: learns whether event text contributes incremental predictive signal beyond time-series history.
- Multi-granularity alignment: aligns high-level event representations and fine-grained text/time-series cues with future market trajectories.
- Forecast decoder: maps fused representations to multi-step financial forecasts.

Training runs in three stages:

1. TS-only pretraining with the sliding-window loader.
2. Text-only pretraining with the event loader.
3. Full multimodal training with gated fusion and alignment objectives.

## Dataset Access

Download the GS-FUSE dataset from [Google Drive for Dataset](https://drive.google.com/file/d/1fH436rkOHVYIG2JPROrFTXXemnY7JPdB/view?usp=sharing).

The dataset combines CAMEF event data with FNDPID financial time-series data. After extraction, place or symlink the files so the project can find:

```text
data/event/     # event text files by type/date
data/series/    # per-asset time-series CSV files
```

Large datasets, model weights, checkpoints, outputs, and logs are intentionally ignored by git.

## Code Quick Start

### Environment

Create the unified conda environment:

```bash
conda env create -f environment.yml
conda activate gs-fuse
```

If the environment already exists:

```bash
conda env update -n gs-fuse -f environment.yml --prune
conda activate gs-fuse
```

`environment.yml` supports MOMENT and Kronos time-series encoders, plus LLaMA-family and Phi-family text encoders.

### Evaluate A Trained Model

Download a trained checkpoint from [Alipan](https://www.alipan.com/s/NBvf1wWskh4), or the [Google Drive trained-model mirror](https://drive.google.com/drive/folders/1-Mx2VeBnB8YKGrDCaZSNkbc_7bj6y5Gj?usp=sharing), then evaluate it with `test_pretrained_model.py`.

Example for a MOMENT + LLaMA checkpoint on SP500:

```bash
python test_pretrained_model.py \
  --model-path /path/to/downloaded/checkpoint/best_model.pth \
  --ts-backbone moment \
  --text-backbone llama \
  --moment-path /path/to/MOMETN-1-large \
  --text-model-path /path/to/llama3-2-3B \
  --event-dir data/event \
  --series-dir data/series \
  --series-id SP500 \
  --seq-len 35 \
  --pred-len 140 \
  --batch-size 16 \
  --eval-horizons 35 70 140 \
  --output-json outputs/sp500_metrics.json
```

The evaluation script reports forecasting metrics (`MSE`, `MAE`) and financial metrics including directional hit rate, Sharpe ratio, investment return ratio, maximum drawdown, and Calmar ratio across the requested horizons.

For Kronos checkpoints, replace `--moment-path` with:

```bash
--kronos-model-path /path/to/Kronos-base \
--kronos-tokenizer-path /path/to/Kronos-Tokenizer-base
```

For Phi checkpoints, set:

```bash
--text-backbone phi \
--text-model-path /path/to/Phi-3.5-mini-instruct
```

Use `--inverse-transform` if endpoint financial metrics should be computed in original scaler space instead of scaled space.

### Train From Scratch

The default training configuration uses:

- MOMENT time-series encoder (`--ts-backbone moment`)
- LLaMA-family text encoder (`--text-backbone llama`)
- SP500 series (`--series-id SP500`)
- `seq_len=35`, `pred_len=140`
- three-stage training with 2 epochs per stage by default

Run training with default settings:

```bash
python train.py \
  --moment-path /path/to/MOMETN-1-large \
  --text-model-path /path/to/llama3-2-3B \
  --event-dir data/event \
  --series-dir data/series \
  --output-path outputs/gs_fuse_sp500_default
```

Training writes checkpoints and logs under the selected output directory, including:

```text
outputs/gs_fuse_sp500_default/best_model.pth
outputs/gs_fuse_sp500_default/lastest_model.pth
outputs/gs_fuse_sp500_default/log.txt
outputs/gs_fuse_sp500_default/tokenizer/
```

To change the default forecast or dataset:

```bash
python train.py \
  --moment-path /path/to/MOMETN-1-large \
  --text-model-path /path/to/llama3-2-3B \
  --event-dir data/event \
  --series-dir data/series \
  --series-id NASDAQ \
  --seq-len 35 \
  --pred-len 140 \
  --batch-size 32 \
  --output-path outputs/gs_fuse_nasdaq_len70
```

To train with Kronos instead of MOMENT:

```bash
python train.py \
  --ts-backbone kronos \
  --text-backbone llama \
  --kronos-model-path /path/to/Kronos-base \
  --kronos-tokenizer-path /path/to/Kronos-Tokenizer-base \
  --text-model-path /path/to/llama3-2-3B \
  --event-dir data/event \
  --series-dir data/series \
  --output-path outputs/gs_fuse_kronos_llama
```

## Supported Backbones

- Time-series encoders: MOMENT and Kronos via `--ts-backbone moment|kronos`
- Text encoders: LLaMA-family and Phi-family via `--text-backbone llama|phi`

Any supported time-series encoder can be paired with any supported text encoder.

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

Full training and checkpoint evaluation require the external assets listed above.

## Citation

If you find this repository useful, please cite the arXiv paper:

```bibtex
@article{zhang2026gsfuse,
  title = {GS-FUSE: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting},
  author = {Zhang, Yang and Chun, En and Mao, Ziyun and Wu, Yulu and Wang, Jun},
  journal = {arXiv preprint arXiv:2605.28520},
  year = {2026},
  eprint = {2605.28520},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI}
}
```

Proceedings citation: Coming Soon.
