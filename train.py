#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified GS-FUSE training entrypoint."""

import argparse
import warnings

import torch

from data.dataloader import event_set
from data.sliding_window_dataloader import sliding_window_set
from model.gs_fuse import GSFuse, train_three_stage

warnings.simplefilter("ignore")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.utils._pytree._register_pytree_node is deprecated",
)

torch.autograd.set_detect_anomaly(True)


def parse_args():
    p = argparse.ArgumentParser(description="Train GS-FUSE with selectable TS and text backbones")

    p.add_argument("--ts-backbone", choices=["moment", "kronos"], default="moment")
    p.add_argument("--text-backbone", choices=["llama", "phi"], default="llama")
    p.add_argument("--moment-path", type=str, default=None)
    p.add_argument("--kronos-model-path", type=str, default=None)
    p.add_argument("--kronos-tokenizer-path", type=str, default=None)
    p.add_argument("--text-model-path", type=str, required=True)

    p.add_argument("--event-dir", type=str, default="data/event")
    p.add_argument("--series-dir", type=str, default="data/series")
    p.add_argument("--output-path", type=str, required=True)

    p.add_argument("--series-id", type=str, default="SP500")
    p.add_argument("--event-id", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--pred-len", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--stride", type=int, default=400)

    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--stage1-epochs", type=int, default=2)
    p.add_argument("--stage2-epochs", type=int, default=2)
    p.add_argument("--contrastive-lambda", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=None)

    p.add_argument("--decoder-layers", type=int, default=3)
    p.add_argument("--decoder-heads", type=int, default=8)
    p.add_argument("--max-token-num", type=int, default=1024)
    p.add_argument("--use-ts-memory", action="store_true", default=False)

    p.add_argument("--finetune-ts-last-block", action="store_true", default=False)
    p.add_argument("--finetune-text-last-block", action="store_true", default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.ts_backbone == "moment" and not args.moment_path:
        raise ValueError("--moment-path is required when --ts-backbone moment")
    if args.ts_backbone == "kronos":
        if not args.kronos_model_path or not args.kronos_tokenizer_path:
            raise ValueError(
                "--kronos-model-path and --kronos-tokenizer-path are required when --ts-backbone kronos"
            )

    import model.gs_fuse as gs_fuse_mod

    gs_fuse_mod.FINETUNE_TS_LAST_BLOCK = args.finetune_ts_last_block
    if args.finetune_text_last_block is not None:
        gs_fuse_mod.FINETUNE_LLM_LAST_LAYER = args.finetune_text_last_block

    events = event_set(
        args.seq_len,
        args.pred_len,
        event_id=args.event_id,
        series_id=args.series_id,
        shuffle=False,
        batch_size=args.batch_size,
        scale=True,
        event_dir=args.event_dir,
        series_dir=args.series_dir,
    )

    sw_data = sliding_window_set(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        series_id=args.series_id,
        series_dir=args.series_dir,
        stride=1,
        batch_size=args.batch_size,
    )

    train_loader = events.train_loader
    test_loader = events.test_loader
    vali_loader = events.vali_loader

    model = GSFuse(
        text_model_path=args.text_model_path,
        ts_backbone=args.ts_backbone,
        text_backbone=args.text_backbone,
        moment_path=args.moment_path,
        kronos_model_path=args.kronos_model_path,
        kronos_tokenizer_path=args.kronos_tokenizer_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d=events.d,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        use_ts_memory=args.use_ts_memory,
        max_token_num=args.max_token_num,
    )

    train_three_stage(
        model,
        train_loader,
        test_loader,
        vali_loader,
        sw_train_loader=sw_data.train_loader,
        sw_vali_loader=sw_data.vali_loader,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        num_epochs=args.num_epochs,
        output_path=args.output_path,
        contrastive_lambda=args.contrastive_lambda,
        eval_every_steps=args.eval_every_steps,
    )


if __name__ == "__main__":
    main()
