# -*- coding: utf-8 -*-
"""Shape-contract smoke tests for GS-FUSE encoders (no checkpoint required for imports)."""

import torch

from model.text_encoders import LlamaTextEncoder, PhiTextEncoder, build_text_encoder
from model.ts_encoders import MomentTSEncoder, KronosTSEncoder, build_ts_encoder


def test_build_ts_encoder_factory_unknown():
    try:
        build_ts_encoder("unknown", torch.device("cpu"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_text_encoder_factory_unknown():
    try:
        build_text_encoder("unknown", "/tmp/model")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encoder_classes_exist():
    assert MomentTSEncoder is not None
    assert KronosTSEncoder is not None
    assert LlamaTextEncoder is not None
    assert PhiTextEncoder is not None
