# -*- coding: utf-8 -*-
"""Pluggable time-series encoders for GS-FUSE (MOMENT and Kronos)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from momentfm import MOMENTPipeline
except ImportError:
    MOMENTPipeline = None

try:
    from model.kronos import Kronos, KronosTokenizer, KronosAdapter
except Exception:
    Kronos = None
    KronosTokenizer = None
    KronosAdapter = None


class BaseTSEncoder(nn.Module, ABC):
    """Common contract for GS-FUSE time-series backbones."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def encode_per_timestep(self, x_cf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_cf: [B, d, L] channels-first -> token_emb [B, L, D], inst_emb [B, D]."""

    @abstractmethod
    def last_block_parameters(self):
        ...

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        self.freeze()
        for p in self.last_block_parameters():
            p.requires_grad = True


class MomentTSEncoder(BaseTSEncoder):
    def __init__(self, model_path: str, device: torch.device, cache_dir: str = "moment_model"):
        super().__init__()
        if MOMENTPipeline is None:
            raise ImportError("momentfm is required for MomentTSEncoder")
        self.device = device
        self.backbone = MOMENTPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            model_kwargs={"task_name": "embedding"},
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)
        self.backbone.init()
        self._output_dim = 1024

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def encode_per_timestep(self, x_cf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        output = self.backbone.embed(x_enc=x_cf, reduction="none")
        patch_emb = output.embeddings

        _b, _c, n_patches, _d = patch_emb.shape
        l = x_cf.size(2)
        patch_emb_avg = patch_emb.mean(dim=1)

        if n_patches != l:
            token_emb = torch.nn.functional.interpolate(
                patch_emb_avg.transpose(1, 2),
                size=l,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            token_emb = patch_emb_avg

        inst_emb = token_emb.mean(dim=1)
        return token_emb, inst_emb

    def last_block_parameters(self):
        mm = self.backbone
        for path in [
            ("transformer", "layers"),
            ("encoder", "layers"),
            ("layers",),
            ("backbone", "layers"),
        ]:
            cur = mm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                return list(cur[-1].parameters())

        candidates = [m for m in mm.modules() if isinstance(m, torch.nn.TransformerEncoderLayer)]
        if candidates:
            return list(candidates[-1].parameters())

        last_with_params = None
        for m in mm.modules():
            if any(True for _ in m.parameters(recurse=False)):
                last_with_params = m
        return list(last_with_params.parameters()) if last_with_params is not None else []


class KronosTSEncoder(BaseTSEncoder):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: torch.device,
        seq_len: int,
        d: int,
        fallback_dim: int = 1024,
    ):
        super().__init__()
        if Kronos is None or KronosTokenizer is None or KronosAdapter is None:
            raise ImportError("model.kronos is required for KronosTSEncoder")

        self.device = device
        self.kronos_tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.backbone = Kronos.from_pretrained(model_path)
        self.backbone.to(device).eval()
        self._adapter = KronosAdapter(model=self.backbone, tokenizer=self.kronos_tokenizer, device=str(device))

        try:
            with torch.no_grad():
                dummy = torch.zeros(1, seq_len, d, device=device)
                tok_emb, _inst = self._adapter.embed(
                    dummy,
                    x_timestamp=None,
                    which="context",
                    fuse="avg",
                    pool="mean",
                    normalize=False,
                )
                self._output_dim = int(tok_emb.shape[-1])
        except Exception:
            self._output_dim = fallback_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def encode_per_timestep(self, x_cf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_blc = x_cf.transpose(1, 2).contiguous()
        token_emb, inst_emb = self._adapter.embed(
            x_blc,
            x_timestamp=None,
            which="context",
            fuse="avg",
            pool="mean",
            normalize=False,
        )
        if isinstance(inst_emb, torch.Tensor) and inst_emb.dim() == 3 and inst_emb.size(1) == 1:
            inst_emb = inst_emb[:, 0, :]
        return token_emb, inst_emb

    def last_block_parameters(self):
        mm = self.backbone
        for path in [
            ("transformer", "layers"),
            ("encoder", "layers"),
            ("layers",),
            ("backbone", "layers"),
        ]:
            cur = mm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                return list(cur[-1].parameters())

        candidates = [m for m in mm.modules() if isinstance(m, torch.nn.TransformerEncoderLayer)]
        if candidates:
            return list(candidates[-1].parameters())

        last_with_params = None
        for m in mm.modules():
            if any(True for _ in m.parameters(recurse=False)):
                last_with_params = m
        return list(last_with_params.parameters()) if last_with_params is not None else []


def build_ts_encoder(
    backbone: str,
    device: torch.device,
    *,
    moment_path: Optional[str] = None,
    kronos_model_path: Optional[str] = None,
    kronos_tokenizer_path: Optional[str] = None,
    seq_len: int = 35,
    d: int = 4,
) -> BaseTSEncoder:
    backbone = backbone.lower()
    if backbone == "moment":
        if not moment_path:
            raise ValueError("moment_path is required when ts_backbone='moment'")
        return MomentTSEncoder(moment_path, device)
    if backbone == "kronos":
        if not kronos_model_path or not kronos_tokenizer_path:
            raise ValueError("kronos_model_path and kronos_tokenizer_path are required when ts_backbone='kronos'")
        return KronosTSEncoder(
            kronos_model_path,
            kronos_tokenizer_path,
            device,
            seq_len=seq_len,
            d=d,
        )
    raise ValueError(f"Unknown ts_backbone: {backbone}")
