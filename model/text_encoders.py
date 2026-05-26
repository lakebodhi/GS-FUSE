# -*- coding: utf-8 -*-
"""Pluggable text encoders for GS-FUSE (LLaMA-family and Phi-family)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load_causal_lm(name: str, use_4bit: bool = False):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float32,
        quantization_config=bnb_config if use_4bit else None,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return tok, model


class BaseTextEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def encode_tokens(self, text: str, window: int, stride: int, max_text_tokens: int, device: torch.device):
        """Return hidden token features [1, T, hidden] and mask [1, T]."""

    @abstractmethod
    def encode_instance(self, text: str, window: int, stride: int, device: torch.device) -> torch.Tensor:
        """Return pooled hidden vector [hidden]."""

    @abstractmethod
    def last_block_parameters(self):
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        ...

    @property
    @abstractmethod
    def llm(self):
        ...

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        self.freeze()
        for p in self.last_block_parameters():
            p.requires_grad = True


class HuggingFaceTextEncoder(BaseTextEncoder):
    """Shared chunked hidden-state extraction for causal LM text backbones."""

    def __init__(self, model_path: str, family: str = "llama", use_4bit: bool = False):
        super().__init__()
        self.family = family.lower()
        self._tokenizer, self.backbone = _load_causal_lm(model_path, use_4bit=use_4bit)
        cfg = getattr(self.backbone, "config", None)
        self._output_dim = int(getattr(cfg, "hidden_size", 3072))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def llm(self):
        return self.backbone

    @torch.no_grad()
    def _forward_chunk(self, chunk_ids, chunk_mask):
        out = self.backbone.model(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return out.hidden_states[-1][0].float()

    @torch.no_grad()
    def encode_instance(self, text: str, window: int, stride: int, device: torch.device) -> torch.Tensor:
        ids = self._tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        input_ids = ids["input_ids"][0]
        attn_mask_full = torch.ones_like(input_ids)
        embeddings = []
        start = 0
        while start < input_ids.shape[0]:
            end = min(start + window, input_ids.shape[0])
            chunk_ids = input_ids[start:end].unsqueeze(0).to(device)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0).to(device)
            h = self._forward_chunk(chunk_ids, chunk_mask)
            embeddings.append(h.mean(dim=0, keepdim=True))
            start += stride
        if not embeddings:
            return torch.zeros(self._output_dim, device=device)
        return torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=False).to(device)

    @torch.no_grad()
    def encode_tokens(
        self,
        text: str,
        window: int,
        stride: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_text_tokens,
            add_special_tokens=True,
            padding="max_length",
        )
        input_ids = ids["input_ids"][0].to(device)
        attn_mask_full = ids["attention_mask"][0].to(device)

        feats = []
        start = 0
        while start < input_ids.size(0):
            end = min(start + window, input_ids.size(0))
            chunk_ids = input_ids[start:end].unsqueeze(0)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0)
            feats.append(self._forward_chunk(chunk_ids, chunk_mask))
            start += stride

        if len(feats) == 0:
            tokens = torch.zeros(1, 1, self._output_dim, device=device)
            mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            return tokens, mask

        tok_h = torch.cat(feats, dim=0).unsqueeze(0)
        mask = torch.zeros(1, tok_h.size(1), dtype=torch.bool, device=device)
        return tok_h, mask

    def last_block_parameters(self):
        llm = self.backbone
        if llm is None:
            return []

        for path in [
            ("model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
        ]:
            cur = llm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                params = list(cur[-1].parameters())
                if params:
                    return params

        decoder_layer_types = []
        if self.family == "llama":
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                decoder_layer_types.append(LlamaDecoderLayer)
            except ImportError:
                pass
        elif self.family == "phi":
            for mod_name in (
                "transformers.models.phi.modeling_phi",
                "transformers.models.phi3.modeling_phi3",
            ):
                try:
                    mod = __import__(mod_name, fromlist=["PhiDecoderLayer", "Phi3DecoderLayer"])
                    for attr in ("PhiDecoderLayer", "Phi3DecoderLayer"):
                        if hasattr(mod, attr):
                            decoder_layer_types.append(getattr(mod, attr))
                except ImportError:
                    continue

        for layer_type in decoder_layer_types:
            candidates = [m for m in llm.modules() if isinstance(m, layer_type)]
            if candidates:
                return list(candidates[-1].parameters())

        print("[GS-FUSE] Warning: Could not find text encoder last block parameters")
        return []


class LlamaTextEncoder(HuggingFaceTextEncoder):
    def __init__(self, model_path: str, use_4bit: bool = False):
        super().__init__(model_path, family="llama", use_4bit=use_4bit)


class PhiTextEncoder(HuggingFaceTextEncoder):
    def __init__(self, model_path: str, use_4bit: bool = False):
        super().__init__(model_path, family="phi", use_4bit=use_4bit)


def build_text_encoder(backbone: str, model_path: str, use_4bit: bool = False) -> BaseTextEncoder:
    backbone = backbone.lower()
    if backbone == "llama":
        return LlamaTextEncoder(model_path, use_4bit=use_4bit)
    if backbone == "phi":
        return PhiTextEncoder(model_path, use_4bit=use_4bit)
    raise ValueError(f"Unknown text_backbone: {backbone}")
