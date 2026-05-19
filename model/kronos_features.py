# kronos_features.py
import torch
import torch.nn.functional as F

class KronosFeatureExtractor:
    """
    Feature extractor to mimic MOMENT-style outputs with Kronos:
      - token_features_cont:   continuous per-step features (pre-quantization, from tokenizer encoder)
      - token_features_disc:   discrete-token contextual features (from Kronos transformer)
      - instance_features_*:   pooled [B, D] vectors from the token features
    """
    def __init__(self, tokenizer, kronos, pool='mean'):
        """
        tokenizer: KronosTokenizer (from your script)
        kronos   : Kronos model (from your script)
        pool     : 'mean' | 'cls' | 'attn' (simple attention pooling)
        """
        self.tokenizer = tokenizer
        self.kronos = kronos
        assert pool in ('mean','cls','attn')
        self.pool = pool
        # tiny attention vector for attn pooling
        self.attn_vec = torch.nn.Parameter(torch.randn(self.kronos.d_model) * 0.01, requires_grad=True)

    @torch.no_grad()
    def _pool(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [B, T] with 1 for real, 0 for pad (optional)
        """
        if mask is not None:
            # expand for broadcasting
            mask_f = mask.float().unsqueeze(-1)   # [B, T, 1]

        if self.pool == 'mean':
            if mask is None:
                return x.mean(dim=1)
            else:
                denom = mask_f.sum(dim=1).clamp_min(1.0)  # [B,1]
                return (x * mask_f).sum(dim=1) / denom

        if self.pool == 'cls':
            # use the first time step as a [CLS]-like summary
            return x[:, 0, :]

        # attn pooling
        # score = x @ a, softmax over T, sum
        a = self.attn_vec.to(x.device)  # [D]
        score = torch.matmul(x, a)      # [B, T]
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(score, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * w).sum(dim=1)                  # [B, D]

    @torch.no_grad()
    def token_features_cont(self, x_cont):
        """
        Continuous token features from the tokenizer encoder (pre-quantization).
        x_cont: [B, T, d_in] continuous series (already normalized like your predictor)
        Returns: [B, T, d_model]
        """
        # This mirrors KronosTokenizer.forward up to quant_embed input.
        z = self.tokenizer.embed(x_cont)                  # [B, T, d_model]
        for layer in self.tokenizer.encoder:
            z = layer(z)                                  # [B, T, d_model]
        return z

    @torch.no_grad()
    def token_features_disc(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Discrete contextual token features from Kronos (post-tokenization, model’s internal states).
        Returns: [B, T, d_model]
        """
        # Kronos.decode_s1 returns (s1_logits, context), where `context` is the normalized transformer output.
        _, context = self.kronos.decode_s1(s1_ids, s2_ids, stamp=stamp, padding_mask=padding_mask)
        return context  # [B, T, d_model]

    @torch.no_grad()
    def instance_features_cont(self, x_cont, mask=None):
        """
        Instance-level feature by pooling continuous token features.
        Returns: [B, d_model]
        """
        tok = self.token_features_cont(x_cont)            # [B, T, D]
        return self._pool(tok, mask)

    @torch.no_grad()
    def instance_features_disc(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Instance-level feature by pooling discrete contextual token features.
        Returns: [B, d_model]
        """
        tok = self.token_features_disc(s1_ids, s2_ids, stamp, padding_mask)  # [B, T, D]
        return self._pool(tok, padding_mask)
