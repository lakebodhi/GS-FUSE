import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    KronosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the KronosTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized,


    def indices_to_bits(self, x, half=False):
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0] # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long) # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0 # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0 # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1) # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0 # Extract bits

        x = x.float() * 2 - 1 # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (self.codebook_dim ** 0.5) # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos Model.

    Args:
        s1_bits (int): Number of bits for pre tokens.
        s2_bits (int): Number of bits for post tokens.
        n_layers (int): Number of Transformer blocks.
        d_model (int): Dimension of the model's embeddings and hidden states.
        n_heads (int): Number of attention heads in the MultiheadAttention layers.
        ff_dim (int): Dimension of the feedforward network in the Transformer blocks.
        ffn_dropout_p (float): Dropout probability for the feedforward network.
        attn_dropout_p (float): Dropout probability for the attention layers.
        resid_dropout_p (float): Dropout probability for residual connections.
        token_dropout_p (float): Dropout probability for token embeddings.
        learn_te (bool): Whether to use learnable temporal embeddings.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing for s1 decoding. Defaults to False.
            s1_targets (torch.Tensor, optional): Target s1 token IDs for teacher forcing. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: Logits for s2 token predictions, conditioned on s1. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def ts_embedding(
            self,
            s1_ids: torch.Tensor,
            s2_ids: torch.Tensor,
            stamp: torch.Tensor = None,
            padding_mask: torch.Tensor = None,
            *,
            which: str = "context",  # "context" | "cond" | "both"
            teacher_force_ids: torch.Tensor = None,  # if provided, used to build "cond" path
            fuse: str = "avg",  # when which="both": "avg" | "gate" | "cat-proj"
            pool: str = "mean",  # "mean" | "cls" | "attn"
            normalize: bool = False
    ):
        """
        Extract token-level and instance-level TS embeddings from Kronos.

        - "context": features after transformer + norm, before heads (x).
        - "cond": features after dependency layer conditioned on s1 (x2).
                  If teacher_force_ids is given, use those; otherwise uses argmax over s1 logits.
        - "both": fuse "context" and "cond" by 'avg' | 'gate' | 'cat-proj'.

        Returns:
            token_emb: [B, T, d_model]
            inst_emb:  [B, d_model]
            aux: dict with keys:
                 {"context": x, "cond": x2, "s1_logits": s1_logits, "used_s1_ids": used_ids}
        """
        # 1) Build context path (same as decode_s1 but without returning logits yet)
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)  # [B,T,d_model]

        # 2) Compute s1 logits once (for possible conditioning)
        s1_logits = self.head(x)  # [B,T,s1_vocab]

        # 3) Build conditioned path x2 only if needed
        x2 = None
        used_ids = None
        if which in ("cond", "both"):
            if teacher_force_ids is not None:
                used_ids = teacher_force_ids
            else:
                # greedy argmax over s1
                used_ids = s1_logits.detach().argmax(dim=-1)  # [B,T]
            sibling_embed = self.embedding.emb_s1(used_ids)
            x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)  # [B,T,d_model]

        # 4) Optional normalization before fusion/pooling for scale stability
        if normalize:
            if self._ts_ln is None:
                self._ts_ln = nn.LayerNorm(self.d_model, elementwise_affine=True).to(x.device)
            x = self._ts_ln(x)
            if x2 is not None:
                x2 = self._ts_ln(x2)

        # 5) Select/Fuse token-level features
        if which == "context":
            token_emb = x
        elif which == "cond":
            token_emb = x2
        elif which == "both":
            if fuse == "avg":
                token_emb = torch.stack((x, x2), dim=0).mean(dim=0)
            elif fuse == "gate":
                if self._ts_gate is None:
                    self._ts_gate = nn.Parameter(torch.zeros(1, device=x.device))  # sigmoid(0)=0.5
                g = torch.sigmoid(self._ts_gate)
                token_emb = g * x2 + (1.0 - g) * x
            elif fuse == "cat-proj":
                if self._ts_cat_proj is None:
                    self._ts_cat_proj = nn.Linear(2 * self.d_model, self.d_model).to(x.device)
                token_emb = self._ts_cat_proj(torch.cat([x, x2], dim=-1))
            else:
                raise ValueError(f"Unknown fuse='{fuse}'")
        else:
            raise ValueError(f"Unknown which='{which}'")

        # 6) Pool to instance-level feature
        if pool == "mean":
            inst_emb = token_emb.mean(dim=1)
        elif pool == "cls":
            inst_emb = token_emb[:, 0, :]
        elif pool == "attn":
            if self._ts_attn_pool is None:
                self._ts_attn_pool = nn.Linear(self.d_model, 1).to(token_emb.device)
            attn_scores = self._ts_attn_pool(token_emb).squeeze(-1)  # [B,T]
            attn_w = F.softmax(attn_scores, dim=-1)  # [B,T]
            inst_emb = torch.bmm(attn_w.unsqueeze(1), token_emb).squeeze(1)  # [B,d_model]
        else:
            raise ValueError(f"Unknown pool='{pool}'")

        aux = {
            "context": x,  # [B,T,d_model]
            "cond": x2,  # [B,T,d_model] or None
            "s1_logits": s1_logits,  # [B,T,s1_vocab]
            "used_s1_ids": used_ids  # [B,T] or None
        }
        return token_emb, inst_emb, aux

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Decodes only the s1 tokens.

        This method performs a forward pass to predict only s1 tokens. It returns the s1 logits
        and the context representation from the Transformer, which can be used for subsequent s2 decoding.

        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - context: Context representation from the Transformer. Shape: [batch_size, seq_len, d_model]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        Decodes the s2 tokens, conditioned on the context and s1 tokens.

        This method decodes s2 tokens based on a pre-computed context representation (typically from `decode_s1`)
        and the s1 token IDs. It uses the dependency-aware layer and the conditional s2 head to predict s2 tokens.

        Args:
            context (torch.Tensor): Context representation from the transformer (output of decode_s1).
                                     Shape: [batch_size, seq_len, d_model]
            s1_ids (torch.torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: s2 logits. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):

            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i

            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            torch.cuda.empty_cache()

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class KronosPredictor:

    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):

        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # Fill missing volume with zeros
            df[self.amt_vol] = 0.0  # Fill missing amount with zeros
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df


    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        Perform parallel (batch) prediction on multiple time series. All series must have the same historical length and prediction length (pred_len).

        Args:
            df_list (List[pd.DataFrame]): List of input DataFrames, each containing price columns and optional volume/amount columns.
            x_timestamp_list (List[pd.DatetimeIndex or Series]): List of timestamps corresponding to historical data, length should match the number of rows in each DataFrame.
            y_timestamp_list (List[pd.DatetimeIndex or Series]): List of future prediction timestamps, length should equal pred_len.
            pred_len (int): Number of prediction steps.
            T (float): Sampling temperature.
            top_k (int): Top-k filtering threshold.
            top_p (float): Top-p (nucleus sampling) threshold.
            sample_count (int): Number of parallel samples per series, automatically averaged internally.
            verbose (bool): Whether to display autoregressive progress.

        Returns:
            List[pd.DataFrame]: List of prediction results in the same order as input, each DataFrame contains
                                `open, high, low, close, volume, amount` columns, indexed by corresponding `y_timestamp`.
        """
        # Basic validation
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        # Require all series to have consistent historical and prediction lengths for batch processing
        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)           # (B, seq_len, feat)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32) # (B, seq_len, time_feat)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32) # (B, pred_len, time_feat)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)
        # preds: (B, pred_len, feat)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs

    @torch.no_grad()
    def ts_embed(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        *,
        which: str = "context",   # "context" | "cond" | "both"
        fuse: str = "avg",        # when which="both": "avg" | "gate" | "cat-proj"
        pool: str = "mean",       # "mean" | "cls" | "attn"
        normalize: bool = False,
        return_numpy: bool = True
    ):
        """
        Produce TS embeddings for a single series using the same preprocessing as predict().
        Returns (token_emb [T, d_model], inst_emb [d_model], aux dict).
        """
        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price/volume columns.")

        # timestamps -> stamp features
        x_time_df = calc_time_stamps(x_timestamp)

        # values -> normalize/clip exactly like predict()
        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        # to tensors on device
        x_tensor = torch.from_numpy(x_norm[None, :, :]).to(self.device)          # [1,T,feat]
        x_stamp_tensor = torch.from_numpy(x_stamp[None, :, :]).to(self.device)   # [1,T,time_feat]

        # tokenize -> (s1_ids, s2_ids)
        s_ids = self.tokenizer.encode(x_tensor, half=True)  # returns tuple/list of 2 tensors
        s1_ids, s2_ids = s_ids[0], s_ids[1]                 # both [1,T] int

        # call model.ts_embedding
        token_emb, inst_emb, aux = self.model.ts_embedding(
            s1_ids=s1_ids,
            s2_ids=s2_ids,
            stamp=x_stamp_tensor,
            padding_mask=None,
            which=which,
            fuse=fuse,
            pool=pool,
            normalize=normalize
        )  # token_emb [1,T,d_model], inst_emb [1,d_model]

        # squeeze batch dim
        token_emb = token_emb.squeeze(0).contiguous()
        inst_emb = inst_emb.squeeze(0).contiguous()

        if return_numpy:
            return token_emb.detach().cpu().numpy(), inst_emb.detach().cpu().numpy(), {
                k: (v.detach().cpu().numpy() if torch.is_tensor(v) and v is not None else v)
                for k, v in aux.items()
            }
        return token_emb, inst_emb, aux

    # @torch.no_grad()
    def ts_embed_batch(
            self,
            df_list,
            *,
            which: str = "context",
            fuse: str = "avg",
            pool: str = "mean",
            normalize: bool = False,
            return_numpy: bool = True
    ):
        """
        Batched embeddings. All series must share the same historical length.
        Accepts:
          - pandas.DataFrame with price columns named as self.price_cols (case-insensitive),
            and optionally self.vol_col / self.amt_vol; OR
          - pandas.DataFrame without meaningful column names but ordered as:
                [open, high, low, close, (optional) volume, (optional) amount]
          - numpy.ndarray or torch.Tensor shaped [T, F] with F in {4,5,6} matching the same order.
        Returns:
          token_emb [B, T, d_model], inst_emb [B, d_model], aux dict
        """

        def _standardize_one_to_ohlcva(x, idx: int) -> np.ndarray:
            """
            Convert one item (DataFrame / ndarray / tensor) to a standardized numpy array
            with columns [open, high, low, close, volume, amount]. Missing volume/amount
            will be filled (volume->0, amount->volume*avg_price or 0 if volume missing).
            """
            # ---- Case A: torch.Tensor / np.ndarray ----
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                arr = x
            # ---- Case B: pandas.DataFrame ----
            else:
                if not hasattr(x, "values"):
                    raise ValueError(f"Item at index {idx} must be DataFrame/ndarray/tensor, got {type(x)}")
                df = x

                # Try column-name based mapping (case-insensitive match)
                cols_lower = [str(c).lower() for c in df.columns]
                want = [c.lower() for c in self.price_cols]  # expected names for price cols
                try_name_map = {}
                for name in want:
                    if name in cols_lower:
                        try_name_map[name] = cols_lower.index(name)

                has_named_price = len(try_name_map) == len(self.price_cols)

                if has_named_price:
                    # Build ordered indices for price cols
                    price_idx = [try_name_map[c.lower()] for c in self.price_cols]
                    # Volume / Amount (optional)
                    vol_idx = cols_lower.index(self.vol_col.lower()) if self.vol_col.lower() in cols_lower else None
                    amt_idx = cols_lower.index(self.amt_vol.lower()) if self.amt_vol.lower() in cols_lower else None

                    cols_pick = price_idx + ([vol_idx] if vol_idx is not None else []) + (
                        [amt_idx] if amt_idx is not None else [])
                    arr = df.iloc[:, cols_pick].values.astype(np.float32)

                else:
                    # No reliable names -> rely on positional convention
                    # Expect at least 4 columns: OHLC; 5th (optional)=Volume; 6th (optional)=Amount
                    if df.shape[1] < 4:
                        raise ValueError(f"DataFrame at index {idx} must have >=4 columns when unnamed/unknown.")
                    arr = df.iloc[:, :min(df.shape[1], 6)].values.astype(np.float32)

            # At this point `arr` is [T, F], F in {4,5,6} in positional order if unnamed
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"Series at index {idx} must be [T, F>=4], got shape {arr.shape}.")

            T, F = arr.shape
            # Slice price block assuming first 4 are OHLC (positional or mapped)
            price = arr[:, :4].astype(np.float32)

            # Volume
            if F >= 5:
                volume = arr[:, 4].astype(np.float32)
            else:
                volume = np.zeros((T,), dtype=np.float32)

            # Amount
            if F >= 6:
                amount = arr[:, 5].astype(np.float32)
            else:
                # If amount missing: amount = volume * average_price
                avg_price = price.mean(axis=1).astype(np.float32)
                amount = (volume * avg_price).astype(np.float32)

            out = np.concatenate([price, volume[:, None], amount[:, None]], axis=1)  # [T, 6]
            return out

        x_list = []
        seq_lens = []

        for i, item in enumerate(df_list):
            # Convert to standardized [T, 6] = [O,H,L,C,V,Amount]
            x = _standardize_one_to_ohlcva(item, idx=i)  # [T,6]

            # Per-series normalize (feature-wise)
            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip).astype(np.float32)

            x_list.append(x_norm)
            seq_lens.append(x_norm.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Batch requires consistent historical lengths, got {seq_lens}")

        # Batch -> torch [B, T, F]
        x_batch = torch.from_numpy(np.stack(x_list, 0).astype(np.float32)).to(self.device)

        # tokenize batch
        s_ids = self.tokenizer.encode(x_batch, half=True)  # expected to return tuple/list of ids
        s1_ids, s2_ids = s_ids[0], s_ids[1]  # [B, T]

        # model embeddings
        token_emb, inst_emb, aux = self.model.ts_embedding(
            s1_ids=s1_ids,
            s2_ids=s2_ids,
            padding_mask=None,
            which=which,
            fuse=fuse,
            pool=pool,
            normalize=normalize
        )  # token_emb [B,T,d_model], inst_emb [B,d_model]

        if return_numpy:
            return (
                token_emb.detach().cpu().numpy(),
                inst_emb.detach().cpu().numpy(),
                {k: (v.detach().cpu().numpy() if torch.is_tensor(v) and v is not None else v) for k, v in aux.items()}
            )
        return token_emb, inst_emb, aux


class KronosAdapter:
    """
# Thin wrapper to provide TS embeddings to GS-FUSE, mimicking a 'feature pipeline'.
    Usage:
        adapter = KronosAdapter(kronos_model, kronos_tokenizer, device="cuda:0", max_context=512, clip=5)
        token_emb, inst_emb = adapter.embed(x_seq, x_timestamp)
    """

    def __init__(self, model: Kronos, tokenizer: KronosTokenizer,
                 device="cuda:0", max_context=512, clip=5):
        self.pred = KronosPredictor(model, tokenizer, device=device, max_context=max_context, clip=clip)

    # @torch.no_grad()
    def embed(self, x_seq: torch.Tensor, x_timestamp: pd.Series = None,
              which: str = "context", fuse: str = "avg",
              pool: str = "mean", normalize: bool = False):
        """
        x_seq: [B, T, F] float tensor on CPU or device
        x_timestamp: pandas Series of length T (per-batch identical stamps supported)
        Returns:
            token_emb: [B, T, d_model], inst_emb: [B, d_model]
        """
        # Use the batch wrapper already in your KronosPredictor
        # Build the inputs as lists to reuse the exact preprocessing
        B, T, F = x_seq.shape
        df_list, ts_list = [], []
        for b in range(B):
            df_b = pd.DataFrame(x_seq[b].cpu().numpy(),
                                columns=self.pred.price_cols + [self.pred.vol_col, self.pred.amt_vol]
                                if F == 6 else [f"f{i}" for i in range(F)])
            # If your features aren’t exactly OHLCVA, KronosPredictor will still normalize/clip safely;
            # but for best results, map your 6 columns to the names above.
            df_list.append(df_b)
            if x_timestamp is not None:
                ts_list.append(x_timestamp)  # same stamps for all items or pass per-item series

        # Reuse the ts-embed path we discussed
        token_emb, inst_emb, _ = self.pred.ts_embed_batch(
            df_list=df_list,
            which=which, fuse=fuse, pool=pool, normalize=normalize,
            return_numpy=False
        )
        return token_emb, inst_emb
