"""
SigLIP 2 — built from scratch in PyTorch.

Architecture
------------
  VisionEncoder  : Patch embedding → Transformer blocks → CLS token projection
  TextEncoder    : Token embedding → Transformer blocks → EOS token projection
  SigLIP2Model   : Combines both, learns a temperature + bias scalar,
                   computes the sigmoid (pairwise) contrastive loss.

Loss (SigLoss)
--------------
  Unlike CLIP which uses softmax cross-entropy, SigLIP treats every
  (image, text) pair independently with a binary sigmoid loss:

      L = -sum( labels * log(sigma(logits)) +
                (1-labels) * log(1-sigma(logits)) ) / n²

  where labels[i,j] = +1 if i==j (positive pair) else -1.
  Division is by n² (total number of pairs), not n.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class LayerNorm(nn.LayerNorm):
    """Thin wrapper — keeps code readable."""
    pass


# ─────────────────────────────────────────────
#  Multi-Head Self-Attention
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        # Project → split into Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                    # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)

        if attn_mask is not None:
            # attn_mask: (B, N) boolean — True = PAD, should be ignored
            attn_mask = attn_mask[:, None, None, :]      # broadcast over heads & queries
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ─────────────────────────────────────────────
#  Feed-Forward Block
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  Transformer Block (pre-norm)
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
#  Vision Encoder  (ViT-style)
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects each to embed_dim."""
    def __init__(self, img_size: int, patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # A single conv with kernel=stride=patch_size is equivalent to
        # flattening each patch then applying a linear layer.
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionEncoder(nn.Module):
    def __init__(self,
                 img_size:    int   = 224,
                 patch_size:  int   = 16,
                 in_channels: int   = 3,
                 embed_dim:   int   = 768,
                 depth:       int   = 12,
                 num_heads:   int   = 12,
                 mlp_ratio:   float = 4.0,
                 proj_dim:    int   = 512,
                 dropout:     float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        # Learnable CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)

        # Linear projection to shared embedding space
        self.head = nn.Linear(embed_dim, proj_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            image embeddings: (B, proj_dim)  — L2-normalised
        """
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)                         # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                          # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                          # CLS token
        return F.normalize(self.head(cls_out), dim=-1)


# ─────────────────────────────────────────────
#  Text Encoder
# ─────────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size:  int   = 16000,
                 max_seq_len: int   = 64,
                 embed_dim:   int   = 512,
                 depth:       int   = 12,
                 num_heads:   int   = 8,
                 mlp_ratio:   float = 4.0,
                 proj_dim:    int   = 512,
                 pad_token_id: int  = 1,
                 dropout:     float = 0.0):
        super().__init__()
        self.pad_token_id = pad_token_id

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, proj_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)  1 = real token, 0 = padding
        Returns:
            text embeddings: (B, proj_dim)  — L2-normalised
        """
        B, L = input_ids.shape
        x = self.token_embed(input_ids)                            # (B, L, D)
        x = self.pos_drop(x + self.pos_embed[:, :L])

        # Build padding mask for attention: True where PAD
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)                       # (B, L)
        else:
            pad_mask = (input_ids == self.pad_token_id)

        for blk in self.blocks:
            x = blk(x, attn_mask=pad_mask)

        x = self.norm(x)

        # Use the EOS (last non-pad) token as the sentence representation
        if attention_mask is not None:
            seq_lens  = attention_mask.sum(dim=1) - 1              # index of last real token
        else:
            seq_lens  = (input_ids != self.pad_token_id).sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        eos_out = x[torch.arange(B, device=x.device), seq_lens]   # (B, D)

        return F.normalize(self.head(eos_out), dim=-1)


# ─────────────────────────────────────────────
#  Sigmoid Contrastive Loss
# ─────────────────────────────────────────────

class SigLoss(nn.Module):
    """
    SigLIP pairwise sigmoid loss.
    Labels are +1 on the diagonal (positive pairs) and -1 off-diagonal.

    FIX: divide by n² (total number of pairs), not n.
    The original divided by n which caused the loss to scale with batch size
    and produced abnormally large gradients.
    """
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        n = logits.size(0)
        labels = 2 * torch.eye(n, device=logits.device) - 1       # +1 / -1
        # FIX: divide by n*n not n — we are summing over n² pairs
        loss = -F.logsigmoid(labels * logits).sum() / (n * n)
        return loss


# ─────────────────────────────────────────────
#  Full SigLIP 2 Model
# ─────────────────────────────────────────────

class SigLIP2Model(nn.Module):
    """
    Drop-in replacement for AutoModel.from_pretrained('google/siglip2-*').
    Exposes the same .loss attribute on forward output so train.py needs
    no changes.
    """

    def __init__(self, config):
        super().__init__()

        self.vision_encoder = VisionEncoder(
            img_size    = getattr(config, "img_size",    224),
            patch_size  = getattr(config, "patch_size",  16),
            in_channels = 3,
            embed_dim   = getattr(config, "vision_dim",  768),
            depth       = getattr(config, "vision_depth", 12),
            num_heads   = getattr(config, "vision_heads", 12),
            mlp_ratio   = 4.0,
            proj_dim    = getattr(config, "proj_dim",    512),
            dropout     = getattr(config, "dropout",     0.0),
        )

        self.text_encoder = TextEncoder(
            vocab_size   = getattr(config, "vocab_size",    32000),
            max_seq_len  = getattr(config, "max_seq_length", 64),
            embed_dim    = getattr(config, "text_dim",      512),
            depth        = getattr(config, "text_depth",    12),
            num_heads    = getattr(config, "text_heads",    8),
            mlp_ratio    = 4.0,
            proj_dim     = getattr(config, "proj_dim",      512),
            pad_token_id = getattr(config, "pad_token_id",  1),
            dropout      = getattr(config, "dropout",       0.0),
        )

        # FIX: logit_scale init — log(1/0.07) ≈ 2.66 matches CLIP/SigLIP paper init.
        # The original log(10) ≈ 2.3 (temperature=10) compresses logits early in training.
        # Also tighten the clamp: max temperature of 100 is fine, but add a minimum
        # so the scale can't collapse to ~0 and zero out all gradients.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.logit_bias  = nn.Parameter(torch.zeros(()))

        self.loss_fn = SigLoss()

    # ------------------------------------------------------------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(pixel_values)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    # ------------------------------------------------------------------
    def forward(self,
                pixel_values:   torch.Tensor = None,
                input_ids:      torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                return_loss:    bool         = True):

        img_emb  = self.encode_image(pixel_values)
        txt_emb  = self.encode_text(input_ids, attention_mask)

        # FIX: clamp temperature between 1 and 100 — prevents scale collapsing
        # to near-zero (which zeros all gradients) and exploding above 100.
        scale   = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        logits  = img_emb @ txt_emb.T * scale + self.logit_bias   # (B, B)

        loss = self.loss_fn(logits) if return_loss else None

        # Return a simple namespace so existing code can do outputs.loss
        return _Output(loss=loss, logits_per_image=logits,
                       image_embeds=img_emb, text_embeds=txt_emb)

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config, device="cuda"):
        model = cls(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model


class _Output:
    """Lightweight container — mimics HuggingFace model output."""
    __slots__ = ("loss", "logits_per_image", "image_embeds", "text_embeds")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)