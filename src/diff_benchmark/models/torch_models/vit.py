import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        x = x + x_res

        # MLP
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + x_res
        return x

# -----------------------------
# 3D Patch Embedding
# -----------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(6,16,16), embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_dim)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        s, h, w = self.patch_size
        assert D % s == 0 and H % h == 0 and W % w == 0, "Input dimensions must be divisible by patch size"

        # reshape into non-overlapping 3D patches
        x = rearrange(x, 'b c (d s) (h hh) (w ww) -> b (d h w) (s hh ww c)',
                      s=s, hh=h, ww=w)
        x = self.proj(x)
        return x  # shape: (B, N_patches, embed_dim)

# -----------------------------
# 3D Vision Transformer with optional MAE pretraining
# -----------------------------
class ViT3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_size=(182, 224, 224),
        patch_size=(6, 16, 16),
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        dropout=0.,
        mae_pretrain=False,
        mask_ratio=0.75,
        decoder_dim=192,
        decoder_depth=4
    ):
        super().__init__()

        self.mae_pretrain = mae_pretrain
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        s, h, w = patch_size
        D, H, W = img_size
        self.num_patches = (D // s) * (H // h) * (W // w)

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # MAE decoder (optional, only for pretraining)
        if mae_pretrain:
            self.decoder_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads // 2, mlp_ratio, dropout)
                                                 for _ in range(decoder_depth)])
            self.decoder_norm = nn.LayerNorm(embed_dim)
            self.reconstruction_head = nn.Linear(embed_dim, in_channels * s * h * w)

    # -----------------------------
    # Masking function for MAE
    # -----------------------------
    def random_mask(self, x):
        B, N, _ = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x_masked, ids_restore

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x):
        """
        x: (B, C, D, H, W)
        Returns:
            - if mae_pretrain=False: class token embedding (B, embed_dim)
            - if mae_pretrain=True: reconstructed patches (B, N, patch_flat)
        """
        x = self.patch_embed(x)  # (B, N, embed_dim)
        B, N, _ = x.shape

        # MAE pretraining branch
        if self.mae_pretrain:
            x_masked, ids_restore = self.random_mask(x)
            # Encode visible patches
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_masked = torch.cat((cls_tokens, x_masked), dim=1)
            x_masked = x_masked + self.pos_embed[:, :x_masked.size(1), :]
            for blk in self.blocks:
                x_masked = blk(x_masked)
            x_encoded = x_masked[:, 1:, :]  # remove class token

            # Prepare decoder input
            x_full = torch.zeros_like(x)
            x_full[:, :x_encoded.size(1), :] = x_encoded
            x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
            x_full = torch.cat((cls_tokens, x_full), dim=1)
            x_full = x_full + self.pos_embed

            for blk in self.decoder_blocks:
                x_full = blk(x_full)
            x_full = self.decoder_norm(x_full)
            rec_patches = self.reconstruction_head(x_full[:, 1:, :])
            return rec_patches  # (B, N, patch_flat)

        # Standard forward (feature extraction)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # class token embedding
