"""
Transformer-DWC: Spatial-Temporal Fusion Neural Network
Baseado em: Wang et al. (2024) - Frontiers in Marine Science
Referência [3] do projeto de pesquisa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DROPOUT


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise Separable Convolution (DWC).
    Mais eficiente que Conv2D padrão — separa convolução espacial e de canal.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class PatchEmbedding(nn.Module):
    """
    Divide o espectrograma em patches e projeta para dimensão do Transformer.
    Análogo ao ViT (Vision Transformer).
    """
    def __init__(self, in_channels=1, embed_dim=128, patch_size=16):
        super().__init__()
        self.proj = nn.Sequential(
            DepthwiseConvBlock(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            DepthwiseConvBlock(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, embed_dim) — sequência de tokens
        return x


class TransformerBlock(nn.Module):
    """
    Bloco Transformer com Multi-Head Self-Attention + Feed-Forward.
    Captura dependências temporais e espectrais de longo alcance.
    """
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention com conexão residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward com conexão residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerDWC(nn.Module):
    """
    Transformer-DWC para classificação acústica subaquática.

    Arquitetura:
        1. Patch Embedding com DWC (extração local de features)
        2. Transformer Blocks (captura dependências de longo alcance)
        3. Classificador Global

    Entrada : espectrograma Mel (1, N_MELS, T)
    Saída   : logits (NUM_CLASSES,)

    Referência: Wang et al. (2024) - Frontiers in Marine Science, v.11, p.1331717
    """
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=128,
                 num_heads=4, num_layers=4, ff_dim=256, dropout=DROPOUT):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels=1, embed_dim=embed_dim)

        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)     # (B, tokens, embed_dim)
        x = self.transformer(x)     # (B, tokens, embed_dim)
        x = x.mean(dim=1)           # Global Average Pooling sobre tokens
        x = self.classifier(x)      # (B, num_classes)
        return x
