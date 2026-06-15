"""
Conformer — Convolution-augmented Transformer
Artigo: Classificação de Contatos Acústicos Subaquáticos
        Utilizando Redes Neurais Profundas
Evento: SIGE 2026 — ITA
Autores: Leandro Kfuri de Oliveira, Rigel Procópio Fernandes
Referência: Gulati et al. (2020) — Interspeech
            DOI: 10.21437/Interspeech.2020-3015

Adaptação:
  Entrada 2D (espectrograma Mel) em vez de
  features 1D de reconhecimento de fala.
"""

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """
    Bloco Conformer — padrão Macaron-Net:
        FF(½) → MHSA → DWC 1D → FF(½) → LN

    O peso ½ nas camadas FF segue o padrão
    Macaron-Net: cada FF contribui com metade
    de seu valor para a conexão residual.
    """
    def __init__(self, dim, heads=4,
                 ff_mult=4, conv_kernel=31):
        super().__init__()

        # Feed-Forward 1 (peso 1/2)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim))

        # Multi-Head Self-Attention
        self.attn   = nn.MultiheadAttention(
            dim, heads,
            batch_first=True, dropout=0.1)
        self.norm_a = nn.LayerNorm(dim)

        # Depthwise Convolution 1D
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),              # gating
            nn.Conv1d(dim, dim,
                      conv_kernel,
                      padding=conv_kernel // 2,
                      groups=dim),      # depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1))
        self.norm_c = nn.LayerNorm(dim)

        # Feed-Forward 2 (peso 1/2)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # FF 1 — peso 1/2
        x = x + 0.5 * self.ff1(x)

        # Multi-Head Self-Attention
        a, _ = self.attn(x, x, x)
        x = x + self.norm_a(a)

        # Depthwise Conv 1D
        xc = self.conv(
            x.transpose(1, 2)).transpose(1, 2)
        x  = x + self.norm_c(xc)

        # FF 2 — peso 1/2
        x = x + 0.5 * self.ff2(x)

        return self.norm(x)


class Conformer(nn.Module):
    """
    Conformer para classificação acústica subaquática.

    Arquitetura:
        1. Patch Embedding 2D
           → Conv2D colapsa dimensão de frequência
           → gera sequência temporal de tokens
        2. Conformer Blocks (4 blocos)
        3. AdaptiveAvgPool + Classificador

    Entrada : (B, 1, N_MELS, T) — espectrograma Mel
    Saída   : (B, num_classes)
    Parâmetros: ~2M

    Diferença vs Transformer-DWC:
        O Conformer aplica DWC 1D (kernel=31)
        no domínio temporal APÓS a atenção,
        capturando dependências locais entre
        tokens adjacentes.

    Referência: Gulati et al. (2020) — Interspeech
    """
    def __init__(self, n_mels=128, num_classes=4,
                 dim=128, num_blocks=4, heads=4):
        super().__init__()

        # Patch Embedding — colapsa frequência
        self.patch = nn.Sequential(
            nn.Conv2d(1, dim,
                      kernel_size=(n_mels, 1)),
            nn.ReLU())

        # Conformer Blocks
        self.blocks = nn.Sequential(*[
            ConformerBlock(dim, heads)
            for _ in range(num_blocks)])

        # Classificador
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes))

    def forward(self, x):
        # [B, 1, N_MELS, T] → [B, dim, 1, T]
        x = self.patch(x)
        # [B, dim, T] → [B, T, dim]
        x = x.squeeze(2).transpose(1, 2)
        # Conformer Blocks
        x = self.blocks(x)
        # [B, dim, T] → [B, num_classes]
        return self.head(x.transpose(1, 2))


if __name__ == '__main__':
    model = Conformer(n_mels=128, num_classes=4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {total:,}")
    x = torch.randn(8, 1, 128, 94)
    print(f"Input:  {x.shape}")
    print(f"Output: {model(x).shape}")
