"""
CRNN: Convolutional Recurrent Neural Network
para Classificação Acústica Subaquática.

Arquitetura:
    1. CNN — extrai features espectrais locais do espectrograma Mel
    2. LSTM — captura dependências temporais da sequência de features
    3. Classificador — camadas densas para classificação final

Referências:
    - Sailor et al. (2017) — CRNN para classificação de áudio
    - LECUN; BENGIO; HINTON (2015) — Deep Learning
"""

import torch
import torch.nn as nn
from config import NUM_CLASSES, DROPOUT


class ConvBlock(nn.Module):
    """Bloco convolucional com BatchNorm e ReLU."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CRNN(nn.Module):
    """
    CRNN para classificação de espectrogramas Mel subaquáticos.

    Entrada : (B, 1, N_MELS, T) = (B, 1, 128, 309)
    Saída   : (B, NUM_CLASSES)

    Pipeline:
        CNN  → (B, 256, 1, T//8) → squeeze → (B, T//8, 256)
        LSTM → (B, T//8, 256)    → último hidden → (B, 256)
        FC   → (B, NUM_CLASSES)
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()

        # ── Extrator de features CNN ──────────────────────────
        self.cnn = nn.Sequential(
            # Bloco 1: (1, 128, T) → (64, 64, T)
            ConvBlock(1, 64),
            nn.MaxPool2d(kernel_size=(2, 1)),   # reduz freq, mantém tempo

            # Bloco 2: (64, 64, T) → (128, 32, T)
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Bloco 3: (128, 32, T) → (256, 16, T)
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Bloco 4: (256, 16, T) → (256, 8, T)
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Bloco 5: (256, 8, T) → (256, 4, T)
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Bloco 6: (256, 4, T) → (256, 2, T)
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Bloco 7: (256, 2, T) → (256, 1, T)
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, None)),    # → (256, 1, T)
        )

        # ── LSTM — captura dependências temporais ─────────────
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # ── Classificador ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.cnn(x)             # (B, 256, 1, T)
        x = x.squeeze(2)            # (B, 256, T)
        x = x.permute(0, 2, 1)     # (B, T, 256) — formato para LSTM

        x, _ = self.lstm(x)         # (B, T, 256)
        x = x[:, -1, :]             # último timestep → (B, 256)

        return self.classifier(x)   # (B, NUM_CLASSES)
