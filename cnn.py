import torch
import torch.nn as nn
from config import NUM_CLASSES, DROPOUT


class ConvBlock(nn.Module):
    """Bloco: Conv2D → BatchNorm → ReLU → MaxPool."""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    CNN baseline para classificação acústica.
    Entrada : espectrograma Mel (1, N_MELS, T)
    Saída   : logits (NUM_CLASSES,)
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),   # → (32, N_MELS/2, T/2)
            ConvBlock(32,  64),   # → (64, N_MELS/4, T/4)
            ConvBlock(64, 128),   # → (128, N_MELS/8, T/8)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
