"""CRNN model definition (optional training)."""
from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, img_h: int, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(
            input_size=128 * (img_h // 4),
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        batch, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(batch, width, channels * height)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits
