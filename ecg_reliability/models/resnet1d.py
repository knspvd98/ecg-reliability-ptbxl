from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.layers = nn.Sequential(
            ResidualBlock1D(hidden_dim // 2, hidden_dim // 2, dropout=dropout),
            ResidualBlock1D(hidden_dim // 2, hidden_dim, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim, hidden_dim, dropout=dropout),
            ResidualBlock1D(hidden_dim, hidden_dim * 2, stride=2, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return self.head(x)
