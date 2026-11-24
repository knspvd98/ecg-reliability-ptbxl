from __future__ import annotations

import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1) if in_channels > 1 else nn.Identity()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(bottleneck if in_channels > 1 else in_channels, branch_channels, kernel_size=k, padding=k // 2)
                for k in (9, 19, 39)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm1d(branch_channels * 4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.bottleneck(x)
        outputs = [branch(transformed) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return self.activation(self.bn(torch.cat(outputs, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            InceptionModule(in_channels, hidden_dim),
            InceptionModule(hidden_dim, hidden_dim),
            InceptionModule(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
