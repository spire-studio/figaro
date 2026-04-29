from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_dim(input_shape: tuple[int, ...] | list[int] | int | None, input_channels: int = 1) -> int:
    if isinstance(input_shape, int):
        return input_shape
    if input_shape:
        total = 1
        for dim in input_shape:
            total *= int(dim)
        return total
    return int(input_channels) * 28 * 28


class MclrLogistic(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        input_shape: tuple[int, ...] | list[int] | int | None = None,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim or _flatten_dim(input_shape, input_channels))
        self.linear = nn.Linear(self.input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1).float())


class DNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        input_shape: tuple[int, ...] | list[int] | int | None = None,
        input_dim: int | None = None,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.input_dim = int(input_dim or _flatten_dim(input_shape, input_channels))
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FedAvgCNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TextLogistic(MclrLogistic):
    pass


class TextDNN(DNN):
    pass


class TextCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        input_dim: int = 5000,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 1, -1).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class CharLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 128,
        vocab_size: int = 128,
        embed_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_ids = x.long().clamp(min=0, max=self.vocab_size - 1)
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])
