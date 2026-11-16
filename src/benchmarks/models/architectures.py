"""Multiclass tabular-feature architectures for the E1 standalone benchmark.

For convolutional/recurrent/Transformer models, the saved feature order is
interpreted as a sequence-like axis solely for architectural comparison. These
models do not receive temporal records or learn real time dependencies.
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.unsqueeze(1)))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x)))))
        return self.activation(x + residual)


class ResNet1D(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.blocks = nn.Sequential(ResidualBlock1D(32), ResidualBlock1D(32, dilation=2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x.unsqueeze(1)))
        return self.classifier(self.pool(h).flatten(1))


class RecurrentClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, kind: str, bidirectional: bool = False, cnn_frontend: bool = False):
        super().__init__()
        self.cnn_frontend = cnn_frontend
        if cnn_frontend:
            self.frontend = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            )
            recurrent_input = 32
        else:
            self.frontend = None
            recurrent_input = 1
        rnn = nn.GRU if kind == "GRU" else nn.LSTM
        self.recurrent = rnn(recurrent_input, 48, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Linear(48 * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cnn_frontend:
            sequence = self.frontend(x.unsqueeze(1)).transpose(1, 2)
        else:
            sequence = x.unsqueeze(-1)
        outputs, _ = self.recurrent(sequence)
        return self.classifier(outputs[:, -1, :])


class TCNBlock(nn.Module):
    """Residual dilated same-length block on the ordered feature axis."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.ReLU(),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TCN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_projection = nn.Conv1d(1, 32, kernel_size=1)
        self.blocks = nn.Sequential(TCNBlock(32, 1), TCNBlock(32, 2), TCNBlock(32, 4))
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.input_projection(x.unsqueeze(1))).mean(dim=-1)
        return self.classifier(h)


class TinyTransformer(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_projection = nn.Linear(1, 32)
        self.position_embedding = nn.Parameter(torch.zeros(1, input_dim, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x.unsqueeze(-1)) + self.position_embedding[:, :x.shape[1], :]
        return self.classifier(self.encoder(h).mean(dim=1))


MODEL_REGISTRY = {
    "mlp": ("MLP/DNN", MLP),
    "cnn1d": ("1D-CNN", CNN1D),
    "resnet1d": ("1D-ResNet", ResNet1D),
    "gru": ("GRU", lambda d, k: RecurrentClassifier(d, k, "GRU")),
    "lstm": ("LSTM", lambda d, k: RecurrentClassifier(d, k, "LSTM")),
    "bilstm": ("BiLSTM", lambda d, k: RecurrentClassifier(d, k, "LSTM", bidirectional=True)),
    "cnn_gru": ("CNN-GRU", lambda d, k: RecurrentClassifier(d, k, "GRU", cnn_frontend=True)),
    "cnn_lstm": ("CNN-LSTM", lambda d, k: RecurrentClassifier(d, k, "LSTM", cnn_frontend=True)),
    "tcn": ("TCN", TCN),
    "tiny_transformer": ("Tiny Transformer", TinyTransformer),
}


def build_model(model_key: str, input_dim: int, n_classes: int) -> nn.Module:
    try:
        _, factory = MODEL_REGISTRY[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {model_key}") from exc
    return factory(input_dim, n_classes)
