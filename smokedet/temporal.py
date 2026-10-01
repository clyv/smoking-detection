"""Stage 3 learned model: a small dilated 1D CNN over a few seconds of one person's features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class TemporalCNN(nn.Module):
    """Input (batch, time, features) -> logit that the window shows smoking."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()

        def block(c_in: int, dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(block(n_features, 1), block(hidden, 2), block(hidden, 4), nn.Dropout(dropout))
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h).squeeze(-1)


def make_windows(feats: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Slice a (T, F) sequence into (N, window, F) windows; short tracks are edge-padded."""
    T = len(feats)
    if T == 0:
        return np.zeros((0, window, feats.shape[1]), dtype=np.float32)
    if T < window:
        pad = np.repeat(feats[:1], window - T, axis=0)
        return np.concatenate([pad, feats], 0)[None].astype(np.float32)
    starts = list(range(0, T - window + 1, hop))
    if starts[-1] != T - window:
        starts.append(T - window)
    return np.stack([feats[s : s + window] for s in starts]).astype(np.float32)


class TemporalScorer:
    """Loads a checkpoint written by scripts/09_train_temporal.py and scores windows."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.feature_names: list[str] = ck["feature_names"]
        self.window: int = ck["window"]
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.model = TemporalCNN(len(self.feature_names), **ck.get("model_kwargs", {}))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def score(self, window_feats: np.ndarray) -> float:
        w = window_feats[-self.window :]
        if len(w) < self.window:
            w = np.concatenate([np.repeat(w[:1], self.window - len(w), 0), w], 0)
        x = np.clip((w - self.mean) / self.std, -10, 10)
        x = torch.from_numpy(x[None].astype(np.float32)).to(self.device)
        return float(torch.sigmoid(self.model(x))[0])
