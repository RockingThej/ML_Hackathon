"""
src/models/mlp.py  -  PyTorch candidate scorer (Part D)
=======================================================
CandidateMLP : scores one (GPS point, candidate arc) row -> one logit.
MLPScorer    : sklearn-style wrapper (predict_proba / decision_function) so the
               API can load it with joblib exactly like C's LightGBM / RF models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CandidateMLP(nn.Module):
    def __init__(self, n_features: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., F) -> (...,) logits. Works for flat (N,F) or grouped (P,K,F) input.
        return self.net(x).squeeze(-1)


class MLPScorer:
    """Picklable wrapper: scaler + network on CPU, sklearn-like interface.

    loss='listwise' : network trained with softmax over each point's K candidates.
                      predict_proba returns sigmoid(logit) - same ranking as the
                      softmax, so the API's argmax per point is unchanged. Use
                      point_proba() when you need a distribution per point (HMM).
    loss='bce'      : network trained per row; sigmoid(logit) is P(true arc).
    """

    def __init__(self, model: CandidateMLP, scaler, features, config: dict):
        self.model = model.cpu().eval()
        self.scaler = scaler
        self.features = list(features)
        self.config = dict(config)
        self.classes_ = np.array([0, 1])

    @torch.no_grad()
    def decision_function(self, X) -> np.ndarray:
        X = self.scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        out = []
        for i in range(0, len(X), 200_000):
            out.append(self.model(torch.from_numpy(X[i:i + 200_000])).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X).astype(np.float64)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def point_proba(self, X, k: int = 15) -> np.ndarray:
        """Softmax over consecutive groups of k rows (build_features output order)."""
        z = self.decision_function(X).astype(np.float64)
        if len(z) % k:
            raise ValueError(f"len(X)={len(z)} is not a multiple of k={k}")
        z = z.reshape(-1, k)
        e = np.exp(z - z.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).ravel()
