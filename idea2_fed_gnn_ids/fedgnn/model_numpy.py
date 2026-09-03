"""
Lightweight 2-layer GCN autoencoder - reference implementation in NumPy.

Same shape as the PyTorch Geometric model in `model_torch.py` (8 -> 16 -> 2), so
the two are interchangeable. The NumPy version exists for two reasons:

  1. it runs anywhere, including a Raspberry Pi with no wheels available, which
     is the deployment target;
  2. every gradient is written out, so the maths in the report is auditable
     rather than hidden behind autograd.

Architecture
    H1  = ReLU( A_hat X W1 + b1 )                encoder layer 1
    Z   =       A_hat H1 W2 + b2                 latent node embeddings, dim 2
    X^  = Z W3 + b3                              node feature decoder
    A^  = sigmoid( Z Z^T )                       structural decoder
    E^  = [z_u || z_v] W4 + b4                   edge feature decoder

Loss
    L = lf * MSE(X, X^)
      + ls * BCE(A^, A) on positive edges + sampled negatives
      + le * MSE(E, E^) over observed edges

The edge decoder is what makes the relationship - not just the device - the unit
of detection: an edge whose (bytes, ports, rarity, novelty) profile cannot be
predicted from the two endpoint embeddings is exactly the "camera suddenly
reaches the NAS" case from the problem statement.

Adversarial hardening: Gaussian noise is added to X during training only, which
is the mimicry defence carried over from the week-6 model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class GCNConfig:
    in_dim: int = 8
    hidden_dim: int = 16
    latent_dim: int = 2
    edge_dim: int = 8
    lr: float = 0.01
    lambda_feat: float = 1.0
    lambda_struct: float = 1.0
    lambda_edge: float = 1.0
    noise_std: float = 0.05        # adversarial hardening
    neg_ratio: float = 1.0
    seed: int = 0


class GCNAutoencoder:
    KEYS = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")

    # ----------------------------------------------------------------- setup
    def __init__(self, cfg: Optional[GCNConfig] = None) -> None:
        self.cfg = cfg or GCNConfig()
        rng = np.random.default_rng(self.cfg.seed)
        c = self.cfg

        def glorot(a, b):
            lim = math.sqrt(6.0 / (a + b))
            return rng.uniform(-lim, lim, size=(a, b)).astype(np.float32)

        self.params: Dict[str, np.ndarray] = {
            "W1": glorot(c.in_dim, c.hidden_dim),
            "b1": np.zeros(c.hidden_dim, dtype=np.float32),
            "W2": glorot(c.hidden_dim, c.latent_dim),
            "b2": np.zeros(c.latent_dim, dtype=np.float32),
            "W3": glorot(c.latent_dim, c.in_dim),
            "b3": np.zeros(c.in_dim, dtype=np.float32),
            "W4": glorot(2 * c.latent_dim, c.edge_dim),
            "b4": np.zeros(c.edge_dim, dtype=np.float32),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0
        self.rng = rng

    # ------------------------------------------------------- parameter I/O
    def get_weights(self) -> List[np.ndarray]:
        """Flower-compatible ordering."""
        return [self.params[k].copy() for k in self.KEYS]

    def set_weights(self, ws: Sequence[np.ndarray]) -> None:
        for k, w in zip(self.KEYS, ws):
            self.params[k] = np.asarray(w, dtype=np.float32).copy()

    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # -------------------------------------------------------------- forward
    def forward(self, X: np.ndarray, A: np.ndarray, training: bool = False) -> Dict[str, np.ndarray]:
        p = self.params
        Xin = X
        if training and self.cfg.noise_std > 0:
            Xin = X + self.rng.normal(0.0, self.cfg.noise_std, size=X.shape).astype(np.float32)
        AX = A @ Xin
        H1pre = AX @ p["W1"] + p["b1"]
        H1 = np.maximum(H1pre, 0.0)
        AH1 = A @ H1
        Z = AH1 @ p["W2"] + p["b2"]
        Xhat = Z @ p["W3"] + p["b3"]
        return {"Xin": Xin, "AX": AX, "H1pre": H1pre, "H1": H1, "AH1": AH1, "Z": Z, "Xhat": Xhat}

    def decode_edges(self, Z: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Predict edge features from the two endpoint embeddings."""
        if edge_index.shape[1] == 0:
            return np.zeros((0, self.cfg.edge_dim), dtype=np.float32)
        cat = np.concatenate([Z[edge_index[0]], Z[edge_index[1]]], axis=1)
        return cat @ self.params["W4"] + self.params["b4"]

    # ---------------------------------------------------------------- train
    def _sample_negatives(self, edge_index: np.ndarray, n: int, k: int) -> np.ndarray:
        pos = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        out = []
        tries = 0
        while len(out) < k and tries < 20 * k + 50:
            tries += 1
            s = int(self.rng.integers(0, n))
            d = int(self.rng.integers(0, n))
            if s == d or (s, d) in pos:
                continue
            out.append((s, d))
        if not out:
            return np.zeros((2, 0), dtype=np.int64)
        return np.asarray(out, dtype=np.int64).T

    def train_step(
        self,
        X: np.ndarray,
        A: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray] = None,
    ) -> float:
        c = self.cfg
        p = self.params
        N, F = X.shape
        f = self.forward(X, A, training=True)
        Z, Xhat = f["Z"], f["Xhat"]

        # ---------------- feature reconstruction
        diff = Xhat - X
        loss_feat = float(np.mean(diff**2))
        dXhat = (2.0 / (N * F)) * diff * c.lambda_feat

        gW3 = Z.T @ dXhat
        gb3 = dXhat.sum(0)
        dZ = dXhat @ p["W3"].T

        # ---------------- structural reconstruction
        neg = self._sample_negatives(edge_index, N, int(edge_index.shape[1] * c.neg_ratio))
        src = np.concatenate([edge_index[0], neg[0]]) if neg.size else edge_index[0]
        dst = np.concatenate([edge_index[1], neg[1]]) if neg.size else edge_index[1]
        y = np.concatenate(
            [np.ones(edge_index.shape[1], dtype=np.float32),
             np.zeros(neg.shape[1] if neg.size else 0, dtype=np.float32)]
        )
        loss_struct = 0.0
        if len(y):
            logits = np.einsum("ij,ij->i", Z[src], Z[dst])
            prob = sigmoid(logits)
            eps = 1e-7
            loss_struct = float(
                -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
            )
            dlogit = (prob - y) / len(y) * c.lambda_struct
            # d/dZ of  z_s . z_d  contributes to both endpoints
            np.add.at(dZ, src, dlogit[:, None] * Z[dst])
            np.add.at(dZ, dst, dlogit[:, None] * Z[src])

        # ---------------- edge feature reconstruction
        loss_edge = 0.0
        if edge_attr is not None and edge_index.shape[1] > 0 and c.lambda_edge > 0:
            es, ed = edge_index[0], edge_index[1]
            cat = np.concatenate([Z[es], Z[ed]], axis=1)
            Ehat = cat @ p["W4"] + p["b4"]
            ediff = Ehat - edge_attr
            E, Fe = edge_attr.shape
            loss_edge = float(np.mean(ediff**2))
            dE = (2.0 / (E * Fe)) * ediff * c.lambda_edge
            gW4 = cat.T @ dE
            gb4 = dE.sum(0)
            dcat = dE @ p["W4"].T
            L = c.latent_dim
            np.add.at(dZ, es, dcat[:, :L])
            np.add.at(dZ, ed, dcat[:, L:])
        else:
            gW4 = np.zeros_like(p["W4"])
            gb4 = np.zeros_like(p["b4"])

        # ---------------- backprop through the encoder
        gW2 = f["AH1"].T @ dZ
        gb2 = dZ.sum(0)
        dH1 = (A.T @ dZ) @ p["W2"].T
        dH1pre = dH1 * (f["H1pre"] > 0)
        gW1 = f["AX"].T @ dH1pre
        gb1 = dH1pre.sum(0)

        self._adam({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                    "W3": gW3, "b3": gb3, "W4": gW4, "b4": gb4})
        return (
            c.lambda_feat * loss_feat
            + c.lambda_struct * loss_struct
            + c.lambda_edge * loss_edge
        )

    def _adam(self, grads: Dict[str, np.ndarray], b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mh = self._m[k] / (1 - b1**self._t)
            vh = self._v[k] / (1 - b2**self._t)
            self.params[k] -= self.cfg.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, snapshots: Sequence, epochs: int = 5) -> List[float]:
        """One pass over a site's training snapshots, `epochs` times."""
        history = []
        for _ in range(epochs):
            losses = [
                self.train_step(s.X, s.A, s.edge_index, s.edge_attr)
                for s in snapshots
                if s.n_edges > 0
            ]
            history.append(float(np.mean(losses)) if losses else 0.0)
        return history

    def evaluate_loss(self, snapshots: Sequence) -> float:
        losses = []
        for s in snapshots:
            if s.n_edges == 0:
                continue
            f = self.forward(s.X, s.A, training=False)
            losses.append(float(np.mean((f["Xhat"] - s.X) ** 2)))
        return float(np.mean(losses)) if losses else 0.0
