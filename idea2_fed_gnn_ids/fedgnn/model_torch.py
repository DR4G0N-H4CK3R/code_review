"""
PyTorch Geometric version of the same model, for the stack already in the
Design and Solution doc (PyTorch + PyG + Flower).

It is the direct successor to the week-6 `LightweightIoT_GCN`: same 8 -> 16 -> 2
shape and the same Gaussian feature-noise hardening, but it is now an
AUTOENCODER with three decoders instead of a 2-class classifier, because the
IoT-23 / CICIoT2023 labels are per flow and give no topology-level ground truth
to train a supervised edge classifier against.

`model_numpy.py` is the reference implementation and the two are kept in step.
They are not weight-compatible (different parameter layout), so pick one backend
per federation and stay with it.

Install:
    pip install torch torch-geometric flwr
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    TORCH_AVAILABLE = False
    torch = None
    nn = object


if TORCH_AVAILABLE:

    class LightweightIoTGCNAutoencoder(nn.Module):
        """2-layer GCN encoder + node / structural / edge decoders."""

        def __init__(
            self,
            in_dim: int = 8,
            hidden_dim: int = 16,
            latent_dim: int = 2,
            edge_dim: int = 8,
            noise_std: float = 0.05,
        ) -> None:
            super().__init__()
            self.enc1 = GCNConv(in_dim, hidden_dim)
            self.enc2 = GCNConv(hidden_dim, latent_dim)
            self.node_dec = nn.Linear(latent_dim, in_dim)
            self.edge_dec = nn.Linear(2 * latent_dim, edge_dim)
            self.noise_std = noise_std

        def encode(self, x, edge_index):
            if self.training and self.noise_std > 0:
                # adversarial hardening against sensor mimicry, as in week 6
                x = x + torch.randn_like(x) * self.noise_std
            h = F.relu(self.enc1(x, edge_index))
            return self.enc2(h, edge_index)

        def forward(self, x, edge_index):
            z = self.encode(x, edge_index)
            x_hat = self.node_dec(z)
            e_hat = self.edge_dec(torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1))
            return z, x_hat, e_hat

        # ------------------------------------------------------------- losses
        @staticmethod
        def structural_loss(z, edge_index, num_nodes):
            pos = (z[edge_index[0]] * z[edge_index[1]]).sum(-1)
            neg_src = torch.randint(0, num_nodes, (edge_index.size(1),), device=z.device)
            neg_dst = torch.randint(0, num_nodes, (edge_index.size(1),), device=z.device)
            neg = (z[neg_src] * z[neg_dst]).sum(-1)
            logits = torch.cat([pos, neg])
            target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
            return F.binary_cross_entropy_with_logits(logits, target)

        def loss(self, x, edge_index, edge_attr, lf=1.0, ls=1.0, le=1.0):
            z, x_hat, e_hat = self(x, edge_index)
            l = lf * F.mse_loss(x_hat, x)
            l = l + ls * self.structural_loss(z, edge_index, x.size(0))
            if edge_attr is not None and edge_index.size(1) > 0:
                l = l + le * F.mse_loss(e_hat, edge_attr)
            return l

        # ------------------------------------------------------- Flower hooks
        def get_weights(self) -> List[np.ndarray]:
            return [v.detach().cpu().numpy() for v in self.state_dict().values()]

        def set_weights(self, weights: Sequence[np.ndarray]) -> None:
            sd = self.state_dict()
            new = {k: torch.tensor(np.asarray(w)) for k, w in zip(sd.keys(), weights)}
            self.load_state_dict(new, strict=True)

        @torch.no_grad()
        def edge_scores(self, x, edge_index, edge_attr) -> np.ndarray:
            """Same four components as scoring.py, for parity with the NumPy path."""
            self.eval()
            z, x_hat, e_hat = self(x, edge_index)
            struct = 1.0 - torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(-1))
            node_err = ((x_hat - x) ** 2).mean(1)
            feat = 0.5 * (node_err[edge_index[0]] + node_err[edge_index[1]])
            edge_err = ((e_hat - edge_attr) ** 2).mean(1)
            return (
                torch.stack([struct, feat, edge_err], dim=1).cpu().numpy()
            )


def snapshot_to_pyg(snap):
    """GraphSnapshot -> torch_geometric.data.Data."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch / PyTorch Geometric are not installed")
    from torch_geometric.data import Data

    return Data(
        x=torch.tensor(snap.X, dtype=torch.float32),
        edge_index=torch.tensor(snap.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(snap.edge_attr, dtype=torch.float32),
        y=torch.tensor(snap.edge_label, dtype=torch.long),
    )


def train_epochs(model, graphs, epochs: int = 3, lr: float = 0.01) -> float:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch / PyTorch Geometric are not installed")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    for _ in range(epochs):
        total, n = 0.0, 0
        for g in graphs:
            if g.edge_index.size(1) == 0:
                continue
            opt.zero_grad()
            loss = model.loss(g.x, g.edge_index, g.edge_attr)
            loss.backward()
            opt.step()
            total += float(loss)
            n += 1
        last = total / max(n, 1)
    return last
