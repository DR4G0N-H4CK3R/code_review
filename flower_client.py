"""
Flower client - the real gRPC federation, not the in-process simulation.

This is the successor to the week-6 client. Two things changed:

  1. it trains on REAL graphs built from a flow source, not mock random tensors,
     so `Loss = 0.0 (placeholder)` on the week-6 output slide becomes a real
     learning curve;
  2. it returns evaluation metrics (loss and ROC-AUC on the site's held-out
     windows), so the server's `aggregate_evaluate` has something to aggregate
     instead of warning that no metrics function was provided.

The update sent to the server is clipped to a fixed L2 norm and optionally
noised, which is both the privacy step and what makes the server's robust
aggregation work.

Run two clients and a server in three terminals:

    python flower_server.py --rounds 10 --min-clients 2
    python flower_client.py --site 0 --server 127.0.0.1:8080
    python flower_client.py --site 1 --server 127.0.0.1:8080

Backends:
    --backend numpy   (default) no extra dependencies, runs on a Pi
    --backend torch   PyTorch + PyTorch Geometric
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from fedgnn import GCNAutoencoder, GCNConfig, GraphBuilder, evaluate, load_source
from fedgnn.aggregation import clip_and_noise
from fedgnn.federated import SITE_PROFILES
from fedgnn.graphs import apply_standardise, standardise

try:
    import flwr as fl
except ImportError:  # pragma: no cover
    fl = None


# --------------------------------------------------------------------------
def load_site(args):
    """Build this site's train / test graph snapshots."""
    if args.source == "synthetic":
        profile = SITE_PROFILES[args.site % len(SITE_PROFILES)]
        mix = {k: v for k, v in profile.items() if k != "name"}
        name = f"{profile['name']}-{args.site}"
        flows = load_source(
            "synthetic", site_id=args.site, seed=args.seed, n_devices=args.devices,
            role_mix=mix, duration_s=args.duration, attack_rate=0.03,
        )
    else:
        name = f"{args.source}-{args.site}"
        flows = load_source(args.source, path=args.data, limit=args.limit)

    snaps = GraphBuilder(window_s=args.window).build_all(flows)
    if len(snaps) < 4:
        raise SystemExit("not enough windows - lengthen the capture or shrink --window")
    cut = max(2, int(len(snaps) * 0.5))
    train, test = snaps[:cut], snaps[cut:]
    mu, sd = standardise(train)
    apply_standardise(train, mu, sd)
    apply_standardise(test, mu, sd)
    return name, train, test


# --------------------------------------------------------------------------
class NumpyGCNClient(fl.client.NumPyClient if fl else object):
    def __init__(self, name, train, test, args):
        self.name = name
        self.train_snaps = train
        self.test_snaps = test
        self.args = args
        self.model = GCNAutoencoder(GCNConfig(seed=args.seed))
        self.rng = np.random.default_rng(args.seed)
        print(f"[{name}] {len(train)} train windows, {len(test)} test windows, "
              f"{self.model.n_params()} parameters")

    # -------------------------------------------------------------- Flower API
    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        before = [w.copy() for w in parameters]

        hist = self.model.fit(self.train_snaps, epochs=self.args.local_epochs)
        after = self.model.get_weights()

        delta = clip_and_noise(
            [a - b for a, b in zip(after, before)],
            clip_norm=self.args.clip_norm,
            noise_std=self.args.noise_std,
            rng=self.rng,
        )
        sent = [b + d for b, d in zip(before, delta)]
        loss = float(hist[-1]) if hist else 0.0
        print(f"[{self.name}] local training done, loss={loss:.4f}")
        return sent, len(self.train_snaps), {"train_loss": loss}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        m = evaluate(self.model, self.test_snaps, top_k=10)
        loss = self.model.evaluate_loss(self.test_snaps)
        print(f"[{self.name}] eval loss={loss:.4f} auc={m['auc']:.3f} p@10={m['p@k']:.2f}")
        return loss, len(self.test_snaps), {
            "auc": float(m["auc"]),
            "p_at_10": float(m["p@k"]),
            "malicious_edges": int(m["n_malicious"]),
        }


class TorchGCNClient(fl.client.NumPyClient if fl else object):
    def __init__(self, name, train, test, args):
        from fedgnn.model_torch import (
            TORCH_AVAILABLE,
            LightweightIoTGCNAutoencoder,
            snapshot_to_pyg,
            train_epochs,
        )

        if not TORCH_AVAILABLE:
            raise SystemExit("--backend torch needs: pip install torch torch-geometric")
        self._train_epochs = train_epochs
        self.name = name
        self.args = args
        self.train_graphs = [snapshot_to_pyg(s) for s in train]
        self.test_snaps = test
        self.model = LightweightIoTGCNAutoencoder()
        print(f"[{name}] torch backend, {len(train)} train windows")

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self._train_epochs(self.model, self.train_graphs, epochs=self.args.local_epochs)
        print(f"[{self.name}] local training done, loss={loss:.4f}")
        return self.model.get_weights(), len(self.train_graphs), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        import torch

        from fedgnn.model_torch import snapshot_to_pyg
        from fedgnn.scoring import DEFAULT_WEIGHTS, _rank_normalise, roc_auc

        self.model.set_weights(parameters)
        ys, ss = [], []
        for snap in self.test_snaps:
            if snap.n_edges == 0:
                continue
            g = snapshot_to_pyg(snap)
            comps = self.model.edge_scores(g.x, g.edge_index, g.edge_attr)
            w1, w2, w3, w4 = DEFAULT_WEIGHTS
            nov = snap.edge_attr[:, 5]
            score = (
                w1 * _rank_normalise(comps[:, 0])
                + w2 * _rank_normalise(comps[:, 1])
                + w3 * _rank_normalise(comps[:, 2])
                + w4 * nov
            )
            ss.append(score)
            ys.append(snap.edge_label)
        auc = roc_auc(np.concatenate(ys), np.concatenate(ss)) if ys else 0.5
        print(f"[{self.name}] eval auc={auc:.3f}")
        return float(1.0 - auc), len(self.test_snaps), {"auc": float(auc)}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, default=0)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=2400.0)
    ap.add_argument("--devices", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if fl is None:
        raise SystemExit("Flower is not installed: pip install flwr")

    name, train, test = load_site(args)
    cls = NumpyGCNClient if args.backend == "numpy" else TorchGCNClient
    client = cls(name, train, test, args)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
