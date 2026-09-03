"""
Flower server with a robust aggregation strategy.

The week-6 run used the stock FedAvg strategy and logged two warnings:
`No fit_metrics_aggregation_fn provided` and the same for evaluate. Both are
fixed here, so the round summary now carries a real training loss and a real
mean AUC instead of `loss = 0.0`.

The strategy also implements the aggregator-side defences from the threat model:

  * update validation - reject non-finite or over-sized updates before they go
    anywhere near the aggregate (T4 malicious client update);
  * robust aggregation - coordinate median, trimmed mean or Krum instead of a
    plain weighted mean (T4 / T5 Sybil and free-rider sites);
  * a model registry with per-round versioning and rollback (T8 registry
    tampering), written to disk so a poisoned round can be reverted.

Run:
    python flower_server.py --rounds 10 --min-clients 2 --strategy trimmed_mean
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from fedgnn.aggregation import aggregate, validate_update

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
except ImportError:  # pragma: no cover
    fl = None


REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "registry")


def fit_metrics_aggregation(metrics: List[Tuple[int, Dict]]) -> Dict:
    total = sum(n for n, _ in metrics) or 1
    return {
        "train_loss": sum(n * m.get("train_loss", 0.0) for n, m in metrics) / total,
        "clients": len(metrics),
    }


def eval_metrics_aggregation(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics:
        return {}
    aucs = [m.get("auc", 0.5) for _, m in metrics]
    out = {"mean_auc": float(np.mean(aucs)), "min_auc": float(np.min(aucs))}
    p10 = [m["p_at_10"] for _, m in metrics if "p_at_10" in m]
    if p10:
        out["mean_p_at_10"] = float(np.mean(p10))
    return out


if fl is not None:

    class RobustFedStrategy(fl.server.strategy.FedAvg):
        """FedAvg with update validation, a robust aggregation rule and a registry."""

        def __init__(self, strategy: str = "fedavg", max_norm: float = 10.0,
                     beta: int = 1, f: int = 1, **kw):
            super().__init__(
                fit_metrics_aggregation_fn=fit_metrics_aggregation,
                evaluate_metrics_aggregation_fn=eval_metrics_aggregation,
                **kw,
            )
            self.rule = strategy
            self.max_norm = max_norm
            self.beta = beta
            self.f = f
            self.previous: Optional[List[np.ndarray]] = None
            os.makedirs(REGISTRY_DIR, exist_ok=True)

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}

            updates, sizes, rejected = [], [], []
            for client, fitres in results:
                w = parameters_to_ndarrays(fitres.parameters)
                delta = (
                    [a - b for a, b in zip(w, self.previous)] if self.previous else w
                )
                ok, why = validate_update(delta, max_norm=self.max_norm)
                if not ok:
                    rejected.append((client.cid, why))
                    continue
                updates.append(w)
                sizes.append(fitres.num_examples)

            for cid, why in rejected:
                print(f"  [round {server_round}] REJECTED update from {cid}: {why}")

            if not updates:
                print(f"  [round {server_round}] every update rejected - keeping previous model")
                if self.previous is None:
                    return None, {"rejected": len(rejected)}
                return ndarrays_to_parameters(self.previous), {"rejected": len(rejected)}

            kw = {}
            if self.rule == "trimmed_mean":
                kw = {"beta": self.beta}
            elif self.rule == "krum":
                kw = {"f": self.f, "multi": max(1, len(updates) - 2 * self.f - 2)}
            agg = aggregate(self.rule, updates, sizes, **kw)

            self.previous = agg
            self._register(server_round, agg, len(rejected))

            metrics = fit_metrics_aggregation(
                [(fr.num_examples, fr.metrics) for _, fr in results]
            )
            metrics["rejected"] = len(rejected)
            metrics["rule"] = self.rule
            print(
                f"  [round {server_round}] aggregated {len(updates)} updates with "
                f"{self.rule}, rejected {len(rejected)}, train_loss="
                f"{metrics['train_loss']:.4f}"
            )
            return ndarrays_to_parameters(agg), metrics

        # ------------------------------------------------------------ registry
        def _register(self, rnd: int, weights: List[np.ndarray], rejected: int) -> None:
            path = os.path.join(REGISTRY_DIR, f"round_{rnd:03d}.npz")
            np.savez_compressed(path, *weights)
            index = os.path.join(REGISTRY_DIR, "index.json")
            entries = []
            if os.path.exists(index):
                with open(index) as fh:
                    entries = json.load(fh)
            entries.append({
                "round": rnd,
                "file": os.path.basename(path),
                "rejected_updates": rejected,
                "l2_norm": float(np.linalg.norm(np.concatenate([w.ravel() for w in weights]))),
            })
            with open(index, "w") as fh:
                json.dump(entries, fh, indent=2)

        @staticmethod
        def rollback(rnd: int) -> List[np.ndarray]:
            """Restore a previous global model (T8 registry tampering / rollback)."""
            data = np.load(os.path.join(REGISTRY_DIR, f"round_{rnd:03d}.npz"))
            return [data[k] for k in data.files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--min-clients", type=int, default=2)
    ap.add_argument("--address", default="0.0.0.0:8080")
    ap.add_argument("--strategy", default="fedavg",
                    choices=["fedavg", "median", "trimmed_mean", "krum"])
    ap.add_argument("--max-norm", type=float, default=10.0)
    ap.add_argument("--beta", type=int, default=1, help="trimmed_mean: values dropped per end")
    ap.add_argument("--f", type=int, default=1, help="krum: assumed number of attackers")
    args = ap.parse_args()

    if fl is None:
        raise SystemExit("Flower is not installed: pip install flwr")

    strategy = RobustFedStrategy(
        strategy=args.strategy,
        max_norm=args.max_norm,
        beta=args.beta,
        f=args.f,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    print(f"server: {args.rounds} rounds, rule={args.strategy}, "
          f"min clients={args.min_clients}, registry={REGISTRY_DIR}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
