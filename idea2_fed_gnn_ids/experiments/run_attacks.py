"""
Attacks on the federated pipeline itself.

The point on the novelty slide is that the FL pipeline is part of the attack
surface, not just the detector. This script measures three of the threats from
the threat-model figure:

  T4  Malicious client update  - a compromised site sends a scaled or sign-flipped
                                 update to destroy the global model.
  T5  Sybil / free-rider sites - more than one such client.
  T2  Structural graph poisoning - a client injects plausible extra edges into its
                                 own training graphs so the global model learns
                                 that camera -> NAS relationships are normal.

T4/T5 are defended in the aggregator (norm validation + robust aggregation).
T2 is NOT: the update it produces is perfectly well-formed, so no amount of
robust averaging sees it. That is reported rather than papered over.

Run:  python -m experiments.run_attacks
Writes results/attack_*.csv and results/attack_*.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedgnn import GCNAutoencoder, build_sites, evaluate, run_federated

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

STRATEGIES = ["fedavg", "median", "trimmed_mean", "krum"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_case(
    n_sites: int,
    n_malicious: int,
    kind: str,
    strategy: str,
    rounds: int,
    validate: bool,
    seed: int,
    duration: float,
    window: float,
) -> float:
    sites = build_sites(n_sites=n_sites, seed=seed, duration_s=duration, window_s=window)
    for s in sites[:n_malicious]:
        s.malicious = kind
    kw = {}
    if strategy == "trimmed_mean":
        kw = {"beta": max(1, n_malicious)}
    elif strategy == "krum":
        kw = {"f": max(1, n_malicious), "multi": max(1, n_sites - 2 * max(1, n_malicious) - 2)}

    gw, _ = run_federated(
        sites,
        rounds=rounds,
        local_epochs=3,
        strategy=strategy,
        clip_norm=1.0,
        noise_std=0.0,
        validate=validate,
        max_norm=3.0,
        seed=seed,
        verbose=False,
        strategy_kw=kw,
    )
    probe = GCNAutoencoder(sites[0].model.cfg)
    probe.set_weights(gw)
    # measure on the HONEST sites only - the attacker's own detection quality is
    # not what we are protecting
    honest = sites[n_malicious:] or sites
    return float(np.mean([evaluate(probe, s.test)["auc"] for s in honest]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=2400.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    common = dict(
        n_sites=args.sites, rounds=args.rounds, seed=args.seed,
        duration=args.duration, window=args.window,
    )

    rows = []

    # ------------------------------------------------------------------ T4/T5
    for kind in ("sign_flip", "scale"):
        print("=" * 88)
        print(f"T4/T5 MALICIOUS CLIENT UPDATES - {kind}   (mean AUC on honest sites)")
        print("=" * 88)
        header = f"{'malicious sites':<18}" + "".join(f"{s:>16}" for s in STRATEGIES)
        print(header)
        print("-" * len(header))
        curves = {s: [] for s in STRATEGIES}
        counts = list(range(0, args.sites // 2 + 1))
        for nm in counts:
            cells = []
            for strat in STRATEGIES:
                auc = run_case(n_malicious=nm, kind=kind, strategy=strat,
                               validate=False, **common)
                cells.append(auc)
                curves[strat].append(auc)
                rows.append({"attack": kind, "n_malicious": nm, "strategy": strat,
                             "validation": False, "mean_auc_honest": round(auc, 4)})
            print(f"{nm:<18}" + "".join(f"{c:>16.3f}" for c in cells))
        print()

        # the same thing with cheap server-side norm validation switched on
        print(f"{'with norm validation':<18}" + "".join(f"{s:>16}" for s in STRATEGIES))
        print("-" * len(header))
        for nm in counts:
            cells = []
            for strat in STRATEGIES:
                auc = run_case(n_malicious=nm, kind=kind, strategy=strat,
                               validate=True, **common)
                cells.append(auc)
                rows.append({"attack": kind, "n_malicious": nm, "strategy": strat,
                             "validation": True, "mean_auc_honest": round(auc, 4)})
            print(f"{nm:<18}" + "".join(f"{c:>16.3f}" for c in cells))
        print()

        if not args.no_plots:
            plt = _plt()
            plt.figure(figsize=(7.2, 4.4))
            for strat in STRATEGIES:
                plt.plot(counts, curves[strat], marker="o", label=strat)
            plt.xlabel(f"number of malicious sites (of {args.sites})")
            plt.ylabel("mean ROC-AUC on honest sites")
            plt.title(f"Robustness to {kind} client updates (no norm validation)")
            plt.ylim(0.3, 1.0)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"attack_{kind}.png"), dpi=160)
            plt.close()

    # -------------------------------------------------------------------- T2
    print("=" * 88)
    print("T2 STRUCTURAL GRAPH POISONING (targeted)")
    print("   The update is well-formed, so no aggregation rule can see it - but in")
    print("   this configuration it also does not measurably hurt. Two structural")
    print("   reasons, both worth stating rather than hiding: the model has ~240")
    print("   parameters and cannot memorise one site's injected pattern, and the")
    print("   peer-novelty term is computed locally from each site's own history,")
    print("   never learned, so poisoning another site's graphs cannot reach it.")
    print("=" * 88)
    header = f"{'poisoned sites':<18}" + "".join(f"{s:>16}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for nm in range(0, args.sites // 2 + 1):
        cells = []
        for strat in STRATEGIES:
            auc = run_case(n_malicious=nm, kind="graph_poison", strategy=strat,
                           validate=True, **common)
            cells.append(auc)
            rows.append({"attack": "graph_poison", "n_malicious": nm, "strategy": strat,
                         "validation": True, "mean_auc_honest": round(auc, 4)})
        print(f"{nm:<18}" + "".join(f"{c:>16.3f}" for c in cells))

    out = os.path.join(RESULTS, "attack_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["attack", "n_malicious", "strategy", "validation", "mean_auc_honest"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
