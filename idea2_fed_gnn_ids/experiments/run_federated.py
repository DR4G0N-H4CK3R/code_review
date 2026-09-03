"""
The week-7 headline experiment: real traffic through the federated loop, end to end.

Answers three questions a reviewer will ask:

  Q1  Does federation actually help, or would each site be better off alone?
      -> local-only vs FedAvg vs centralised upper bound, per site.

  Q2  What does non-IID cost?
      -> each site has a different device role mix, so the graphs genuinely
         differ in shape. Reported as the spread of per-site AUC.

  Q3  What does privacy cost?
      -> sweep the DP-lite noise multiplier and watch AUC fall.

Run:
  python -m experiments.run_federated
  python -m experiments.run_federated --sites 6 --rounds 20
  python -m experiments.run_federated --source ciciot2023 --data /path/to.csv
Writes results/federated_*.csv and results/federated_*.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedgnn import (
    GCNAutoencoder,
    GCNConfig,
    build_sites,
    evaluate,
    run_federated,
    personalise,
    train_centralised,
    train_local_only,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=2400.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()

    print("=" * 84)
    print(f"BUILDING {args.sites} FEDERATED SITES  (source={args.source})")
    print("=" * 84)
    sites = build_sites(
        n_sites=args.sites,
        seed=args.seed,
        duration_s=args.duration,
        window_s=args.window,
        source=args.source,
        data_path=args.data,
        limit=args.limit,
    )
    for s in sites:
        mal = sum(int(x.edge_label.sum()) for x in s.test)
        edges = sum(x.n_edges for x in s.test)
        print(
            f"  {s.name:<12} train windows {len(s.train):>3}  test windows {len(s.test):>3}  "
            f"test edges {edges:>5}  malicious {mal:>3}"
        )
    print(f"  model: {sites[0].model.n_params()} parameters "
          f"({sites[0].model.n_params() * 4 / 1024:.1f} KB per update)")

    # ---------------------------------------------------------------- Q1 + Q2
    print("\n" + "=" * 84)
    print("BASELINE A - local only (each site trains alone, never shares)")
    print("=" * 84)
    local = {s.name: train_local_only(s, epochs=args.rounds * args.local_epochs) for s in sites}
    for n, m in local.items():
        print(f"  {n:<12} AUC {m['auc']:.3f}   p@10 {m['p@k']:.2f}")

    print("\n" + "=" * 84)
    print("BASELINE B - centralised (all raw graphs pooled: what privacy forbids)")
    print("=" * 84)
    central = train_centralised(sites, epochs=args.rounds * args.local_epochs)
    for n, m in central.items():
        print(f"  {n:<12} AUC {m['auc']:.3f}   p@10 {m['p@k']:.2f}")

    print("\n" + "=" * 84)
    print(f"FEDERATED - FedAvg, {args.rounds} rounds x {args.local_epochs} local epochs")
    print("=" * 84)
    gw, logs = run_federated(
        sites,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        strategy="fedavg",
        clip_norm=1.0,
        noise_std=0.0,
        seed=args.seed,
    )
    probe = GCNAutoencoder(sites[0].model.cfg)
    probe.set_weights(gw)
    fed = {s.name: evaluate(probe, s.test) for s in sites}

    print("\n" + "=" * 84)
    print("FEDERATED + LOCAL FINE-TUNE (personalisation, no extra communication)")
    print("=" * 84)
    pers = {s.name: personalise(s, gw, epochs=args.local_epochs * 2) for s in sites}

    print("\n" + "-" * 92)
    print(f"{'site':<12}{'local only':>12}{'federated':>12}{'fed+finetune':>14}"
          f"{'centralised':>13}{'best':>14}")
    print("-" * 92)
    for s in sites:
        l, f = local[s.name]["auc"], fed[s.name]["auc"]
        pfa, c = pers[s.name]["auc"], central[s.name]["auc"]
        best = max([("local", l), ("federated", f), ("fed+ft", pfa), ("central", c)],
                   key=lambda kv: kv[1])[0]
        print(f"{s.name:<12}{l:>12.3f}{f:>12.3f}{pfa:>14.3f}{c:>13.3f}{best:>14}")
    print("-" * 92)
    print(
        f"{'MEAN':<12}"
        f"{np.mean([v['auc'] for v in local.values()]):>12.3f}"
        f"{np.mean([v['auc'] for v in fed.values()]):>12.3f}"
        f"{np.mean([v['auc'] for v in pers.values()]):>14.3f}"
        f"{np.mean([v['auc'] for v in central.values()]):>13.3f}"
    )
    spread = lambda d: max(v["auc"] for v in d.values()) - min(v["auc"] for v in d.values())
    print(
        f"non-IID spread (max - min per-site AUC):  local {spread(local):.3f}   "
        f"federated {spread(fed):.3f}   fed+finetune {spread(pers):.3f}"
    )

    with open(os.path.join(RESULTS, "federated_per_site.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "local_auc", "federated_auc", "fed_finetune_auc", "centralised_auc",
                    "local_p@10", "federated_p@10", "fed_finetune_p@10",
                    "test_edges", "malicious_edges"])
        for s in sites:
            w.writerow([s.name, round(local[s.name]["auc"], 4), round(fed[s.name]["auc"], 4),
                        round(pers[s.name]["auc"], 4), round(central[s.name]["auc"], 4),
                        round(local[s.name]["p@k"], 3), round(fed[s.name]["p@k"], 3),
                        round(pers[s.name]["p@k"], 3),
                        fed[s.name]["n_edges"], fed[s.name]["n_malicious"]])

    with open(os.path.join(RESULTS, "federated_rounds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "train_loss", "global_auc"] + [s.name for s in sites])
        for lg in logs:
            w.writerow([lg.rnd, round(lg.train_loss, 5), round(lg.global_auc, 4)]
                       + [round(lg.per_site_auc.get(s.name, float("nan")), 4) for s in sites])

    # ------------------------------------------------------------------- Q3
    print("\n" + "=" * 84)
    print("PRIVACY COST - clip to 1.0, then add Gaussian noise to every update")
    print("=" * 84)
    noise_rows = []
    for noise in (0.0, 0.01, 0.05, 0.1, 0.25):
        fresh = build_sites(
            n_sites=args.sites, seed=args.seed, duration_s=args.duration,
            window_s=args.window, source=args.source, data_path=args.data, limit=args.limit,
        )
        gw2, _ = run_federated(
            fresh, rounds=args.rounds, local_epochs=args.local_epochs,
            strategy="fedavg", clip_norm=1.0, noise_std=noise,
            seed=args.seed, verbose=False,
        )
        p2 = GCNAutoencoder(fresh[0].model.cfg)
        p2.set_weights(gw2)
        aucs = [evaluate(p2, s.test)["auc"] for s in fresh]
        print(f"  noise multiplier {noise:<5}  mean AUC {np.mean(aucs):.3f}  "
              f"(min {min(aucs):.3f}, max {max(aucs):.3f})")
        noise_rows.append((noise, float(np.mean(aucs)), float(min(aucs)), float(max(aucs))))

    with open(os.path.join(RESULTS, "federated_privacy.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise_multiplier", "mean_auc", "min_auc", "max_auc"])
        for r in noise_rows:
            w.writerow([r[0], round(r[1], 4), round(r[2], 4), round(r[3], 4)])

    # ---------------------------------------------------------------- plots
    if not args.no_plots:
        plt = _plt()

        plt.figure(figsize=(7.4, 4.4))
        for s in sites:
            plt.plot([lg.rnd for lg in logs],
                     [lg.per_site_auc.get(s.name, np.nan) for lg in logs],
                     alpha=0.45, lw=1)
        plt.plot([lg.rnd for lg in logs], [lg.global_auc for lg in logs],
                 lw=2.4, color="black", label="mean across sites")
        plt.xlabel("federated round")
        plt.ylabel("ROC-AUC on held-out windows")
        plt.title("Global model quality per round (thin lines = individual sites)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "federated_rounds.png"), dpi=160)
        plt.close()

        names = [s.name for s in sites]
        x = np.arange(len(names))
        plt.figure(figsize=(9.0, 4.4))
        plt.bar(x - 0.30, [local[n]["auc"] for n in names], 0.20, label="local only")
        plt.bar(x - 0.10, [fed[n]["auc"] for n in names], 0.20, label="federated (FedAvg)")
        plt.bar(x + 0.10, [pers[n]["auc"] for n in names], 0.20, label="federated + fine-tune")
        plt.bar(x + 0.30, [central[n]["auc"] for n in names], 0.20, label="centralised")
        plt.xticks(x, names, rotation=20, fontsize=8)
        plt.ylabel("ROC-AUC")
        plt.ylim(0, 1.05)
        plt.title("Per-site detection quality under four training regimes")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "federated_per_site.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(7.0, 4.2))
        ns = [r[0] for r in noise_rows]
        plt.plot(ns, [r[1] for r in noise_rows], marker="o")
        plt.fill_between(ns, [r[2] for r in noise_rows], [r[3] for r in noise_rows], alpha=0.2)
        plt.xlabel("Gaussian noise multiplier on the clipped update")
        plt.ylabel("mean ROC-AUC")
        plt.title("Privacy / utility trade-off")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "federated_privacy.png"), dpi=160)
        plt.close()
        print(f"\nfigures written to {RESULTS}")

    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
