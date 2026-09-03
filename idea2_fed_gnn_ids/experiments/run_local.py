"""
Single-site pipeline, end to end.

flows -> windowed graphs -> node/edge features -> 2-layer GCN autoencoder
      -> per-edge anomaly scores -> attribution -> temporal consensus

Run:
  python -m experiments.run_local                       # synthetic, runs today
  python -m experiments.run_local --source ciciot2023 --data /path/to.csv
  python -m experiments.run_local --source iot23 --data /path/conn.csv --limit 500000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fedgnn import (
    EDGE_FEATURES,
    NODE_FEATURES,
    GCNAutoencoder,
    GCNConfig,
    GraphBuilder,
    alerts_for_snapshot,
    evaluate,
    load_source,
    temporal_consensus,
)
from fedgnn.graphs import apply_standardise, standardise
from fedgnn.scoring import _components, _rank_normalise, roc_auc, score_snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "ciciot2023", "iot23"])
    ap.add_argument("--data", default=None, help="path to the dataset CSV")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--devices", type=int, default=24)
    ap.add_argument("--duration", type=float, default=2400.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("=" * 78)
    print(f"1. LOADING FLOWS  (source={args.source})")
    print("=" * 78)
    if args.source == "synthetic":
        flows = load_source(
            "synthetic", site_id=0, seed=args.seed,
            n_devices=args.devices, duration_s=args.duration, attack_rate=0.03,
        )
    else:
        flows = load_source(args.source, path=args.data, limit=args.limit)
    mal = sum(f.label for f in flows)
    print(f"   {len(flows)} flow records, {mal} labelled malicious "
          f"({100.0 * mal / max(len(flows), 1):.2f}%)")
    print(f"   time span {flows[-1].ts - flows[0].ts:.0f} s, "
          f"{len({f.src for f in flows} | {f.dst for f in flows})} distinct devices")

    print("\n" + "=" * 78)
    print(f"2. GRAPH CONSTRUCTION  (window = {args.window:.0f} s)")
    print("=" * 78)
    snaps = GraphBuilder(window_s=args.window).build_all(flows)
    if len(snaps) < 4:
        print("   not enough windows to train on - lengthen the capture or shrink --window")
        return
    print(f"   {len(snaps)} graph snapshots")
    print(f"   nodes/window : min {min(s.n_nodes for s in snaps)}  "
          f"max {max(s.n_nodes for s in snaps)}  "
          f"mean {np.mean([s.n_nodes for s in snaps]):.1f}")
    print(f"   edges/window : min {min(s.n_edges for s in snaps)}  "
          f"max {max(s.n_edges for s in snaps)}  "
          f"mean {np.mean([s.n_edges for s in snaps]):.1f}")
    print(f"   node features ({len(NODE_FEATURES)}): {', '.join(NODE_FEATURES)}")
    print(f"   edge features ({len(EDGE_FEATURES)}): {', '.join(EDGE_FEATURES)}")

    cut = max(2, int(len(snaps) * 0.5))
    train, test = snaps[:cut], snaps[cut:]
    mu, sd = standardise(train)
    apply_standardise(train, mu, sd)
    apply_standardise(test, mu, sd)
    print(f"   split: {len(train)} training windows / {len(test)} test windows")
    print(f"   malicious edges in test: {sum(int(s.edge_label.sum()) for s in test)}")

    print("\n" + "=" * 78)
    print("3. TRAINING  (2-layer GCN autoencoder, 8 -> 16 -> 2)")
    print("=" * 78)
    model = GCNAutoencoder(GCNConfig(seed=0))
    print(f"   trainable parameters: {model.n_params()}")
    hist = model.fit(train, epochs=args.epochs)
    for i in range(0, len(hist), max(1, len(hist) // 6)):
        print(f"   epoch {i + 1:>3}  loss = {hist[i]:.4f}")
    print(f"   epoch {len(hist):>3}  loss = {hist[-1]:.4f}")

    print("\n" + "=" * 78)
    print("4. DETECTION")
    print("=" * 78)
    m = evaluate(model, test, top_k=10)
    print(f"   edges scored        : {m['n_edges']}")
    print(f"   malicious edges     : {m['n_malicious']}")
    print(f"   ROC-AUC             : {m['auc']:.3f}")
    print(f"   precision@10        : {m['p@k']:.3f}")
    print(f"   recall@10           : {m['r@k']:.3f}")

    # per-component contribution, so a reviewer can see what is carrying the signal
    ys, comps = [], {i: [] for i in range(4)}
    for s in test:
        if s.n_edges == 0:
            continue
        c = _components(model, s)
        for i in range(3):
            comps[i].append(_rank_normalise(np.asarray(c[i], dtype=float)))
        comps[3].append(np.asarray(c[3], dtype=float))
        ys.append(s.edge_label)
    y = np.concatenate(ys)
    print("\n   per-component AUC:")
    for i, name in enumerate(["structural", "node feature", "edge feature", "peer novelty"]):
        print(f"     {name:<15} {roc_auc(y, np.concatenate(comps[i])):.3f}")

    print("\n" + "=" * 78)
    print("5. TOP ALERTS WITH ATTRIBUTION")
    print("=" * 78)
    shown = 0
    for s in test:
        for a in alerts_for_snapshot(model, s, top_k=2):
            flag = "TRUE POSITIVE " if a.label else "false positive"
            print(f"   w{a.window_index:<3} [{flag}] {a.explain()}")
            shown += 1
        if shown >= 12:
            break

    print("\n" + "=" * 78)
    print("6. TEMPORAL CONSENSUS  (2 of the last 3 windows above 0.7)")
    print("=" * 78)
    scored = [(s, score_snapshot(model, s)) for s in test if s.n_edges]
    confirmed = temporal_consensus(scored, k_of_n=(2, 3), threshold=0.7)
    if not confirmed:
        print("   no relationship confirmed across windows")
    for (u, v), w in sorted(confirmed.items(), key=lambda kv: kv[1]):
        print(f"   confirmed incident at window {w}: {u} -> {v}")


if __name__ == "__main__":
    main()
