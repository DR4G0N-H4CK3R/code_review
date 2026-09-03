"""
Per-edge anomaly scoring, attribution and temporal consensus.

Score for edge (u, v) in a window:

    score = w1*structural + w2*node_feature + w3*edge_feature + w4*peer_novelty

  structural     1 - sigmoid(z_u . z_v)    the model does not believe this
                                           relationship should exist at all
  node_feature   mean reconstruction error of the two endpoints - a device whose
                                           own behaviour changed
  edge_feature   reconstruction error of the edge's own profile from the two
                                           endpoint embeddings - a relationship
                                           that does not look like anything these
                                           two devices should be doing
  peer_novelty   1 if v is a peer u has never contacted in any prior window

Each term is rank-normalised inside its window before mixing, so the three are
comparable regardless of scale and no per-site threshold calibration is needed.

Attribution simply reports the three terms, which is what an analyst needs to
act: "this edge is anomalous because the peer is new, not because the volume
changed".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graphs import EDGE_FEATURES, GraphSnapshot
from .model_numpy import sigmoid

NOVELTY_COL = EDGE_FEATURES.index("peer_novelty")

DEFAULT_WEIGHTS = (0.10, 0.35, 0.35, 0.20)   # struct, node feat, edge feat, novelty


def _rank_normalise(v: np.ndarray) -> np.ndarray:
    """Map to [0, 1] by rank. Robust to outliers and to per-site scale drift."""
    if v.size == 0:
        return v
    if v.size == 1:
        return np.zeros(1, dtype=np.float32)
    order = v.argsort()
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average tied ranks, otherwise a mostly-constant component would fabricate
    # an ordering out of array position
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks / max(v.size - 1, 1)).astype(np.float32)


@dataclass
class EdgeAlert:
    window_index: int
    src: str
    dst: str
    score: float
    structural: float
    feature: float
    edge_feature: float
    novelty: float
    label: int

    def explain(self) -> str:
        parts = [
            ("unexpected relationship", self.structural),
            ("endpoint behaviour change", self.feature),
            ("unusual traffic profile for this pair", self.edge_feature),
            ("never-seen peer", self.novelty),
        ]
        top = max(parts, key=lambda p: p[1])
        return f"{self.src} -> {self.dst}  score={self.score:.3f}  driver: {top[0]} ({top[1]:.2f})"


def _components(model, snap: GraphSnapshot):
    """Return the four raw per-edge score components."""
    f = model.forward(snap.X, snap.A, training=False)
    Z, Xhat = f["Z"], f["Xhat"]
    s, d = snap.edge_index[0], snap.edge_index[1]

    struct = 1.0 - sigmoid(np.einsum("ij,ij->i", Z[s], Z[d]))

    node_err = np.mean((Xhat - snap.X) ** 2, axis=1)
    feat = 0.5 * (node_err[s] + node_err[d])

    Ehat = model.decode_edges(Z, snap.edge_index)
    edge_err = np.mean((Ehat - snap.edge_attr) ** 2, axis=1)

    nov = snap.edge_attr[:, NOVELTY_COL]
    return struct, feat, edge_err, nov


def score_snapshot(
    model, snap: GraphSnapshot, weights: Tuple[float, ...] = DEFAULT_WEIGHTS
) -> np.ndarray:
    """Return per-edge anomaly scores in [0, 1]."""
    if snap.n_edges == 0:
        return np.zeros(0, dtype=np.float32)
    struct, feat, edge_err, nov = _components(model, snap)
    w1, w2, w3, w4 = weights
    return (
        w1 * _rank_normalise(struct)
        + w2 * _rank_normalise(feat)
        + w3 * _rank_normalise(edge_err)
        + w4 * nov
    ).astype(np.float32)


def alerts_for_snapshot(
    model, snap: GraphSnapshot, top_k: int = 5, weights=DEFAULT_WEIGHTS
) -> List[EdgeAlert]:
    scores = score_snapshot(model, snap, weights)
    if scores.size == 0:
        return []
    raw_struct, raw_feat, raw_edge, nov = _components(model, snap)
    struct = _rank_normalise(raw_struct)
    feat = _rank_normalise(raw_feat)
    edge_f = _rank_normalise(raw_edge)
    s, d = snap.edge_index[0], snap.edge_index[1]

    order = np.argsort(-scores)[:top_k]
    return [
        EdgeAlert(
            window_index=snap.window_index,
            src=snap.nodes[s[e]],
            dst=snap.nodes[d[e]],
            score=float(scores[e]),
            structural=float(struct[e]),
            feature=float(feat[e]),
            edge_feature=float(edge_f[e]),
            novelty=float(nov[e]),
            label=int(snap.edge_label[e]),
        )
        for e in order
    ]


def temporal_consensus(
    per_window_scores: Sequence[Tuple[GraphSnapshot, np.ndarray]],
    k_of_n: Tuple[int, int] = (2, 3),
    threshold: float = 0.7,
) -> Dict[Tuple[str, str], int]:
    """
    Multi-window consensus: an edge only becomes an incident if it scores above
    `threshold` in k of the last n windows. Cuts single-window false positives
    without adding a tunable per-site threshold.
    """
    k, n = k_of_n
    hits: Dict[Tuple[str, str], List[int]] = {}
    for snap, scores in per_window_scores:
        for e in range(snap.n_edges):
            if scores[e] >= threshold:
                key = (snap.nodes[snap.edge_index[0, e]], snap.nodes[snap.edge_index[1, e]])
                hits.setdefault(key, []).append(snap.window_index)

    confirmed: Dict[Tuple[str, str], int] = {}
    for key, wins in hits.items():
        wins.sort()
        for i in range(len(wins)):
            recent = [w for w in wins if wins[i] - n < w <= wins[i]]
            if len(recent) >= k:
                confirmed[key] = wins[i]
                break
    return confirmed


# --------------------------------------------------------------------------
# Metrics (no sklearn dependency, so this runs on the Pi too)
# --------------------------------------------------------------------------
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC. Returns 0.5 when a class is missing."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = s.argsort()
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    k = min(k, len(y))
    idx = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[idx].sum() / k)


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    y = np.asarray(y)
    total = int(y.sum())
    if total == 0:
        return 0.0
    k = min(k, len(y))
    idx = np.argsort(-np.asarray(s))[:k]
    return float(y[idx].sum() / total)


def evaluate(model, snaps: Sequence[GraphSnapshot], top_k: int = 10) -> Dict[str, float]:
    ys, ss = [], []
    for snap in snaps:
        if snap.n_edges == 0:
            continue
        ss.append(score_snapshot(model, snap))
        ys.append(snap.edge_label)
    if not ys:
        return {"auc": 0.5, "p@k": 0.0, "r@k": 0.0, "n_edges": 0, "n_malicious": 0}
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    return {
        "auc": roc_auc(y, s),
        "p@k": precision_at_k(y, s, top_k),
        "r@k": recall_at_k(y, s, top_k),
        "n_edges": int(len(y)),
        "n_malicious": int(y.sum()),
    }
