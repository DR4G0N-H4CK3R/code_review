"""
Dynamic graph construction (week-7 deliverable 2 and 3).

Turns a stream of flow records into a sequence of device-communication graph
snapshots with node and edge features. Devices are nodes, communications are
edges, exactly as in the architecture figure.

Design decisions worth defending in review:

* One graph per TIME WINDOW, not per flow. The unit of detection is a
  relationship over a window, which is what a per-flow IDS cannot see.
* Node identity is a device handle with ageing, so a device that goes quiet for
  several windows is dropped rather than kept as a dead node forever.
* Every feature is computed from the window alone - no future information, no
  global statistics - so the same code runs online at the edge.
* `peer_novelty` is carried per edge against a per-node history of peers seen in
  previous windows. That is the signal the whole project rests on: "a hacked
  device starts talking to peers it has never contacted before".
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import BENIGN, MALICIOUS, FlowRecord

NODE_FEATURES = [
    "log_deg_in",
    "log_deg_out",
    "log_bytes_in",
    "log_bytes_out",
    "peer_ratio",
    "proto_entropy",
    "port_entropy",
    "external_egress_ratio",
]
EDGE_FEATURES = [
    "log_bytes",
    "log_pkts",
    "log_duration",
    "dport_rarity",
    "is_tcp",
    "peer_novelty",
    "flow_count",
    "time_of_day",
]


@dataclass
class GraphSnapshot:
    """One window. Arrays are plain NumPy so this is framework-agnostic."""

    window_index: int
    t_start: float
    t_end: float
    nodes: List[str]
    X: np.ndarray                     # [N, 8] node features
    edge_index: np.ndarray            # [2, E] int
    edge_attr: np.ndarray             # [E, 8] edge features
    edge_label: np.ndarray            # [E] 1 = contains malicious flow
    A: np.ndarray = field(default=None, repr=False)   # [N, N] normalised adjacency

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def index_of(self, node: str) -> int:
        return self.nodes.index(node)


def _entropy(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps))          # normalised to [0, 1]


def normalised_adjacency(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Symmetric GCN normalisation  D^-1/2 (A + I) D^-1/2, dense."""
    A = np.zeros((n, n), dtype=np.float32)
    for s, d in zip(edge_index[0], edge_index[1]):
        A[s, d] = 1.0
        A[d, s] = 1.0                      # treat the relationship as undirected
    A += np.eye(n, dtype=np.float32)
    deg = A.sum(1)
    dinv = np.power(np.maximum(deg, 1e-8), -0.5)
    return (A * dinv[:, None]) * dinv[None, :]


class GraphBuilder:
    """
    Streaming builder. Call `build_all()` for a batch, or `push()` per window if
    you are wiring it to a live Zeek/SPAN feed on the Raspberry Pi.
    """

    def __init__(
        self,
        window_s: float = 60.0,
        node_ttl_windows: int = 5,
        min_nodes: int = 4,
        external_prefixes: Tuple[str, ...] = ("gateway", "cloud", "8.8.", "1.1."),
    ) -> None:
        self.window_s = window_s
        self.node_ttl_windows = node_ttl_windows
        self.min_nodes = min_nodes
        self.external_prefixes = external_prefixes
        self.peer_history: Dict[str, set] = defaultdict(set)
        self.dport_history: Counter = Counter()
        self.last_seen: Dict[str, int] = {}

    # ------------------------------------------------------------- utilities
    def _is_external(self, node: str) -> bool:
        return any(p in node for p in self.external_prefixes)

    # ----------------------------------------------------------------- build
    def build_all(self, flows: Sequence[FlowRecord]) -> List[GraphSnapshot]:
        if not flows:
            return []
        t0 = flows[0].ts
        buckets: Dict[int, List[FlowRecord]] = defaultdict(list)
        for f in flows:
            buckets[int((f.ts - t0) // self.window_s)].append(f)

        snaps: List[GraphSnapshot] = []
        for w in sorted(buckets):
            snap = self._build_window(w, t0, buckets[w])
            if snap is not None:
                snaps.append(snap)
        return snaps

    def _build_window(self, w: int, t0: float, flows: List[FlowRecord]) -> Optional[GraphSnapshot]:
        # ---- node identity with ageing
        active = {f.src for f in flows} | {f.dst for f in flows}
        for n in active:
            self.last_seen[n] = w
        alive = sorted(n for n, lw in self.last_seen.items() if w - lw < self.node_ttl_windows)
        if len(alive) < self.min_nodes:
            return None
        idx = {n: i for i, n in enumerate(alive)}

        # ---- aggregate flows onto edges
        agg: Dict[Tuple[str, str], Dict] = {}
        for f in flows:
            if f.src not in idx or f.dst not in idx:
                continue
            key = (f.src, f.dst)
            a = agg.setdefault(
                key,
                {"bytes": 0.0, "pkts": 0.0, "dur": 0.0, "n": 0, "dports": Counter(),
                 "protos": Counter(), "label": BENIGN, "ts": f.ts},
            )
            a["bytes"] += f.bytes_
            a["pkts"] += f.pkts
            a["dur"] += f.duration
            a["n"] += 1
            a["dports"][f.dport] += 1
            a["protos"][f.proto] += 1
            a["label"] = max(a["label"], f.label)

        if not agg:
            return None

        # ---- node feature accumulators
        deg_in = Counter(); deg_out = Counter()
        b_in = Counter(); b_out = Counter()
        peers: Dict[str, set] = defaultdict(set)
        protos: Dict[str, Counter] = defaultdict(Counter)
        ports: Dict[str, Counter] = defaultdict(Counter)
        ext_bytes = Counter(); tot_bytes = Counter()

        for (s, d), a in agg.items():
            deg_out[s] += 1; deg_in[d] += 1
            b_out[s] += a["bytes"]; b_in[d] += a["bytes"]
            peers[s].add(d); peers[d].add(s)
            protos[s].update(a["protos"]); protos[d].update(a["protos"])
            ports[s].update(a["dports"]); ports[d].update(a["dports"])
            tot_bytes[s] += a["bytes"]; tot_bytes[d] += a["bytes"]
            if self._is_external(d):
                ext_bytes[s] += a["bytes"]

        N = len(alive)
        X = np.zeros((N, len(NODE_FEATURES)), dtype=np.float32)
        for n, i in idx.items():
            X[i] = [
                math.log1p(deg_in[n]),
                math.log1p(deg_out[n]),
                math.log1p(b_in[n]),
                math.log1p(b_out[n]),
                len(peers[n]) / max(N - 1, 1),
                _entropy(list(protos[n].values())),
                _entropy(list(ports[n].values())),
                ext_bytes[n] / max(tot_bytes[n], 1.0),
            ]

        # ---- edges
        E = len(agg)
        edge_index = np.zeros((2, E), dtype=np.int64)
        edge_attr = np.zeros((E, len(EDGE_FEATURES)), dtype=np.float32)
        edge_label = np.zeros(E, dtype=np.int64)

        total_dports = max(sum(self.dport_history.values()), 1)
        for e, ((s, d), a) in enumerate(sorted(agg.items())):
            edge_index[0, e] = idx[s]
            edge_index[1, e] = idx[d]
            top_dport = a["dports"].most_common(1)[0][0]
            rarity = 1.0 - (self.dport_history[top_dport] / total_dports)
            novelty = 0.0 if d in self.peer_history[s] else 1.0
            tod = ((a["ts"] - t0) % 86400.0) / 86400.0
            edge_attr[e] = [
                math.log1p(a["bytes"]),
                math.log1p(a["pkts"]),
                math.log1p(a["dur"]),
                rarity,
                1.0 if a["protos"].most_common(1)[0][0] == "tcp" else 0.0,
                novelty,
                math.log1p(a["n"]),
                tod,
            ]
            edge_label[e] = a["label"]

        # ---- update history AFTER featurising (no leakage into this window)
        for (s, d), a in agg.items():
            self.peer_history[s].add(d)
            self.peer_history[d].add(s)
            self.dport_history.update(a["dports"])

        snap = GraphSnapshot(
            window_index=w,
            t_start=t0 + w * self.window_s,
            t_end=t0 + (w + 1) * self.window_s,
            nodes=alive,
            X=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label=edge_label,
        )
        snap.A = normalised_adjacency(edge_index, N)
        return snap


def standardise(snaps: Sequence[GraphSnapshot]) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean/std on a set of snapshots (a site's own training data)."""
    allX = np.concatenate([s.X for s in snaps], axis=0)
    mu = allX.mean(0)
    sd = allX.std(0) + 1e-6
    return mu, sd


def apply_standardise(snaps: Sequence[GraphSnapshot], mu: np.ndarray, sd: np.ndarray) -> None:
    for s in snaps:
        s.X = (s.X - mu) / sd
