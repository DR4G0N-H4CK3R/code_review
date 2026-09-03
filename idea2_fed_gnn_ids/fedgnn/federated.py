"""
Federated training loop, simulated in one process.

This is the same protocol the Flower scripts run (`flower_client.py` /
`flower_server.py`), minus the gRPC transport. Having both matters for review:
the Flower run proves the plumbing works, this one lets you sweep 30 rounds x 8
sites x 5 aggregation strategies in a few seconds and actually get curves.

Non-IID is not simulated by shuffling labels. Each site gets a different DEVICE
ROLE MIX, so its communication graph genuinely has a different shape - which is
the real reason plain FedAvg struggles here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate, clip_and_noise, validate_update
from .datasets import load_source
from .graphs import GraphBuilder, GraphSnapshot, apply_standardise, standardise
from .model_numpy import GCNAutoencoder, GCNConfig
from .scoring import evaluate


# --------------------------------------------------------------------------
SITE_PROFILES = [
    # deliberately different mixes -> different topologies -> non-IID
    {"name": "home-a",    "camera": 0.10, "bulb": 0.40, "thermostat": 0.12, "speaker": 0.10,
     "nas": 0.02, "laptop": 0.10, "printer": 0.02, "hub": 0.08, "nvr": 0.02, "gateway": 0.04},
    {"name": "office-b",  "camera": 0.08, "bulb": 0.10, "thermostat": 0.06, "speaker": 0.04,
     "nas": 0.08, "laptop": 0.36, "printer": 0.10, "hub": 0.06, "nvr": 0.04, "gateway": 0.08},
    {"name": "campus-c",  "camera": 0.34, "bulb": 0.10, "thermostat": 0.06, "speaker": 0.04,
     "nas": 0.06, "laptop": 0.12, "printer": 0.04, "hub": 0.08, "nvr": 0.10, "gateway": 0.06},
    {"name": "clinic-d",  "camera": 0.14, "bulb": 0.16, "thermostat": 0.20, "speaker": 0.06,
     "nas": 0.06, "laptop": 0.18, "printer": 0.08, "hub": 0.06, "nvr": 0.02, "gateway": 0.04},
    {"name": "retail-e",  "camera": 0.24, "bulb": 0.24, "thermostat": 0.08, "speaker": 0.12,
     "nas": 0.02, "laptop": 0.10, "printer": 0.04, "hub": 0.08, "nvr": 0.04, "gateway": 0.04},
    {"name": "flat-f",    "camera": 0.06, "bulb": 0.44, "thermostat": 0.10, "speaker": 0.16,
     "nas": 0.02, "laptop": 0.10, "printer": 0.02, "hub": 0.06, "nvr": 0.00, "gateway": 0.04},
]


@dataclass
class Site:
    """One federated client."""

    site_id: int
    name: str
    train: List[GraphSnapshot]
    test: List[GraphSnapshot]
    model: GCNAutoencoder
    malicious: str = "none"      # none | sign_flip | scale | graph_poison

    @property
    def n_train(self) -> int:
        return len(self.train)


def build_sites(
    n_sites: int = 4,
    seed: int = 0,
    n_devices: int = 24,
    duration_s: float = 2400.0,
    window_s: float = 60.0,
    train_frac: float = 0.5,
    source: str = "synthetic",
    data_path: Optional[str] = None,
    limit: Optional[int] = None,
    cfg: Optional[GCNConfig] = None,
) -> List[Site]:
    """
    Build one client per site.

    With `source="synthetic"` each site gets its own role mix. With a real
    dataset the flows are split into `n_sites` contiguous chunks, which keeps
    each site's device population coherent instead of interleaving them.
    """
    sites: List[Site] = []

    if source != "synthetic":
        flows = load_source(source, path=data_path, limit=limit)
        chunk = max(1, len(flows) // n_sites)
        chunks = [flows[i * chunk : (i + 1) * chunk] for i in range(n_sites)]
    else:
        chunks = [None] * n_sites

    for i in range(n_sites):
        profile = SITE_PROFILES[i % len(SITE_PROFILES)]
        name = f"{profile['name']}-{i}"
        if source == "synthetic":
            mix = {k: v for k, v in profile.items() if k != "name"}
            flows_i = load_source(
                "synthetic",
                site_id=i,
                seed=seed,
                n_devices=n_devices,
                role_mix=mix,
                duration_s=duration_s,
                attack_start_frac=0.6,
                attack_rate=0.03,
            )
        else:
            flows_i = chunks[i]

        snaps = GraphBuilder(window_s=window_s).build_all(flows_i)
        if len(snaps) < 4:
            continue
        cut = max(2, int(len(snaps) * train_frac))
        train, test = snaps[:cut], snaps[cut:]

        # standardise on the site's OWN training windows - no cross-site leakage
        mu, sd = standardise(train)
        apply_standardise(train, mu, sd)
        apply_standardise(test, mu, sd)

        sites.append(
            Site(
                site_id=i,
                name=name,
                train=train,
                test=test,
                model=GCNAutoencoder(cfg or GCNConfig(seed=seed + i)),
            )
        )
    return sites


# --------------------------------------------------------------------------
def local_train(
    site: Site,
    global_weights: List[np.ndarray],
    epochs: int = 3,
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    rng=None,
) -> Tuple[List[np.ndarray], int, float]:
    """Client round: load global weights, train locally, return the new weights."""
    site.model.set_weights(global_weights)

    train = site.train
    if site.malicious == "graph_poison":
        train = _poison_graphs(train, rng)

    hist = site.model.fit(train, epochs=epochs)
    new_w = site.model.get_weights()

    # send a bounded, optionally noised DELTA, then re-express as weights
    delta = [n - g for n, g in zip(new_w, global_weights)]
    delta = clip_and_noise(delta, clip_norm=clip_norm, noise_std=noise_std, rng=rng)

    if site.malicious == "sign_flip":
        delta = [-8.0 * d for d in delta]
    elif site.malicious == "scale":
        delta = [25.0 * d for d in delta]

    out = [g + d for g, d in zip(global_weights, delta)]
    return out, site.n_train, (hist[-1] if hist else 0.0)


def _poison_graphs(snaps: Sequence[GraphSnapshot], rng=None, ratio: float = 0.35) -> List[GraphSnapshot]:
    """
    T2 structural graph poisoning, TARGETED.

    Random extra edges barely move the model - we measured that first and it was
    not an attack. The version that matters mimics the signature the detector
    relies on: the client injects low-role -> high-role edges (camera/bulb
    reaching a laptop/NAS) carrying a never-seen-peer flag and a rare
    destination port, and labels them normal. If the global model learns to
    reconstruct that profile, the edge-feature and novelty components of the
    score both lose their grip.
    """
    from .graphs import EDGE_FEATURES, normalised_adjacency

    rng = rng or np.random.default_rng(0)
    novelty_col = EDGE_FEATURES.index("peer_novelty")
    rarity_col = EDGE_FEATURES.index("dport_rarity")

    out = []
    for s in snaps:
        s2 = copy.copy(s)
        n = s.n_nodes
        if n < 3 or s.n_edges == 0:
            out.append(s)
            continue
        low = [i for i, name in enumerate(s.nodes) if "camera" in name or "bulb" in name]
        high = [i for i, name in enumerate(s.nodes)
                if any(t in name for t in ("laptop", "nas", "printer", "nvr"))]
        if not low or not high:
            low = list(range(n))
            high = list(range(n))

        extra = max(1, int(ratio * s.n_edges))
        new_src = rng.choice(low, size=extra)
        new_dst = rng.choice(high, size=extra)

        template = s.edge_attr.mean(0).copy()
        template[novelty_col] = 1.0
        template[rarity_col] = float(np.percentile(s.edge_attr[:, rarity_col], 95))
        attrs = np.tile(template, (extra, 1)).astype(np.float32)
        attrs += rng.normal(0.0, 0.05, size=attrs.shape).astype(np.float32)

        s2.edge_index = np.concatenate(
            [s.edge_index, np.stack([new_src, new_dst]).astype(np.int64)], axis=1
        )
        s2.edge_attr = np.concatenate([s.edge_attr, attrs], axis=0)
        s2.edge_label = np.concatenate([s.edge_label, np.zeros(extra, dtype=np.int64)])
        s2.A = normalised_adjacency(s2.edge_index, n)
        out.append(s2)
    return out


# --------------------------------------------------------------------------
@dataclass
class RoundLog:
    rnd: int
    train_loss: float
    global_auc: float
    per_site_auc: Dict[str, float] = field(default_factory=dict)
    rejected: int = 0


def run_federated(
    sites: Sequence[Site],
    rounds: int = 15,
    local_epochs: int = 3,
    strategy: str = "fedavg",
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    validate: bool = True,
    max_norm: float = 10.0,
    seed: int = 0,
    verbose: bool = True,
    strategy_kw: Optional[dict] = None,
) -> Tuple[List[np.ndarray], List[RoundLog]]:
    rng = np.random.default_rng(seed)
    global_w = sites[0].model.get_weights()
    logs: List[RoundLog] = []

    for r in range(1, rounds + 1):
        updates, sizes, losses, rejected = [], [], [], 0
        for site in sites:
            w, n, loss = local_train(
                site, global_w, epochs=local_epochs,
                clip_norm=clip_norm, noise_std=noise_std, rng=rng,
            )
            if validate:
                delta = [a - b for a, b in zip(w, global_w)]
                ok, _ = validate_update(delta, max_norm=max_norm)
                if not ok:
                    rejected += 1
                    continue
            updates.append(w)
            sizes.append(n)
            losses.append(loss)

        if not updates:                        # everything was rejected
            logs.append(RoundLog(r, float("nan"), logs[-1].global_auc if logs else 0.5, {}, rejected))
            continue

        global_w = aggregate(strategy, updates, sizes, **(strategy_kw or {}))

        # evaluate the global model on every site's held-out windows
        per_site, aucs = {}, []
        probe = GCNAutoencoder(sites[0].model.cfg)
        probe.set_weights(global_w)
        for site in sites:
            m = evaluate(probe, site.test)
            per_site[site.name] = m["auc"]
            aucs.append(m["auc"])
        log = RoundLog(r, float(np.mean(losses)) if losses else float("nan"),
                       float(np.mean(aucs)), per_site, rejected)
        logs.append(log)
        if verbose:
            print(
                f"  round {r:>2}  loss={log.train_loss:7.4f}  "
                f"global AUC={log.global_auc:.3f}  rejected={rejected}"
            )
    return global_w, logs


def train_local_only(site: Site, epochs: int = 30) -> Dict[str, float]:
    """Baseline: the site trains alone and never talks to anyone."""
    m = GCNAutoencoder(site.model.cfg)
    m.fit(site.train, epochs=epochs)
    return evaluate(m, site.test)


def personalise(site: Site, global_weights: List[np.ndarray], epochs: int = 5) -> Dict[str, float]:
    """
    FedAvg + local fine-tuning.

    Plain FedAvg drags a site with an unusual topology toward the federation
    mean - which is exactly the non-IID failure the project set out to handle.
    Starting from the global weights and taking a few local steps keeps the
    transfer from other sites while re-fitting the site's own graph shape. No
    extra communication: the fine-tune happens after the last round.
    """
    m = GCNAutoencoder(site.model.cfg)
    m.set_weights(global_weights)
    m.fit(site.train, epochs=epochs)
    return evaluate(m, site.test)


def train_centralised(sites: Sequence[Site], epochs: int = 30) -> Dict[str, Dict[str, float]]:
    """Upper bound: all raw graphs pooled in one place (what we are NOT allowed to do)."""
    m = GCNAutoencoder(sites[0].model.cfg)
    pooled = [s for site in sites for s in site.train]
    m.fit(pooled, epochs=epochs)
    return {site.name: evaluate(m, site.test) for site in sites}
