"""
Unit tests for the graph, model, scoring and aggregation layers.

Run:  python -m tests.test_core
      pytest tests/
"""
from __future__ import annotations

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
    build_sites,
    evaluate,
    load_source,
    run_federated,
)
from fedgnn.aggregation import aggregate, clip_and_noise, flatten, validate_update
from fedgnn.graphs import normalised_adjacency
from fedgnn.scoring import roc_auc, score_snapshot, temporal_consensus


def _snaps(n_devices=20, seed=3):
    flows = load_source("synthetic", site_id=0, seed=seed, n_devices=n_devices,
                        duration_s=1800.0, attack_rate=0.04)
    return GraphBuilder(window_s=60.0).build_all(flows)


# ------------------------------------------------------------------- graphs
def test_graph_shapes():
    snaps = _snaps()
    assert len(snaps) > 5
    for s in snaps:
        assert s.X.shape == (s.n_nodes, len(NODE_FEATURES))
        assert s.edge_attr.shape == (s.n_edges, len(EDGE_FEATURES))
        assert s.edge_index.shape == (2, s.n_edges)
        assert s.A.shape == (s.n_nodes, s.n_nodes)
        assert s.edge_index.max(initial=0) < max(s.n_nodes, 1)


def test_adjacency_is_symmetric_and_normalised():
    ei = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    A = normalised_adjacency(ei, 3)
    assert np.allclose(A, A.T)
    assert np.all(A >= 0) and np.all(A <= 1)


def test_no_future_leakage_in_novelty():
    """peer_novelty must be computed against PAST windows only."""
    snaps = _snaps()
    nov_col = EDGE_FEATURES.index("peer_novelty")
    first = snaps[0].edge_attr[:, nov_col]
    assert first.mean() == 1.0, "every peer in the first window is new by definition"
    later = np.concatenate([s.edge_attr[:, nov_col] for s in snaps[3:]])
    assert later.mean() < 0.5, "novelty should decay as the history fills in"


def test_features_are_finite():
    for s in _snaps():
        assert np.all(np.isfinite(s.X))
        assert np.all(np.isfinite(s.edge_attr))


# -------------------------------------------------------------------- model
def test_training_reduces_loss():
    snaps = _snaps()
    m = GCNAutoencoder(GCNConfig(seed=0))
    hist = m.fit(snaps[:10], epochs=25)
    assert hist[-1] < hist[0] * 0.6, (hist[0], hist[-1])


def test_gradients_match_finite_differences():
    """Spot-check the hand-written backward pass against numerical gradients."""
    snaps = _snaps()
    s = next(x for x in snaps if x.n_edges > 3)
    cfg = GCNConfig(seed=0, noise_std=0.0, lambda_struct=0.0)   # struct term is stochastic
    m = GCNAutoencoder(cfg)

    def loss_only(model):
        f = model.forward(s.X, s.A, training=False)
        l = float(np.mean((f["Xhat"] - s.X) ** 2)) * cfg.lambda_feat
        Ehat = model.decode_edges(f["Z"], s.edge_index)
        return l + float(np.mean((Ehat - s.edge_attr) ** 2)) * cfg.lambda_edge

    before = {k: v.copy() for k, v in m.params.items()}
    m.train_step(s.X, s.A, s.edge_index, s.edge_attr)
    step = {k: m.params[k] - before[k] for k in before}

    # Adam's first step is lr * sign(g); check the sign agrees with a numerical grad
    m.params = {k: v.copy() for k, v in before.items()}
    eps = 1e-4
    checked = 0
    for key in ("W1", "W2", "W3", "W4"):
        for _ in range(3):
            i = np.random.randint(m.params[key].shape[0])
            j = np.random.randint(m.params[key].shape[1])
            m.params[key][i, j] = before[key][i, j] + eps
            hi = loss_only(m)
            m.params[key][i, j] = before[key][i, j] - eps
            lo = loss_only(m)
            m.params[key][i, j] = before[key][i, j]
            num_grad = (hi - lo) / (2 * eps)
            if abs(num_grad) < 1e-7:
                continue
            assert np.sign(step[key][i, j]) == -np.sign(num_grad), (
                key, i, j, num_grad, step[key][i, j]
            )
            checked += 1
    assert checked >= 4, f"only {checked} coordinates were informative"


def test_weight_roundtrip():
    m = GCNAutoencoder(GCNConfig(seed=0))
    w = m.get_weights()
    m2 = GCNAutoencoder(GCNConfig(seed=7))
    m2.set_weights(w)
    for a, b in zip(w, m2.get_weights()):
        assert np.allclose(a, b)


def test_model_is_lightweight():
    """The whole point is that it fits on a Pi: keep the update under 10 KB."""
    m = GCNAutoencoder(GCNConfig())
    assert m.n_params() < 2500
    assert m.n_params() * 4 < 10 * 1024


# ------------------------------------------------------------------ scoring
def test_scores_in_range():
    snaps = _snaps()
    m = GCNAutoencoder(GCNConfig(seed=0))
    m.fit(snaps[:10], epochs=10)
    for s in snaps[10:]:
        sc = score_snapshot(m, s)
        assert sc.shape == (s.n_edges,)
        assert np.all(sc >= -1e-6) and np.all(sc <= 1 + 1e-6)


def test_roc_auc_matches_known_values():
    assert abs(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])) - 0.0) < 1e-9
    assert abs(roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-9
    assert roc_auc(np.array([0, 0]), np.array([0.1, 0.2])) == 0.5


def test_detection_beats_chance():
    sites = build_sites(n_sites=2, seed=1, duration_s=2400.0, window_s=60.0)
    m = GCNAutoencoder(GCNConfig(seed=0))
    m.fit(sites[0].train, epochs=30)
    assert evaluate(m, sites[0].test)["auc"] > 0.65


def test_temporal_consensus_needs_repetition():
    snaps = _snaps()
    m = GCNAutoencoder(GCNConfig(seed=0))
    m.fit(snaps[:10], epochs=10)
    scored = [(s, score_snapshot(m, s)) for s in snaps[10:] if s.n_edges]
    strict = temporal_consensus(scored, k_of_n=(3, 3), threshold=0.95)
    loose = temporal_consensus(scored, k_of_n=(1, 3), threshold=0.5)
    assert len(strict) <= len(loose)


# -------------------------------------------------------------- aggregation
def test_clip_bounds_the_norm():
    rng = np.random.default_rng(0)
    upd = [rng.normal(size=(8, 16)).astype(np.float32), rng.normal(size=16).astype(np.float32)]
    out = clip_and_noise(upd, clip_norm=1.0, noise_std=0.0)
    assert np.linalg.norm(flatten(out)) <= 1.0 + 1e-5


def test_validation_rejects_garbage():
    ok, _ = validate_update([np.array([np.inf], dtype=np.float32)])
    assert not ok
    ok, _ = validate_update([np.full(100, 5.0, dtype=np.float32)], max_norm=1.0)
    assert not ok
    ok, _ = validate_update([np.zeros(10, dtype=np.float32)], max_norm=1.0)
    assert ok


def test_robust_rules_resist_one_outlier():
    good = [[np.ones((4, 4), dtype=np.float32)] for _ in range(5)]
    bad = [np.full((4, 4), 500.0, dtype=np.float32)]
    updates = good + [bad]
    sizes = [10] * 6
    mean = aggregate("fedavg", updates, sizes)[0].mean()
    med = aggregate("median", updates, sizes)[0].mean()
    trim = aggregate("trimmed_mean", updates, sizes, beta=1)[0].mean()
    assert mean > 50, mean
    assert abs(med - 1.0) < 1e-5
    assert abs(trim - 1.0) < 1e-5


def test_federated_loop_runs_and_learns():
    sites = build_sites(n_sites=3, seed=1, duration_s=2400.0, window_s=60.0)
    _, logs = run_federated(sites, rounds=6, local_epochs=2, strategy="fedavg", verbose=False)
    assert len(logs) == 6
    assert logs[-1].train_loss < logs[0].train_loss
    assert logs[-1].global_auc > 0.5


def test_sites_are_non_iid():
    sites = build_sites(n_sites=4, seed=1, duration_s=2400.0, window_s=60.0)
    means = [np.mean([s.X.mean() for s in site.train]) for site in sites]
    densities = [
        np.mean([s.n_edges / max(s.n_nodes, 1) for s in site.train]) for site in sites
    ]
    assert np.std(densities) > 0.05, densities


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
