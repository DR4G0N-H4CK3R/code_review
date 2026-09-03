"""
Aggregation strategies and update-side privacy.

FedAvg is the baseline. The other two exist because the threat model lists
T4 malicious client update and T5 Sybil / free-rider sites: a single client that
sends a scaled or sign-flipped update can move a plain weighted mean anywhere it
likes, and the defence has to live in the aggregator.

  fedavg        weighted mean, weight n_i          - no robustness
  trimmed_mean  drop the beta lowest/highest per coordinate
  krum          pick the update closest to its n-f-2 nearest neighbours
  median        coordinate-wise median

`clip_and_noise` implements the client-side DP-SGD-lite step from the design
doc: bound the update norm, then add Gaussian noise. Norm clipping is doing
double duty here - it is also what makes trimmed mean and Krum effective, since
an unbounded update is the easiest attack there is.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Weights = List[np.ndarray]


# --------------------------------------------------------------------------
def flatten(ws: Weights) -> np.ndarray:
    return np.concatenate([w.ravel() for w in ws])


def unflatten(vec: np.ndarray, like: Weights) -> Weights:
    out, i = [], 0
    for w in like:
        n = w.size
        out.append(vec[i : i + n].reshape(w.shape).astype(np.float32))
        i += n
    return out


def clip_and_noise(
    update: Weights, clip_norm: float = 1.0, noise_std: float = 0.0, rng=None
) -> Weights:
    """Client-side: bound the L2 norm of the update, then add Gaussian noise."""
    rng = rng or np.random.default_rng()
    vec = flatten(update)
    norm = float(np.linalg.norm(vec))
    if norm > clip_norm and norm > 0:
        vec = vec * (clip_norm / norm)
    if noise_std > 0:
        vec = vec + rng.normal(0.0, noise_std * clip_norm, size=vec.shape)
    return unflatten(vec, update)


def validate_update(update: Weights, max_norm: float = 10.0) -> Tuple[bool, str]:
    """
    Server-side sanity checks before an update is allowed near the aggregate.
    Cheap, and catches the unsubtle half of T4.
    """
    vec = flatten(update)
    if not np.all(np.isfinite(vec)):
        return False, "non-finite values"
    norm = float(np.linalg.norm(vec))
    if norm > max_norm:
        return False, f"norm {norm:.2f} exceeds {max_norm}"
    return True, "ok"


# --------------------------------------------------------------------------
def fedavg(updates: Sequence[Weights], sizes: Sequence[int]) -> Weights:
    total = float(sum(sizes)) or 1.0
    out = [np.zeros_like(w) for w in updates[0]]
    for u, n in zip(updates, sizes):
        for i, w in enumerate(u):
            out[i] += w * (n / total)
    return out


def coordinate_median(updates: Sequence[Weights], sizes=None) -> Weights:
    stack = np.stack([flatten(u) for u in updates])
    return unflatten(np.median(stack, axis=0), updates[0])


def trimmed_mean(updates: Sequence[Weights], sizes=None, beta: int = 1) -> Weights:
    stack = np.stack([flatten(u) for u in updates])
    n = stack.shape[0]
    beta = max(0, min(beta, (n - 1) // 2))
    if beta == 0:
        return unflatten(stack.mean(0), updates[0])
    s = np.sort(stack, axis=0)[beta : n - beta]
    return unflatten(s.mean(0), updates[0])


def krum(updates: Sequence[Weights], sizes=None, f: int = 1, multi: int = 1) -> Weights:
    """
    Blanchard et al. Krum / Multi-Krum. Needs n > 2f + 2 to be meaningful; falls
    back to the coordinate median when there are too few clients, which is the
    honest thing to do rather than pretending to be robust.
    """
    n = len(updates)
    if n <= 2 * f + 2:
        return coordinate_median(updates)
    flat = np.stack([flatten(u) for u in updates])
    d2 = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
    k = n - f - 2
    scores = np.array([np.sort(d2[i])[1 : k + 1].sum() for i in range(n)])
    chosen = np.argsort(scores)[: max(1, multi)]
    return unflatten(flat[chosen].mean(0), updates[0])


STRATEGIES = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
}


def aggregate(name: str, updates: Sequence[Weights], sizes: Sequence[int], **kw) -> Weights:
    fn = STRATEGIES.get(name)
    if fn is None:
        raise ValueError(f"unknown strategy {name!r}; have {list(STRATEGIES)}")
    if name == "fedavg":
        return fn(updates, sizes)
    return fn(updates, sizes, **kw)
