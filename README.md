# Idea 2 — Lightweight Federated GNN for IoT Device-Relationship Anomaly Detection

Bhaskar SL (AM.SC.U4CYS23017) · Arjun S (AM.SC.U4CYS23012) · Guide: Dr. Kurunandan Jain

Week-7 code drop. The core pipeline needs only NumPy; matplotlib is used for the
figures, and Flower / PyTorch Geometric only for the real gRPC federation.

```
python -m tests.test_core            # 17 tests, includes a finite-difference gradient check
python -m experiments.run_local      # one site, flows -> graphs -> model -> alerts
python -m experiments.run_federated  # local vs FedAvg vs fine-tune vs centralised + DP sweep
python -m experiments.run_attacks    # malicious clients, robust aggregation, graph poisoning
```

Against a real dataset:

```
python -m experiments.run_local --source ciciot2023 --data /path/to/CICIoT2023.csv
python -m experiments.run_local --source iot23 --data /path/to/conn.csv --limit 500000
```

## What changed since week 6

The week-6 run trained on mock random tensors, so the slide honestly said
`Loss = 0.0 (placeholder — no real data yet)`. Three things are different now:

1. **Real graphs.** Flow records go through windowing, node identity with ageing,
   and 8 node + 8 edge features. Adapters for CICIoT2023 and IoT-23 are
   column-mapped, so if a header does not carry the fields we need the loader
   says which ones are missing instead of building garbage graphs.
2. **The model became an autoencoder.** The 2-class classifier had nothing to
   train against — both datasets label per flow and give no topology-level
   ground truth. The model is now unsupervised: it learns what normal
   relationships look like and scores the ones it cannot reconstruct.
3. **An edge decoder was added.** Reconstructing each edge's own profile from
   the two endpoint embeddings is what makes the *relationship*, not the device,
   the unit of detection. It lifted pooled AUC from 0.74 to 0.87.

## Layout

| path | what it is |
| --- | --- |
| `fedgnn/datasets.py` | CICIoT2023 / IoT-23 adapters + synthetic flow generator |
| `fedgnn/graphs.py` | windowing, node identity with ageing, node & edge features |
| `fedgnn/model_numpy.py` | 2-layer GCN autoencoder, 8→16→2, hand-written gradients |
| `fedgnn/model_torch.py` | same model in PyTorch Geometric, for the documented stack |
| `fedgnn/scoring.py` | per-edge score, attribution, temporal consensus, metrics |
| `fedgnn/aggregation.py` | FedAvg, median, trimmed mean, Krum, clipping + DP noise |
| `fedgnn/federated.py` | non-IID site construction, federated loop, personalisation |
| `flower_client.py` / `flower_server.py` | the real gRPC federation |
| `experiments/`, `tests/`, `results/` | runnable experiments, tests, generated output |

## The model

```
H1  = ReLU( Â X W1 + b1 )            encoder layer 1
Z   =       Â H1 W2 + b2             latent node embeddings, dim 2
X̂   = Z W3 + b3                      node feature decoder
Â   = σ( Z Zᵀ )                      structural decoder
Ê   = [z_u ‖ z_v] W4 + b4            edge feature decoder
```

242 parameters, 0.9 KB per federated update. Gaussian feature noise during
training is the mimicry hardening carried over from week 6.

Per-edge score, each term rank-normalised inside its window so no per-site
threshold calibration is needed:

```
score = 0.10·structural + 0.35·node_feature + 0.35·edge_feature + 0.20·peer_novelty
```

Sweeping those weights over four settings moves pooled AUC only between 0.854
and 0.870, so nothing here is tuned into existence.

## Results

**Single site** (`run_local`): ROC-AUC 0.926, precision@10 0.90 on 663 test edges
with 25 malicious. Temporal consensus confirmed the compromised camera across
windows, and every alert carries its driver — "never-seen peer", "endpoint
behaviour change", "unusual traffic profile for this pair".

**Six non-IID sites** (`run_federated`, 15 rounds × 3 local epochs, 19 s total):

| site | local only | federated | fed + fine-tune | centralised |
| --- | --- | --- | --- | --- |
| home-a | 0.981 | 0.737 | **0.986** | 0.824 |
| office-b | 0.830 | **0.837** | 0.828 | 0.802 |
| campus-c | 0.811 | **0.849** | 0.812 | 0.827 |
| clinic-d | 0.996 | 0.830 | **0.998** | 0.895 |
| retail-e | 0.733 | **0.867** | 0.838 | 0.862 |
| flat-f | 0.746 | 0.856 | **0.928** | 0.873 |
| **mean** | 0.850 | 0.829 | **0.898** | 0.847 |

Two findings, and the second is the more interesting one:

* **Plain FedAvg is not a free win.** Its mean (0.829) is *below* local-only
  (0.850). It lifts the three weakest sites by +0.04 to +0.13 and drags the two
  strongest down by −0.17 and −0.24. That is the non-IID failure already named
  on the challenges slide, now measured rather than predicted: every site's
  device mix differs, so the average fits nobody in particular.
* **Federation plus a short local fine-tune fixes it.** Starting from the global
  weights and taking a few local steps — no extra communication — gives 0.898,
  beating local-only, plain FedAvg *and* the centralised upper bound, and winning
  or tying at every single site. Sharing is worth doing; sharing *without*
  re-fitting to your own topology is not.

**Privacy cost.** Clip to L2 = 1.0, then add Gaussian noise: mean AUC goes
0.829 → 0.823 → 0.804 → 0.757 → 0.625 at multipliers 0, 0.01, 0.05, 0.1, 0.25.
Anything up to 0.05 is roughly free.

**Attacks on the pipeline** (`run_attacks`, mean AUC on honest sites):

| malicious sites (of 6) | FedAvg | median | trimmed mean | Krum |
| --- | --- | --- | --- | --- |
| 0 | 0.834 | 0.824 | 0.839 | 0.881 |
| 1 sign-flip | **0.393** | 0.771 | 0.772 | 0.733 |
| 2 sign-flip | 0.442 | 0.761 | 0.761 | 0.761 |
| 3 sign-flip | 0.335 | 0.418 | 0.418 | 0.418 |

One compromised site is enough to destroy plain FedAvg. Robust aggregation
recovers most of it up to a minority of attackers and collapses at 50%, exactly
as the theory says it should. Cheap server-side norm validation (reject any
update whose L2 exceeds 3.0) neutralises both scaling and sign-flipping
completely — but only because these attackers are loud; a patient attacker
staying inside the norm bound is not covered and is the obvious next experiment.

**Targeted graph poisoning did not work**, and that is reported rather than
dropped: AUC stayed at 0.83–0.85 for up to 3 poisoned sites. Two structural
reasons — the model has ~240 parameters and cannot memorise one site's injected
pattern, and the peer-novelty term is computed locally from each site's own
history and never learned, so poisoning another site's graphs cannot reach it.
The second is arguably a design property worth claiming.

## Honest limitations

1. **All numbers above are on synthetic flows.** The graph structure and the
   lateral-movement injection are modelled on device roles, not captured. The
   dataset adapters are written and column-mapped, but until CICIoT2023 and
   IoT-23 have actually been pushed through, treat every AUC here as a pipeline
   check rather than a detection result.
2. **The structural term is weak** (~0.49 AUC alone). At latent dim 2 the
   inner-product decoder mostly encodes degree, and the malicious edge is already
   in the adjacency when it is scored. Scoring each candidate edge with that edge
   held out of Â would fix it at O(E) forward passes per window — too slow for a
   Pi as written, so it needs a batched trick.
3. **No CAGA yet.** The architecture search from the design doc is not
   implemented; hidden dim, layers and the score weights are all fixed by hand.
4. **Small graphs.** 24 devices per site is a home, not a campus. Behaviour at
   500+ nodes per window is untested.
