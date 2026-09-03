# Trustworthy IoT Security: Physical-State Verification for Agents and Federated Graph Learning for Relationship Anomalies

Bhaskar SL (AM.SC.U4CYS23017) · Arjun S (AM.SC.U4CYS23012) · Guide: Dr. Kurunandan Jain

Two parallel lines of work on the same problem — a security decision in a connected home or IoT network is only as good as the evidence it rests on, and that evidence is produced by cheap, heterogeneous, individually untrustworthy devices. Idea 1 attacks the problem at the point of action, asking whether an agent's belief about the physical world can be corroborated before it is allowed to act. Idea 2 attacks it at the point of observation, asking whether the relationships between devices can be learned across homes without any of them sharing traffic. Both are at the week-7 code drop stage; both run on a stock Python 3.9+ install, with heavier dependencies confined to figures and to the real federation.

---

# Idea 1 — Cross-Sensor Physical State Verification for Smart-Home LLM Agents

LLM agents with home-automation tool access are vulnerable to indirect prompt injection: a poisoned calendar entry, device description or message body can convince the agent that the house is in a state it is not in, and the agent then takes a physically consequential action such as unlocking an exterior door. Permission systems do not help, because the action is one the agent is legitimately allowed to perform — the failure is in the premise, not the privilege. Sensor readings could ground the premise, but a single sensor is trivially spoofed, and naive voting across sensors is defeated by any adversary who owns several devices from the same vendor. What is required is a corroboration procedure whose score an attacker cannot raise beyond a bound set by how much of the home they actually control.

## Objectives

1. To formalise the physically consequential action of a home agent as a verifiable claim over the physical state, with risk tiers determining the evidence burden
2. To design a corroboration score that is signed, freshness-gated and grouped by independence class, so that the influence of any single vendor, modality or placement is capped
3. To add a predicate-bound active challenge that closes challenge-laundering attacks which a liveness-only probe cannot detect
4. To evaluate the guardrail against a parameterised adversary across a threat catalogue, and to establish the operating window of the tier thresholds empirically

## Evidence Collection and Corroboration

1. **Physical World and Sensor Population:** A ground-truth physical state is simulated and observed by a population of heterogeneous sensors, each carrying provenance — vendor, modality and placement. The adversary manipulates this world rather than being handed a score, which is what makes the guardrail falsifiable.
2. **Claim Extraction and Risk Tiering:** The agent's intended action is reduced to a claim `p` about the physical state. The configured risk tier fixes the corroboration threshold `theta(r)` and the minimum number of independence classes that must contribute.
3. **Evidence Filtering:** Each evidence item passes a freshness gate and a continuity gate before it can contribute, discounting stale readings and readings that jump discontinuously from the device's own history.
4. **Independence Grouping and Scoring:** Per-item support is computed as `s_i = sign_i · trust_i · freshness_i · continuity_i`, summed inside each independence class and clamped to `±cap`, then normalised: `C(p) = max(0, Σ_j G_j) / (cap · M)`. Contradicting evidence carries a negative sign, so a lie must overcome the truth rather than merely add to it, and the class cap prevents one vendor with six devices from outvoting the house. An attacker owning `m` of `M` classes can raise `C` by at most `m/M`.
5. **Predicate-Bound Challenge:** Tier 3 additionally issues an active probe whose ultrasonic or IR return energy is a function of the predicate under test. An honest answer given while the attacker is silent still reports the real state and contradicts the spoofed claim, which is what closes T17 challenge laundering.

## Decision Engine and Adversarial Evaluation

1. **Decision Engine:** The guardrail applies freshness, continuity, independence grouping, corroboration and probe verdict in the §7.6 order and returns allow, deny or escalate for the requested action.
2. **Baseline Ladder:** Five comparison configurations are implemented — B0 no defence, B1 permission-only, B3 corroboration, B5 corroboration plus liveness probe, and B6 the full defence — so that each mechanism's marginal contribution is separable.
3. **Adversary Model:** The attacker is parameterised by `k`, the number of devices compromised, and `m`, the number of independence classes spanned. Nineteen threats T1–T19 are instantiated against this model, including replay, vendor collusion, threshold gaming and probe decoy.
4. **Attack Harness and Ablation:** Every threat is run against every defence over 30 trials per cell, together with an ablation that removes the predicate-bound probe and then substitutes a liveness-only probe, isolating the probe's contribution from corroboration's.
5. **Adversary-Bound Sweeps:** Attack success rate is swept over `m/M`, over vendor concentration in a brand-heavy home, and over `theta`, producing the three figures that locate the operating window and test the `m/M` bound empirically.

## Running the code

```bash
python -m tests.test_core                  # 11 invariant tests
python -m experiments.run_demo             # 4 walkthrough scenarios
python -m experiments.run_attack_harness   # 19 threats x 5 defences + ablation
python -m experiments.run_sweep            # adversary-bound sweeps -> 3 figures
```

| path | what it is |
| --- | --- |
| `csvguard/config.py` | risk tiers, per-tier thresholds `theta(r)`, class requirements, adversary bound |
| `csvguard/evidence.py` | `Claim`, `Evidence` with provenance, evidence collector |
| `csvguard/home.py` | ground-truth physical state, sensor population, probe channel |
| `csvguard/verifier.py` | freshness → continuity → independence grouping → corroboration → probes |
| `csvguard/guardrail.py` | decision engine, baselines B0/B1/B3/B5/B6, mock LLM agent |
| `attacks.py` | adversary model and threats T1–T19 |
| `experiments/`, `results/` | demo, harness, sweeps; generated CSVs, figures, transcripts |

## Results

Attack success rate, 30 trials per cell, tier-3 exterior door unlock while the house is empty:

| adversary | B0 none | B1 permission-only | B3 corroboration | B5 + liveness probe | B6 ours |
| --- | --- | --- | --- | --- | --- |
| in-bound k=2, m=2 | 1.00 | 0.16 | 0.00 | 0.00 | 0.00 |
| worst case k=8, m=M | 1.00 | 0.79 | 0.68 | 0.68 | 0.05 |

False denial rate on benign traffic is 0.00 for every configuration. The p95 decision latency is 0.10 ms for the full defence against 0.02 ms for permission-only. Ablation at k=8, m=M gives mean ASR over all 19 threats of 0.05 for the full defence, 0.68 without the predicate-bound probe and 0.74 with a liveness probe substituted. The residual 0.05 is T18 probe decoy, an absorber or thermal mass that forges the return.

## Honest limitations

- The physics is modelled, not measured. Phase 6 must replace `home.py`'s `answer_predicate_probe` with real ultrasonic and IR returns from a Raspberry Pi hub with Zigbee/Z-Wave radios and ESP32 emitters. Until then the probe's separation between occupied and empty (0.62 vs 0.18) is an assumption, and validating it is the single most important next task.
- T15 threshold gaming is not solved; it is a reported residual risk.
- Sensor error is i.i.d. in the simulator. Correlated environmental failure such as a power cut or Wi-Fi outage would look like an attack and needs its own study.

---

# Idea 2 — Lightweight Federated GNN for IoT Device-Relationship Anomaly Detection

Per-flow and per-device intrusion detection misses the class of IoT compromise that shows up only in the relationships between devices — a camera that begins talking to a thermostat, a lateral-movement path that is unremarkable flow by flow. Modelling those relationships as a graph is the natural fit, but the data is exactly the data that cannot leave the home, the labels do not exist at topology level, and the hardware is a Raspberry Pi rather than a GPU. Federated learning answers the privacy constraint but introduces two of its own: the device mix at every site differs, so a single averaged model fits nobody in particular, and a single compromised participant can poison the global model.

## Objectives

1. To construct device-relationship graphs from raw flow records with stable node identity, and to detect anomalies at the granularity of the relationship rather than the device
2. To design an unsupervised graph autoencoder small enough for edge deployment, trained without topology-level labels
3. To extend the model to a federated setting across non-IID sites, with personalisation and differential privacy quantified rather than assumed
4. To evaluate robustness against malicious clients and graph poisoning under Byzantine-tolerant aggregation

## Graph Construction and Local Model Training

1. **Data Collection and Processing:** Flow records are ingested through column-mapped adapters for CICIoT2023 and IoT-23, plus a synthetic flow generator modelled on device roles. If a header does not carry the required fields the loader names the missing ones instead of silently building garbage graphs.
2. **Graph Construction:** Flows are windowed into graphs, node identity is maintained across windows with ageing, and 8 node features and 8 edge features are computed per window.
3. **Unsupervised Model Training:** A 2-layer GCN autoencoder (8→16→2) is trained to reconstruct normal relationships. Both datasets label per flow and give no topology-level ground truth, so a supervised classifier had nothing to train against; the model instead scores what it cannot reconstruct.

   ```
   H1 = ReLU( Â X W1 + b1 )        encoder layer 1
   Z  =       Â H1 W2 + b2         latent node embeddings, dim 2
   X̂  = Z W3 + b3                  node feature decoder
   Â  = σ( Z Zᵀ )                  structural decoder
   Ê  = [z_u ‖ z_v] W4 + b4        edge feature decoder
   ```

   242 parameters, 0.9 KB per federated update. Gaussian feature noise during training is the mimicry hardening carried over from week 6. The edge decoder — reconstructing each edge's own profile from its two endpoint embeddings — is what makes the relationship the unit of detection, and it lifted pooled AUC from 0.74 to 0.87.
4. **Scoring and Attribution:** Each edge receives `score = 0.10·structural + 0.35·node_feature + 0.35·edge_feature + 0.20·peer_novelty`, with every term rank-normalised inside its window so no per-site threshold calibration is needed. Alerts carry their driver — never-seen peer, endpoint behaviour change, unusual traffic profile for this pair — and temporal consensus confirms findings across windows. Sweeping the weights over four settings moves pooled AUC only between 0.854 and 0.870.

## Federated Deployment and Robustness

1. **Non-IID Site Construction:** Six sites with deliberately different device mixes (home, office, campus, clinic, retail, flat) are constructed to reproduce the heterogeneity that a real deployment would face.
2. **Federated Training:** The federated loop runs 15 rounds of 3 local epochs under FedAvg, with `flower_client.py` / `flower_server.py` providing the real gRPC federation over PyTorch Geometric for the documented stack.
3. **Personalisation:** Each site takes a short local fine-tune starting from the global weights, at no additional communication cost, re-fitting the shared model to its own topology.
4. **Differential Privacy:** Local updates are clipped to L2 = 1.0 and Gaussian noise is added before transmission, with the utility cost measured across noise multipliers.
5. **Byzantine Defences:** The server aggregates under median, trimmed mean or Krum in place of FedAvg, with cheap norm validation rejecting any update whose L2 exceeds 3.0.
6. **Attack Evaluation:** Sign-flipping, scaling and targeted graph poisoning are run against 1–3 malicious sites of 6, and honest-site AUC is reported under each aggregation rule.

## Running the code

```bash
python -m tests.test_core            # 17 tests, includes a finite-difference gradient check
python -m experiments.run_local      # one site, flows -> graphs -> model -> alerts
python -m experiments.run_federated  # local vs FedAvg vs fine-tune vs centralised + DP sweep
python -m experiments.run_attacks    # malicious clients, robust aggregation, graph poisoning

python -m experiments.run_local --source ciciot2023 --data /path/to/CICIoT2023.csv
python -m experiments.run_local --source iot23 --data /path/to/conn.csv --limit 500000
```

| path | what it is |
| --- | --- |
| `fedgnn/datasets.py` | CICIoT2023 / IoT-23 adapters, synthetic flow generator |
| `fedgnn/graphs.py` | windowing, node identity with ageing, node and edge features |
| `fedgnn/model_numpy.py` | 2-layer GCN autoencoder with hand-written gradients |
| `fedgnn/model_torch.py` | the same model in PyTorch Geometric |
| `fedgnn/scoring.py` | per-edge score, attribution, temporal consensus, metrics |
| `fedgnn/aggregation.py` | FedAvg, median, trimmed mean, Krum, clipping and DP noise |
| `fedgnn/federated.py` | non-IID site construction, federated loop, personalisation |
| `flower_client.py`, `flower_server.py` | the real gRPC federation |

## Results

Single site: ROC-AUC 0.926, precision@10 0.90 on 663 test edges with 25 malicious.

Six non-IID sites, 15 rounds × 3 local epochs, 19 s total:

| site | local only | federated | fed + fine-tune | centralised |
| --- | --- | --- | --- | --- |
| home-a | 0.981 | 0.737 | 0.986 | 0.824 |
| office-b | 0.830 | 0.837 | 0.828 | 0.802 |
| campus-c | 0.811 | 0.849 | 0.812 | 0.827 |
| clinic-d | 0.996 | 0.830 | 0.998 | 0.895 |
| retail-e | 0.733 | 0.867 | 0.838 | 0.862 |
| flat-f | 0.746 | 0.856 | 0.928 | 0.873 |
| mean | 0.850 | 0.829 | 0.898 | 0.847 |

Plain FedAvg is not a free win: its mean sits below local-only, lifting the three weakest sites by +0.04 to +0.13 and dragging the two strongest down by −0.17 and −0.24. Federation plus a short local fine-tune fixes it, beating local-only, plain FedAvg and the centralised upper bound, and winning or tying at every site. Sharing is worth doing; sharing without re-fitting to your own topology is not.

Privacy cost — mean AUC goes 0.829 → 0.823 → 0.804 → 0.757 → 0.625 at noise multipliers 0, 0.01, 0.05, 0.1, 0.25. Anything up to 0.05 is roughly free.

Robustness, mean AUC on honest sites:

| malicious sites (of 6) | FedAvg | median | trimmed mean | Krum |
| --- | --- | --- | --- | --- |
| 0 | 0.834 | 0.824 | 0.839 | 0.881 |
| 1 sign-flip | 0.393 | 0.771 | 0.772 | 0.733 |
| 2 sign-flip | 0.442 | 0.761 | 0.761 | 0.761 |
| 3 sign-flip | 0.335 | 0.418 | 0.418 | 0.418 |

One compromised site destroys plain FedAvg. Robust aggregation recovers most of it up to a minority of attackers and collapses at 50%, as the theory predicts. Norm validation neutralises scaling and sign-flipping completely, but only because these attackers are loud; a patient attacker staying inside the norm bound is not covered and is the obvious next experiment. Targeted graph poisoning did not work — AUC stayed at 0.83–0.85 for up to 3 poisoned sites — for two structural reasons: the model has ~240 parameters and cannot memorise one site's injected pattern, and the peer-novelty term is computed locally from each site's own history and never learned, so poisoning another site's graphs cannot reach it.

## Honest limitations

- All numbers above are on synthetic flows. The adapters are written and column-mapped, but until CICIoT2023 and IoT-23 have actually been pushed through, treat every AUC here as a pipeline check rather than a detection result.
- The structural term is weak (~0.49 AUC alone). At latent dim 2 the inner-product decoder mostly encodes degree, and the malicious edge is already in the adjacency when it is scored. Held-out edge scoring would fix it at O(E) forward passes per window — too slow for a Pi as written, so it needs a batched trick.
- No CAGA yet. The architecture search from the design doc is not implemented; hidden dim, layers and score weights are all fixed by hand.
- 24 devices per site is a home, not a campus. Behaviour at 500+ nodes per window is untested.
