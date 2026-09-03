# Idea 1 — Cross-Sensor Physical State Verification for Smart-Home LLM Agents

Bhaskar SL (AM.SC.U4CYS23017) · Arjun S (AM.SC.U4CYS23012) · Guide: Dr. Kurunandan Jain

Week-7 code drop. Everything here runs on a stock Python 3.9+ install; only the
sweep figures need matplotlib.

```
python -m tests.test_core              # 11 invariant tests
python -m experiments.run_demo         # the 4 walkthrough scenarios
python -m experiments.run_attack_harness   # 19 threats x 5 defences + ablation
python -m experiments.run_sweep        # adversary-bound sweeps -> 3 figures
```

## What this answers

The week-6 code took the corroboration score, the risk tier and the probe verdict
as inputs from the caller. That made the guardrail unfalsifiable — it could not
be attacked, only demonstrated. This version computes all three from a simulated
physical world that an adversary can manipulate, so every claim in the
presentation is now measured rather than asserted.

## Layout

| path | what it is |
| --- | --- |
| `csvguard/config.py` | risk tiers, per-tier thresholds `theta(r)` and class requirements, adversary bound |
| `csvguard/evidence.py` | `Claim`, `Evidence` with provenance, evidence collector |
| `csvguard/home.py` | ground-truth physical state, sensor population, probe channel |
| `csvguard/verifier.py` | freshness → continuity → independence grouping → corroboration → probes |
| `csvguard/guardrail.py` | decision engine (§7.6 order), baselines B0/B1/B3/B5/B6, mock LLM agent |
| `attacks.py` | adversary model + all 19 threats T1–T19 |
| `experiments/` | demo, attack harness, adversary-bound sweeps |
| `results/` | generated CSVs, figures and console transcripts |

## The core computation

For a claim `p` over an evidence window:

```
s_i  = sign_i · trust_i · freshness_i · continuity_i        per evidence item
G_j  = clamp( Σ_{i ∈ class j} s_i , −cap , +cap )           per independence class
C(p) = max(0, Σ_j G_j) / (cap · M)                          in [0, 1]
```

* `sign_i` is −1 for contradicting evidence, so a lie has to overcome the truth,
  not merely add to it.
* the class cap is what stops one vendor with six devices from outvoting the house.
* an attacker owning `m` of `M` classes can raise `C` by at most `m/M`, so any
  `theta(r) > m/M` is unreachable. Fig 1 tests this empirically.

Tier 3 additionally requires a **predicate-bound** challenge: the ultrasonic / IR
return energy is a function of the predicate under test, so an honest answer given
while the attacker is silent still reports the real state and contradicts the
spoofed claim. That is what closes T17 challenge laundering, which a liveness-only
probe cannot.

## Results produced by this code

Attack success rate, 30 trials per cell, tier-3 exterior door unlock while the
house is empty:

| adversary | B0 none | B1 permission-only | B3 corroboration | B5 + liveness probe | B6 ours |
| --- | --- | --- | --- | --- | --- |
| in-bound `k=2, m=2` | 1.00 | 0.16 | 0.00 | 0.00 | 0.00 |
| worst case `k=8, m=M` | 1.00 | 0.79 | 0.68 | 0.68 | **0.05** |

False denial rate on benign traffic is 0.00 for every configuration; p95 decision
latency is 0.10 ms for the full defence versus 0.02 ms for permission-only.

Ablation, mean ASR over all 19 threats at `k=8, m=M`:

| configuration | ASR |
| --- | --- |
| full defence | 0.05 |
| − predicate-bound probe | 0.68 |
| − probe, liveness probe only | 0.74 |

The residual 0.05 is **T18 probe decoy** — an absorber or thermal mass that forges
the return. It is reported, not hidden; the threat model already classes it as
*bounded*, and T15 threshold gaming stays open.

## Figures

* `results/sweep_class_span.png` — ASR vs `m/M`. B3 stays at 0 until the attacker
  owns every class; B6 stays at 0 throughout because the probe is orthogonal to
  corroboration.
* `results/sweep_independence_cap.png` — vendor-heavy home. Naive per-sensor
  voting is defeated once the attacker owns 4 devices of one brand; the capped
  score never moves.
* `results/sweep_threshold.png` — the operating window. Any `theta ∈ (0.5, 0.75]`
  gives ASR 0 and FDR 0 against an `m=2` adversary, which is where `theta=0.85`
  for tier 3 sits.

## Honest limitations

1. The physics is modelled, not measured. Phase 6 (Raspberry Pi hub, Zigbee /
   Z-Wave radios, ESP32 probe emitters) has to replace `home.py`'s
   `answer_predicate_probe` with real ultrasonic and IR returns. Until then the
   probe's separation between occupied and empty (0.62 vs 0.18) is an assumption,
   and validating it is the single most important next task.
2. `T15 threshold gaming` is not solved; it is a residual risk.
3. Sensor error is i.i.d. in the simulator. Correlated environmental failure
   (a power cut, a Wi-Fi outage) would look like an attack and needs its own study.
