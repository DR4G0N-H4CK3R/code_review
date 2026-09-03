"""
Adversary-bound sweeps.

  Fig 1  ASR vs m/M, checking the safety condition theta(r) > m/M (S7.2).
  Fig 2  ASR vs sensors owned inside one class, grouping on vs off,
         on a vendor-heavy deployment.
  Fig 3  ASR and FDR vs theta.

Run:  python -m experiments.run_sweep
Writes results/sweep_*.csv and results/sweep_*.png
"""
from __future__ import annotations

import csv as csvmod
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attacks import Adversary, T1_transducer_spoofing
from csvguard import (
    ActionProposal,
    Claim,
    GuardrailEngine,
    SmartHome,
    VerificationLayer,
    VerifierConfig,
)

CLAIM = Claim("occupancy", "living_room", True)
ACTION = "UNLOCK_EXTERIOR_DOOR"
TRIALS = 40
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def trial(
    *,
    seed: int,
    k: int,
    m: int,
    mode: str,
    cfg: VerifierConfig,
    occupied: bool = False,
    vendor_heavy: int = 0,
    theta: float = None,
    jam_others: bool = True,
    offline_classes: int = 0,
    prefer_largest: bool = False,
    min_classes: int = None,
) -> bool:
    home = SmartHome(seed=seed)
    if vendor_heavy:
        home.commission_vendor_heavy(vendor_heavy)
    else:
        home.commission_default()
    home.state.occupancy["living_room"] = occupied
    home.tick(duration_s=40.0)

    if not occupied:
        adv = Adversary(home, k=k, m=m, seed=seed + 7)
        adv.select_targets(CLAIM.predicate, CLAIM.target, prefer_largest=prefer_largest)
        T1_transducer_spoofing(home, adv, CLAIM)
        if jam_others:
            # Attacker denies corroboration first (T5) so no honest class can
            # contradict, then forges. C(p) is then exactly m/M, which is the
            # setting the safety condition is stated for.
            adv.silence_others(CLAIM)
    elif offline_classes:
        # benign but degraded: some classes offline, T16-style attrition with
        # no attacker. This is where false denials come from.
        classes = sorted({(sp.modality, sp.vendor, sp.transport) for sp in home.sensors
                          if sp.predicate == CLAIM.predicate})[:offline_classes]
        dead = {sp.sensor_id for sp in home.sensors
                if (sp.modality, sp.vendor, sp.transport) in classes}
        home.bus.drop_where(lambda e: e.sensor_id in dead)

    guard = GuardrailEngine(
        VerificationLayer(home, cfg),
        mode=mode,
        theta_override=theta,
        min_classes_override=min_classes,
    )
    return guard.evaluate(ActionProposal(ACTION, CLAIM)).allowed


def rate(**kw) -> float:
    hits = sum(trial(seed=1000 + t, **kw) for t in range(TRIALS))
    return hits / TRIALS


# --------------------------------------------------------------------------
def fig1_class_span() -> None:
    """ASR as the attacker spans more independence classes."""
    M = 4
    xs = list(range(0, M + 1))
    series: Dict[str, List[float]] = {}
    for label, mode in (
        ("B1 permission-only", "permission"),
        ("B3 corroboration only", "corroboration"),
        ("B6 ours (+ predicate-bound probe)", "full"),
    ):
        series[label] = [
            rate(k=8, m=m, mode=mode, cfg=VerifierConfig()) for m in xs
        ]

    plt.figure(figsize=(7.2, 4.4))
    for label, ys in series.items():
        plt.plot([m / M for m in xs], ys, marker="o", label=label)
    plt.axvline(0.85, ls="--", c="grey")
    plt.text(0.855, 0.55, "theta(tier 3) = 0.85", rotation=90, fontsize=8, color="grey")
    plt.xlabel("fraction of independence classes controlled by the attacker  (m / M)")
    plt.ylabel("attack success rate")
    plt.title("Fig 1 - ASR vs adversary class span (tier-3 unlock, 40 trials/point)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sweep_class_span.png"), dpi=160)
    plt.close()

    with open(os.path.join(RESULTS, "sweep_class_span.csv"), "w", newline="") as fh:
        w = csvmod.writer(fh)
        w.writerow(["m", "m_over_M"] + list(series))
        for i, m in enumerate(xs):
            w.writerow([m, m / M] + [series[k][i] for k in series])

    print("Fig 1  ASR vs m/M")
    print(f"{'m/M':>6}" + "".join(f"{k.split()[0]:>10}" for k in series))
    for i, m in enumerate(xs):
        print(f"{m / M:>6.2f}" + "".join(f"{series[k][i]:>10.2f}" for k in series))
    print()


def fig2_independence_cap() -> None:
    """One vendor, many devices: does the cap stop a majority takeover?"""
    counts = [1, 2, 3, 4, 5, 6]
    with_cap, without_cap = [], []
    for n in counts:
        with_cap.append(
            rate(k=n, m=1, mode="corroboration", cfg=VerifierConfig(),
                 vendor_heavy=6, prefer_largest=True)
        )
        without_cap.append(
            rate(
                k=n,
                m=1,
                mode="corroboration",
                cfg=VerifierConfig(use_independence_grouping=False),
                vendor_heavy=6,
                prefer_largest=True,
            )
        )

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(counts, without_cap, marker="s", label="naive per-sensor voting (no grouping)")
    plt.plot(counts, with_cap, marker="o", label="independence-capped corroboration")
    plt.xlabel("sensors the attacker owns inside ONE vendor/modality/transport class")
    plt.ylabel("attack success rate")
    plt.title("Fig 2 - why one class is capped (vendor-heavy home, 9 sensors)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sweep_independence_cap.png"), dpi=160)
    plt.close()

    with open(os.path.join(RESULTS, "sweep_independence_cap.csv"), "w", newline="") as fh:
        w = csvmod.writer(fh)
        w.writerow(["sensors_in_class", "no_grouping_asr", "capped_asr"])
        for i, n in enumerate(counts):
            w.writerow([n, without_cap[i], with_cap[i]])

    print("Fig 2  one-class mass compromise")
    print(f"{'k in class':>11}{'no grouping':>14}{'capped':>10}")
    for i, n in enumerate(counts):
        print(f"{n:>11}{without_cap[i]:>14.2f}{with_cap[i]:>10.2f}")
    print()


def fig3_threshold_tradeoff() -> None:
    """Security vs usability as theta moves."""
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    asr, fdr = [], []
    for th in thetas:
        asr.append(rate(k=8, m=2, mode="corroboration", cfg=VerifierConfig(),
                        theta=th, min_classes=2))
        fdr.append(
            1.0
            - sum(
                trial(seed=3000 + t, k=0, m=0, mode="corroboration",
                      cfg=VerifierConfig(), occupied=True, theta=th,
                      offline_classes=1, min_classes=2)
                for t in range(TRIALS)
            )
            / TRIALS
        )

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(thetas, asr, marker="o", label="attack success rate (m=2 of M=4)")
    plt.plot(thetas, fdr, marker="s", label="false denial rate (benign)")
    plt.axvline(0.5, ls="--", c="grey")
    plt.text(0.505, 0.6, "m/M = 0.50", rotation=90, fontsize=8, color="grey")
    plt.xlabel("corroboration threshold theta")
    plt.ylabel("rate")
    plt.title("Fig 3 - security / usability knee\n(probe disabled to isolate theta; benign home has 1 of 4 classes offline)", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sweep_threshold.png"), dpi=160)
    plt.close()

    with open(os.path.join(RESULTS, "sweep_threshold.csv"), "w", newline="") as fh:
        w = csvmod.writer(fh)
        w.writerow(["theta", "asr_m2", "fdr_benign"])
        for i, th in enumerate(thetas):
            w.writerow([th, asr[i], fdr[i]])

    print("Fig 3  threshold sweep")
    print(f"{'theta':>7}{'ASR':>8}{'FDR':>8}")
    for i, th in enumerate(thetas):
        print(f"{th:>7.2f}{asr[i]:>8.2f}{fdr[i]:>8.2f}")
    print()


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    fig1_class_span()
    fig2_independence_cap()
    fig3_threshold_tradeoff()
    print(f"figures and CSVs written to {RESULTS}")
