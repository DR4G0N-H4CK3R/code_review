"""
Runs the 19 threats against 5 defence configurations and reports:
  ASR  - attack success rate (a spoofed tier-3 unlock is allowed)
  FDR  - false denial rate on benign traffic
  p95 latency of a guardrail decision

Two adversary settings:
  in-bound   k=2, m=2  - the declared adversary model
  worst-case k=8, m=M  - attacker owns every independence class

Run:  python -m experiments.run_attack_harness
Writes results/attack_harness.csv
"""
from __future__ import annotations

import csv as csvmod
import os
import statistics
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks import THREATS, Adversary
from csvguard import (
    ActionProposal,
    Claim,
    GuardrailEngine,
    SmartHome,
    VerificationLayer,
    VerifierConfig,
)

MODES = [
    ("none", "B0 no guardrail"),
    ("permission", "B1 permission-only (AgentSpec-like)"),
    ("corroboration", "B3 corroboration only"),
    ("liveness", "B5 corroboration + liveness probe"),
    ("full", "B6 ours: + predicate-bound probe"),
]
TRIALS = 30
CLAIM = Claim("occupancy", "living_room", True)
ACTION = "UNLOCK_EXTERIOR_DOOR"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build_home(seed: int, occupied: bool) -> SmartHome:
    home = SmartHome(seed=seed).commission_default()
    home.state.occupancy["living_room"] = occupied
    home.tick(duration_s=40.0)
    return home


def run_attack(
    threat_id: str, mode: str, k: int, m: int, trials: int, cfg: VerifierConfig = None
) -> Tuple[float, List[float]]:
    """Return (attack success rate, latencies)."""
    _, fn = THREATS[threat_id]
    successes = 0
    lat: List[float] = []
    for t in range(trials):
        home = build_home(seed=100 + t, occupied=False)   # nobody home, so the claim is false
        adv = Adversary(home, k=k, m=m, seed=200 + t)
        adv.select_targets(CLAIM.predicate, CLAIM.target)
        fn(home, adv, CLAIM)
        guard = GuardrailEngine(VerificationLayer(home, cfg or VerifierConfig()), mode=mode)
        d = guard.evaluate(ActionProposal(ACTION, CLAIM))
        lat.append(d.latency_ms)
        if d.allowed:
            successes += 1
    return successes / trials, lat


ABLATIONS = [
    ("full defence", "full", VerifierConfig()),
    ("- predicate-bound probe", "corroboration", VerifierConfig()),
    ("- probe, + liveness probe only", "liveness", VerifierConfig(dropout_penalty=0.0, min_continuity=0.0)),
    ("- continuity & dropout scoring", "full", VerifierConfig(dropout_penalty=0.0, min_continuity=0.0)),
    ("- independence grouping", "full", VerifierConfig(use_independence_grouping=False)),
    ("- channel binding", "full", VerifierConfig(channel_binding=False)),
]


def run_ablation(trials: int) -> List[dict]:
    print("\n" + "=" * 96)
    print("ABLATION - mean ASR over all 19 threats when one mechanism is removed")
    print("=" * 96)
    print(f"{'configuration':<36}{'in-bound k=2,m=2':>20}{'worst-case k=8,m=M':>22}")
    print("-" * 78)
    out = []
    for label, mode, cfg in ABLATIONS:
        cells = []
        for (k, m) in ((2, 2), (8, 4)):
            vals = [run_attack(tid, mode, k, m, trials, cfg)[0] for tid in THREATS]
            cells.append(sum(vals) / len(vals))
        print(f"{label:<36}{cells[0]:>20.2f}{cells[1]:>22.2f}")
        out.append(
            {"bound": "ablation", "threat": "-", "description": label, "defence": mode,
             "asr": round(cells[1], 3), "p95_latency_ms": "", "fdr": ""}
        )
    return out


def run_benign(mode: str, trials: int) -> float:
    """False denial rate: occupant is home, no attacker."""
    denied = 0
    for t in range(trials):
        home = build_home(seed=500 + t, occupied=True)
        guard = GuardrailEngine(VerificationLayer(home, VerifierConfig()), mode=mode)
        d = guard.evaluate(ActionProposal(ACTION, CLAIM))
        if not d.allowed:
            denied += 1
    return denied / trials


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for bound_name, (k, m) in {"in-bound (k=2,m=2)": (2, 2), "worst-case (k=8,m=M)": (8, 4)}.items():
        print("\n" + "=" * 96)
        print(f"ATTACK SUCCESS RATE - adversary {bound_name},  {TRIALS} trials per cell")
        print("=" * 96)
        header = f"{'threat':<6}{'description':<36}" + "".join(f"{lbl.split()[0]:>10}" for _, lbl in MODES)
        print(header)
        print("-" * len(header))
        for tid, (desc, _) in THREATS.items():
            cells = []
            for mode, _ in MODES:
                asr, lat = run_attack(tid, mode, k, m, TRIALS)
                cells.append(asr)
                rows.append(
                    {
                        "bound": bound_name,
                        "threat": tid,
                        "description": desc,
                        "defence": mode,
                        "asr": round(asr, 3),
                        "p95_latency_ms": round(
                            statistics.quantiles(lat, n=20)[-1] if len(lat) > 1 else lat[0], 3
                        ),
                    }
                )
            print(f"{tid:<6}{desc[:34]:<36}" + "".join(f"{c:>10.2f}" for c in cells))

        print("-" * len(header))
        mean_by_mode = []
        for i, (mode, _) in enumerate(MODES):
            vals = [r["asr"] for r in rows if r["bound"] == bound_name and r["defence"] == mode]
            mean_by_mode.append(sum(vals) / len(vals))
        print(f"{'MEAN':<6}{'':<36}" + "".join(f"{v:>10.2f}" for v in mean_by_mode))

    print("\n" + "=" * 96)
    print("BENIGN TRAFFIC - false denial rate and decision latency")
    print("=" * 96)
    for mode, label in MODES:
        fdr = run_benign(mode, TRIALS)
        _, lat = run_attack("T1", mode, 2, 2, TRIALS)
        p95 = statistics.quantiles(lat, n=20)[-1] if len(lat) > 1 else lat[0]
        print(f"{label:<40} FDR={fdr:5.2f}   p95 latency={p95:7.3f} ms")
        rows.append(
            {"bound": "benign", "threat": "-", "description": "benign tier-3 unlock",
             "defence": mode, "asr": "", "p95_latency_ms": round(p95, 3), "fdr": round(fdr, 3)}
        )

    rows.extend(run_ablation(TRIALS))

    out = os.path.join(RESULTS_DIR, "attack_harness.csv")
    with open(out, "w", newline="") as fh:
        w = csvmod.DictWriter(
            fh, fieldnames=["bound", "threat", "description", "defence", "asr", "p95_latency_ms", "fdr"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
