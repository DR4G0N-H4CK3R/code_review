"""
Attack harness (Phase 4).

Instantiates each threat from the rev.4 threat model inside the simulator so
the guardrail can be measured against it.

Adversary model: the attacker controls at most k of N sensors, spanning at
most m of M independence classes, and may emit into the probe volume.

Compromising a sensor means owning its whole stream for the window, not
appending one message. take_over() removes the honest readings of the
controlled sensors and replays the attacker's version at the device's own
cadence, which is what a firmware implant, RF injector or stolen cloud token
actually gives you.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from csvguard.evidence import Claim, Evidence
from csvguard.home import SensorSpec, SmartHome

WINDOW_S = 30.0


@dataclass
class Adversary:
    home: SmartHome
    k: int = 1                # sensors controlled
    m: int = 1                # independence classes spanned
    seed: int = 0
    controlled: List[SensorSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_targets(
        self, predicate: str, target: str, prefer_largest: bool = False
    ) -> List[SensorSpec]:
        """
        Pick k sensors spanning at most m independence classes.
        prefer_largest models an attacker going after the vendor that sold the
        most devices in the house.
        """
        pool = [s for s in self.home.sensors if s.predicate == predicate and s.target == target]
        by_class: Dict[Tuple[str, str, str], List[SensorSpec]] = {}
        for s in pool:
            by_class.setdefault((s.modality, s.vendor, s.transport), []).append(s)

        classes = sorted(by_class.keys())
        if prefer_largest:
            classes.sort(key=lambda c: -len(by_class[c]))
        else:
            self.rng.shuffle(classes)
        chosen: List[SensorSpec] = []
        for cls in classes[: max(0, self.m)]:
            for s in by_class[cls]:
                if len(chosen) < self.k:
                    chosen.append(s)
        self.controlled = chosen
        return chosen

    def _expected(self, spec: SensorSpec) -> int:
        return max(1, int(WINDOW_S / spec.period_s))

    def emit(
        self,
        spec: SensorSpec,
        claim: Claim,
        *,
        age: float = 0.0,
        reports: Optional[int] = None,
        calibration_ok: bool = True,
        trust_prior: Optional[float] = None,
        channel_id: Optional[str] = None,
        value: Any = None,
    ) -> Evidence:
        """Publish one attacker-controlled reading onto the bus."""
        ev = Evidence(
            sensor_id=spec.sensor_id,
            vendor=spec.vendor,
            modality=spec.modality,
            transport=spec.transport,
            placement=spec.placement,
            predicate=claim.predicate,
            target=claim.target,
            value=claim.value if value is None else value,
            timestamp=self.home.now - age,
            trust_prior=spec.trust_prior if trust_prior is None else trust_prior,
            calibration_ok=calibration_ok,
            reports_in_window=self._expected(spec) if reports is None else reports,
            expected_in_window=self._expected(spec),
            channel_id=channel_id,
        )
        self.home.bus.publish(ev)
        return ev

    def silence(self, spec: SensorSpec, claim: Claim) -> None:
        """Remove one device's readings for this predicate from the bus."""
        self.home.bus.drop_where(
            lambda e: e.sensor_id == spec.sensor_id
            and e.predicate == claim.predicate
            and e.target == claim.target
        )

    def take_over(
        self,
        claim: Claim,
        *,
        n_reports: Optional[int] = None,
        age_offset: float = 0.0,
        **kw,
    ) -> None:
        """
        Replace the honest stream of every controlled sensor with the
        attacker's assertion, at the device's own cadence.

        n_reports=1 is the hit-and-run case (T17): assert once, then go silent
        so an honest sensor answers the challenge.
        """
        for spec in self.controlled:
            self.silence(spec, claim)
            total = self._expected(spec) if n_reports is None else n_reports
            for i in range(total):
                self.emit(spec, claim, age=age_offset + i * spec.period_s, reports=total, **kw)

    def silence_others(self, claim: Claim) -> None:
        """Jam / drain every sensor the attacker does NOT control."""
        ids = {s.sensor_id for s in self.controlled}
        self.home.bus.drop_where(
            lambda e: e.sensor_id not in ids
            and e.predicate == claim.predicate
            and e.target == claim.target
        )


# Threats. Each takes (home, adv, claim) and mutates the world or the bus.
# The docstring gives the trust boundary and what is meant to stop it.
def T1_transducer_spoofing(home, adv, claim):
    """TB1. Lie to the physics before any firmware, e.g. IR into a PIR."""
    adv.take_over(claim)


def T2_sensor_masking(home, adv, claim):
    """TB1. Shrink the evidence set by hiding sensors that would contradict."""
    adv.take_over(claim)


def T3_rf_replay(home, adv, claim):
    """TB2. Replay a captured Zigbee / Z-Wave frame. Freshness should kill it."""
    adv.take_over(claim, age_offset=90.0)


def T4_firmware_compromise(home, adv, claim):
    """TB2. One vendor CVE collapses an entire independence class."""
    pool = [s for s in home.sensors if s.predicate == claim.predicate and s.target == claim.target]
    vendor = adv.controlled[0].vendor if adv.controlled else pool[0].vendor
    adv.controlled = [s for s in pool if s.vendor == vendor]
    adv.take_over(claim)


def T5_selective_jamming(home, adv, claim):
    """TB1. Deny corroboration rather than forge it."""
    adv.take_over(claim)
    adv.silence_others(claim)


def T6_cloud_token_theft(home, adv, claim):
    """TB2. Remote and scalable, but class-correlated exactly like T4."""
    adv.take_over(claim)


def T7_text_injection(home, adv, claim):
    """
    TB2 -> agent. Out of scope: captures the agent, not the physical state.
    Kept in the harness to check the guardrail holds when the agent is
    fully compromised.
    """
    return


def T8_multi_modal_replay(home, adv, claim):
    """TB1/TB2. Replay every modality, re-timed to look current."""
    adv.take_over(claim, age_offset=2.0)


def T9_calibration_poisoning(home, adv, claim):
    """TB2. Move what the scorer calls normal; unsigned calibration baseline."""
    adv.take_over(claim, calibration_ok=False)


def T10_staleness_exploitation(home, adv, claim):
    """TB3. Genuine but old readings presented as current."""
    adv.take_over(claim, age_offset=45.0)


def T11_challenge_relay(home, adv, claim):
    """TB3. Relay the challenge somewhere the answer is genuinely true."""
    adv.take_over(claim)
    home.probe_relay = True


def T12_tier_laundering(home, adv, claim):
    """TB4. Reframe a tier-3 action as tier-1; the guardrail reads the actuator."""
    adv.take_over(claim)


def T13_escalation_fatigue(home, adv, claim):
    """TB4. Train the occupant to approve without reading. Rate limited."""
    adv.take_over(claim)


def T14_time_sync_attack(home, adv, claim):
    """TB2. Shift a clock so coherence windows misjudge freshness."""
    adv.take_over(claim, age_offset=-20.0)   # readings dated in the future


def T15_threshold_gaming(home, adv, claim):
    """
    TB3. Craft evidence landing just above theta(r).
    Open problem in the residual-risk register.
    """
    adv.take_over(claim, trust_prior=0.999)


def T16_attritional_degradation(home, adv, claim):
    """TB2. Let batteries die until quorum quietly fails."""
    adv.take_over(claim)
    adv.silence_others(claim)


def T17_challenge_laundering(home, adv, claim):
    """
    TB3. Hit-and-run: assert the predicate, then go silent exactly when the
    probe fires so an uncompromised sensor answers truthfully, turning an
    honest liveness proof into approval of the attacker's claim.

    Two mechanisms should stop it: the predicate-bound probe, whose honest
    answer reports the real state and contradicts the spoofed claim, and
    continuity scoring, which counts the silence against the channel.
    run_attack_harness.py ablates them separately.
    """
    adv.take_over(claim, n_reports=1, age_offset=2.0)
    adv.silence_others(claim)


def T18_probe_decoy(home, adv, claim):
    """TB1. An absorber or thermal mass forges the probe return."""
    adv.take_over(claim)
    home.probe_decoy = True
    home.forge_probe_value = claim.value


def T19_probe_fingerprinting(home, adv, claim):
    """TB1. Learn the keyed modulation over time."""
    adv.take_over(claim)


THREATS: Dict[str, Tuple[str, Callable]] = {
    "T1": ("Transducer spoofing", T1_transducer_spoofing),
    "T2": ("Sensor masking", T2_sensor_masking),
    "T3": ("RF replay", T3_rf_replay),
    "T4": ("Firmware compromise", T4_firmware_compromise),
    "T5": ("Selective jamming", T5_selective_jamming),
    "T6": ("Cloud token theft", T6_cloud_token_theft),
    "T7": ("Text injection (agent capture)", T7_text_injection),
    "T8": ("Multi-modal replay", T8_multi_modal_replay),
    "T9": ("Calibration poisoning", T9_calibration_poisoning),
    "T10": ("Staleness exploitation", T10_staleness_exploitation),
    "T11": ("Challenge relay", T11_challenge_relay),
    "T12": ("Tier laundering", T12_tier_laundering),
    "T13": ("Escalation fatigue", T13_escalation_fatigue),
    "T14": ("Time-sync attack", T14_time_sync_attack),
    "T15": ("Threshold gaming", T15_threshold_gaming),
    "T16": ("Attritional degradation", T16_attritional_degradation),
    "T17": ("Challenge laundering (hit-and-run)", T17_challenge_laundering),
    "T18": ("Probe decoy", T18_probe_decoy),
    "T19": ("Probe fingerprinting", T19_probe_fingerprinting),
}
