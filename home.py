"""
Deterministic smart-home simulator.

Holds the ground-truth physical state, a population of sensors reporting it
onto a bus, and the physical channel the probes are answered over. The
adversary in attacks.py writes to the same bus (TB1/TB2).
"""
from __future__ import annotations

import hashlib
import hmac
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence import Evidence


@dataclass
class PhysicalState:
    """What is actually true in the house at time t."""

    occupancy: Dict[str, bool] = field(default_factory=dict)
    door_state: Dict[str, str] = field(default_factory=dict)  # "closed" | "open"
    alarm_armed: bool = True

    def get(self, predicate: str, target: str) -> Any:
        if predicate == "occupancy":
            return self.occupancy.get(target, False)
        if predicate == "door_state":
            return self.door_state.get(target, "closed")
        if predicate == "alarm_armed":
            return self.alarm_armed
        raise KeyError(predicate)


@dataclass
class SensorSpec:
    sensor_id: str
    vendor: str
    modality: str
    transport: str
    placement: str
    predicate: str
    target: str
    period_s: float = 5.0
    trust_prior: float = 1.0
    error_rate: float = 0.01     # probability of an honest wrong reading


class SensorBus:
    """The untrusted data plane (TB2). Honest sensors and the adversary both write here."""

    def __init__(self) -> None:
        self._readings: List[Evidence] = []

    def publish(self, ev: Evidence) -> None:
        self._readings.append(ev)

    def readings_between(self, start: float, end: float) -> List[Evidence]:
        return [e for e in self._readings if start <= e.timestamp <= end]

    def drop_where(self, pred) -> int:
        """Remove readings matching pred (jamming / masking attacks)."""
        before = len(self._readings)
        self._readings = [e for e in self._readings if not pred(e)]
        return before - len(self._readings)

    def clear(self) -> None:
        self._readings.clear()


class SmartHome:
    """Ground truth, honest sensor population and physical probe channel."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.state = PhysicalState(
            occupancy={"living_room": False, "hallway": False},
            door_state={"front": "closed"},
            alarm_armed=True,
        )
        self.bus = SensorBus()
        self.sensors: List[SensorSpec] = []
        self.now = 1_000_000.0
        # key for the covert probe schedule; stays inside the TCB
        self.probe_key = b"amrita-cys-probe-key"
        # probe-plane flags set by attacks.py (T11 / T18)
        self.probe_relay = False
        self.probe_decoy = False
        self.forge_probe_value = None

    def commission_default(self) -> "SmartHome":
        """Occupancy deployment: 4 independence classes, 8 physical sensors."""
        specs = [
            # class A - PIR motion over Zigbee, vendor Aqara
            SensorSpec("pir_lr_1", "aqara", "motion", "zigbee", "living_room", "occupancy", "living_room", 5.0, 1.0),
            SensorSpec("pir_hall_1", "aqara", "motion", "zigbee", "hallway", "occupancy", "living_room", 5.0, 0.8),
            # class B - mmWave presence over Z-Wave, vendor Aeotec
            SensorSpec("mmwave_lr_1", "aeotec", "mmwave", "zwave", "living_room", "occupancy", "living_room", 4.0, 1.0),
            # class C - camera person-detect over Wi-Fi, vendor Reolink
            SensorSpec("cam_lr_1", "reolink", "camera", "wifi", "living_room", "occupancy", "living_room", 6.0, 1.0),
            # class D - smart-meter power signature over wired bus, vendor Shelly
            SensorSpec("meter_main", "shelly", "power", "wired", "panel", "occupancy", "living_room", 3.0, 0.9),
            # door contact sensors (own predicate)
            SensorSpec("door_front_1", "aqara", "door_contact", "zigbee", "front", "door_state", "front", 5.0, 1.0),
            SensorSpec("door_front_2", "aeotec", "door_contact", "zwave", "front", "door_state", "front", 5.0, 1.0),
            SensorSpec("acoustic_hall", "sonoff", "acoustic", "wifi", "hallway", "door_state", "front", 6.0, 0.7),
        ]
        self.sensors.extend(specs)
        return self

    def commission_vendor_heavy(self, n_same_class: int = 6) -> "SmartHome":
        """
        Deployment where one vendor sells most of the sensors. Used by
        run_sweep.py to show what the independence cap buys.
        """
        self.sensors = [
            SensorSpec("mmwave_lr_1", "aeotec", "mmwave", "zwave", "living_room",
                       "occupancy", "living_room", 4.0, 1.0),
            SensorSpec("cam_lr_1", "reolink", "camera", "wifi", "living_room",
                       "occupancy", "living_room", 6.0, 1.0),
            SensorSpec("meter_main", "shelly", "power", "wired", "panel",
                       "occupancy", "living_room", 3.0, 0.9),
        ]
        for i in range(n_same_class):
            self.sensors.append(
                SensorSpec(f"pir_aqara_{i}", "aqara", "motion", "zigbee", "living_room",
                           "occupancy", "living_room", 5.0, 1.0)
            )
        return self

    def independence_classes_for(self, predicate: str, target: str) -> int:
        """M - commissioned independent classes for this predicate."""
        keys = {
            (s.modality, s.vendor, s.transport)
            for s in self.sensors
            if s.predicate == predicate and s.target == target
        }
        return max(1, len(keys))

    def tick(self, duration_s: float = 30.0, dt: float = 1.0) -> None:
        """Advance time, letting each honest sensor report on its own period."""
        steps = int(duration_s / dt)
        for _ in range(steps):
            self.now += dt
            for s in self.sensors:
                if (self.now % s.period_s) < dt:
                    self._publish_honest(s)

    def _publish_honest(self, s: SensorSpec) -> None:
        truth = self.state.get(s.predicate, s.target)
        value = truth
        if self.rng.random() < s.error_rate:
            value = self._flip(s.predicate, truth)
        expected = max(1, int(30.0 / s.period_s))
        self.bus.publish(
            Evidence(
                sensor_id=s.sensor_id,
                vendor=s.vendor,
                modality=s.modality,
                transport=s.transport,
                placement=s.placement,
                predicate=s.predicate,
                target=s.target,
                value=value,
                timestamp=self.now,
                trust_prior=s.trust_prior,
                calibration_ok=True,
                reports_in_window=expected,
                expected_in_window=expected,
            )
        )

    @staticmethod
    def _flip(predicate: str, value: Any) -> Any:
        if predicate == "door_state":
            return "open" if value == "closed" else "closed"
        return not value

    def probe_schedule_offset(self, epoch: int) -> float:
        """
        Keyed probe schedule (T19). The modulation instant inside each epoch is
        HMAC-derived, so an observer cannot predict the sub-perceptual dim.
        """
        digest = hmac.new(self.probe_key, str(epoch).encode(), hashlib.sha256).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

    def answer_covert_probe(self, delta_w: float) -> float:
        """
        Response to a sub-perceptual light dim: the main meter should see a
        proportional draw change. Returns delta_w plus measurement noise.
        """
        return delta_w * (1.0 + self.rng.gauss(0.0, 0.03))

    def answer_predicate_probe(self, predicate: str, target: str, nonce: int) -> float:
        """
        Predicate-bound challenge (S7.5). Ultrasonic / IR return energy depends
        on whether a body is in the room, so the answer is a function of the
        predicate under test. Returns the measured return energy.
        """
        rng = random.Random(nonce)

        if self.probe_decoy and self.forge_probe_value is not None:
            # T18: attacker physically forges the return. The probe gate is
            # bypassed, but corroboration still has to be satisfied.
            base = self.expected_probe_response(predicate, self.forge_probe_value)
            return base + rng.gauss(0.0, 0.02)

        if self.probe_relay:
            # T11: challenge answered from a different physical volume, so we
            # read the other room's truth.
            other = "hallway" if target != "hallway" else "living_room"
            truth = self.state.get(predicate, other)
            base = 0.62 if truth else 0.18
            return base + rng.gauss(0.0, 0.02)

        truth = self.state.get(predicate, target)
        if predicate == "occupancy":
            base = 0.62 if truth else 0.18   # a body absorbs / reflects differently
        elif predicate == "door_state":
            base = 0.70 if truth == "open" else 0.22
        else:
            base = 0.5
        return base + rng.gauss(0.0, 0.02)

    def expected_probe_response(self, predicate: str, asserted_value: Any) -> float:
        """What the return should look like if the asserted value were true."""
        if predicate == "occupancy":
            return 0.62 if asserted_value else 0.18
        if predicate == "door_state":
            return 0.70 if asserted_value == "open" else 0.22
        return 0.5
