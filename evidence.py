"""
Claims and evidence.

A claim is a predicate over physical state that a proposed action depends on,
e.g. occupancy(living_room) == True. An evidence item is one sensor reading
supporting or contradicting it, carrying the provenance needed to judge
independence: vendor, modality, transport, placement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class Claim:
    """A predicate under test, e.g. Claim('occupancy', 'living_room', True)."""

    predicate: str
    target: str
    value: Any

    def key(self) -> str:
        return f"{self.predicate}({self.target})=={self.value}"

    def __str__(self) -> str:  # pragma: no cover
        return self.key()


@dataclass
class Evidence:
    """One sensor observation admitted into an evidence set."""

    sensor_id: str
    vendor: str
    modality: str          # motion | camera | door_contact | power | acoustic | rf | thermal
    transport: str         # zigbee | zwave | wifi | ble | wired | vendor_cloud
    placement: str         # room / mounting point
    predicate: str
    target: str
    value: Any
    timestamp: float
    trust_prior: float = 1.0        # from attestation / calibration provenance
    calibration_ok: bool = True     # signed calibration baseline verified (T9)
    reports_in_window: int = 0      # observed heartbeat count (T16/T17)
    expected_in_window: int = 0     # expected heartbeat count from device profile
    channel_id: Optional[str] = None  # for channel binding (T11/T17)

    def __post_init__(self) -> None:
        if self.channel_id is None:
            # default channel is the (sensor, transport) pair
            object.__setattr__(self, "channel_id", f"{self.sensor_id}@{self.transport}")

    def supports(self, claim: Claim) -> Optional[bool]:
        """True = supports, False = contradicts, None = irrelevant."""
        if self.predicate != claim.predicate or self.target != claim.target:
            return None
        return self.value == claim.value

    def independence_key(self) -> Tuple[str, str, str]:
        """
        Two items are dependent if they share vendor, modality and transport.
        One vendor CVE (T4) or stolen cloud token (T6) collapses one class.
        """
        return (self.modality, self.vendor, self.transport)


@dataclass
class EvidenceSet:
    """All evidence gathered for one claim over one window."""

    claim: Claim
    items: List[Evidence] = field(default_factory=list)
    window_start: float = 0.0
    window_end: float = 0.0

    def add(self, ev: Evidence) -> None:
        self.items.append(ev)

    @property
    def relevant(self) -> List[Evidence]:
        return [e for e in self.items if e.supports(self.claim) is not None]

    def __len__(self) -> int:  # pragma: no cover
        return len(self.items)


class EvidenceCollector:
    """
    Pulls readings off the bus into an EvidenceSet. Gathers only; every trust
    judgement happens later in the verification layer, to keep the TCB small.
    """

    def __init__(self, bus: "SensorBus", window_s: float = 30.0) -> None:
        self.bus = bus
        self.window_s = window_s

    def collect(self, claim: Claim, now: Optional[float] = None) -> EvidenceSet:
        now = time.time() if now is None else now
        start = now - self.window_s
        es = EvidenceSet(claim=claim, window_start=start, window_end=now)
        for ev in self.bus.readings_between(start, now):
            if ev.supports(claim) is not None:
                es.add(ev)
        return es
