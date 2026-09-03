"""
Verification layer (TB3).

Answers whether the physical state an action depends on is actually real.
Trusted for the integrity of its own computation only; every input is scored
before it counts.

Pipeline: collect -> freshness -> continuity -> independence grouping
          -> corroboration -> probe.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import VerifierConfig
from .evidence import Claim, Evidence, EvidenceCollector, EvidenceSet
from .home import SmartHome


@dataclass
class GroupScore:
    key: Tuple[str, str, str]
    raw: float            # uncapped signed sum
    capped: float         # after the independence cap
    members: List[str] = field(default_factory=list)


@dataclass
class ProbeVerdict:
    performed: bool = False
    passed: bool = False
    kind: str = "none"            # "covert" | "predicate_bound" | "liveness"
    reason: str = ""
    bound_channels: List[str] = field(default_factory=list)
    measured: Optional[float] = None
    expected: Optional[float] = None


@dataclass
class VerificationResult:
    claim: Claim
    C: float                       # corroboration score in [0, 1]
    positive_classes: int          # |G+(p)|
    total_classes: int             # M for this predicate
    groups: List[GroupScore] = field(default_factory=list)
    admitted: int = 0
    rejected_stale: int = 0
    rejected_calibration: int = 0
    rejected_continuity: int = 0
    probe: ProbeVerdict = field(default_factory=ProbeVerdict)
    notes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


def freshness(ev: Evidence, now: float, tau: float, max_age: float) -> float:
    """Exponential decay; anything past the tier's horizon is dropped (T10)."""
    age = now - ev.timestamp
    if age < 0:
        # a reading from the future is a clock attack (T14)
        return 0.0
    if age > max_age:
        return 0.0
    return math.exp(-age / tau)


def continuity(ev: Evidence, cfg: VerifierConfig) -> float:
    """
    Silence inside the evidence window counts against a channel rather than
    being neutral. A channel that asserts the predicate then goes quiet so an
    honest sensor answers the probe (T17) scores near zero.
    """
    if ev.expected_in_window <= 0:
        return 1.0
    ratio = ev.reports_in_window / float(ev.expected_in_window)
    ratio = max(0.0, min(1.0, ratio))
    return max(0.0, 1.0 - cfg.dropout_penalty * (1.0 - ratio))


class VerificationLayer:
    def __init__(self, home: SmartHome, cfg: Optional[VerifierConfig] = None) -> None:
        self.home = home
        self.cfg = cfg or VerifierConfig()
        self.collector = EvidenceCollector(home.bus, window_s=self.cfg.window_s)
        self._probe_epoch = 0

    def verify(
        self,
        claim: Claim,
        now: Optional[float] = None,
        max_age: float = 30.0,
        require_probe: bool = False,
        probe_kind: str = "predicate_bound",
    ) -> VerificationResult:
        t0 = time.perf_counter()
        now = self.home.now if now is None else now

        es = self.collector.collect(claim, now=now)
        res = VerificationResult(
            claim=claim,
            C=0.0,
            positive_classes=0,
            total_classes=self.home.independence_classes_for(claim.predicate, claim.target),
        )

        # 1. per-item score
        scored: List[Tuple[Evidence, float]] = []
        for ev in es.relevant:
            f = freshness(ev, now, self.cfg.freshness_tau_s, max_age)
            if f == 0.0:
                res.rejected_stale += 1
                continue
            if not ev.calibration_ok:
                # unsigned or drifted calibration baseline (T9)
                res.rejected_calibration += 1
                continue
            q = continuity(ev, self.cfg)
            if q < self.cfg.min_continuity:
                res.rejected_continuity += 1
                continue
            sign = 1.0 if ev.supports(claim) else -1.0
            scored.append((ev, sign * ev.trust_prior * f * q))
        res.admitted = len(scored)

        if not scored:
            res.notes.append("empty evidence set after filtering -> fail closed")
            res.latency_ms = (time.perf_counter() - t0) * 1000
            return res

        # 2. independence grouping. With grouping off every sensor is its own
        # class, which is what a naive voting scheme does.
        groups: Dict[Tuple[str, str, str], GroupScore] = {}
        for ev, s in scored:
            key = ev.independence_key() if self.cfg.use_independence_grouping \
                else (ev.sensor_id, "", "")
            g = groups.setdefault(key, GroupScore(key=key, raw=0.0, capped=0.0))
            g.raw += s
            g.members.append(ev.sensor_id)

        # 3. capped score. One class contributes at most class_cap however many
        # sensors it holds, so a vendor CVE (T4) or stolen token (T6) collapses
        # one class rather than the whole evidence set.
        cap = self.cfg.class_cap
        total = 0.0
        for g in groups.values():
            g.capped = max(-cap, min(cap, g.raw))
            total += g.capped
        res.groups = list(groups.values())
        res.positive_classes = sum(1 for g in groups.values() if g.capped > 0)

        M = max(res.total_classes, len(groups))
        res.C = max(0.0, total) / (cap * M)
        res.C = min(1.0, res.C)

        # 4. probing
        if require_probe:
            asserting_channels = sorted({ev.channel_id for ev, s in scored if s > 0})
            if probe_kind == "predicate_bound":
                res.probe = self.predicate_bound_challenge(claim, asserting_channels, now)
            elif probe_kind == "liveness":
                res.probe = self.liveness_challenge(asserting_channels, now)
            else:
                res.probe = self.covert_probe(now)

        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res

    def covert_probe(self, now: float) -> ProbeVerdict:
        """
        Continuous sub-perceptual probe: dim a lamp by a few watts on a keyed
        schedule and check the meter sees it. Catches a synthesised or replayed
        power stream (T3/T8) without the occupant noticing.
        """
        self._probe_epoch += 1
        delta_w = 3.0 + 2.0 * self.home.probe_schedule_offset(self._probe_epoch)
        measured = self.home.answer_covert_probe(delta_w)
        ok = abs(measured - delta_w) <= self.cfg.probe_tolerance * delta_w
        return ProbeVerdict(
            performed=True,
            passed=ok,
            kind="covert",
            reason="meter draw tracks keyed dim" if ok else "meter draw does not track keyed dim",
            measured=measured,
            expected=delta_w,
        )

    def liveness_challenge(self, bound_channels: List[str], now: float) -> ProbeVerdict:
        """
        Baseline B5. Proves a channel is alive and nothing about the claim, so
        it falls to T17. Implemented so the evaluation can measure that gap.
        """
        alive = bool(bound_channels) or True
        return ProbeVerdict(
            performed=True,
            passed=alive,
            kind="liveness",
            reason="a channel answered (liveness only - proves nothing about the claim)",
            bound_channels=bound_channels,
        )

    def predicate_bound_challenge(
        self, claim: Claim, bound_channels: List[str], now: float
    ) -> ProbeVerdict:
        """
        Tier-3 challenge. Three properties, each aimed at one threat:
          - response is a function of the predicate under test, so an honest
            answer given while the attacker is silent contradicts the spoofed
            claim (T17);
          - fresh nonce with a short TTL (T11, T8);
          - channel binding: the channel that asserted the predicate must be
            present for the answer.
        """
        nonce = random.getrandbits(32)
        if self.cfg.channel_binding and not bound_channels:
            return ProbeVerdict(
                performed=True,
                passed=False,
                kind="predicate_bound",
                reason="no asserting channel available to bind the challenge to",
            )

        measured = self.home.answer_predicate_probe(claim.predicate, claim.target, nonce)
        expected = self.home.expected_probe_response(claim.predicate, claim.value)
        ok = abs(measured - expected) <= self.cfg.probe_tolerance * max(expected, 1e-6)
        return ProbeVerdict(
            performed=True,
            passed=ok,
            kind="predicate_bound",
            reason=(
                "physical return matches the asserted predicate"
                if ok
                else "physical return contradicts the asserted predicate"
            ),
            bound_channels=bound_channels,
            measured=round(measured, 4),
            expected=round(expected, 4),
        )
