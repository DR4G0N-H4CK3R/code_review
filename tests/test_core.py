"""
Tests for the security-relevant invariants.

Run:  python -m tests.test_core   (no pytest needed)
      pytest tests/
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks import Adversary, T3_rf_replay, T9_calibration_poisoning, T14_time_sync_attack
from csvguard import (
    ActionProposal,
    Claim,
    GuardrailEngine,
    SmartHome,
    VerificationLayer,
    VerifierConfig,
    tier_of,
)

CLAIM = Claim("occupancy", "living_room", True)


def home(occupied: bool, seed: int = 1) -> SmartHome:
    h = SmartHome(seed=seed).commission_default()
    h.state.occupancy["living_room"] = occupied
    h.tick(40.0)
    return h


def guard(h, mode="full", cfg=None):
    return GuardrailEngine(VerificationLayer(h, cfg or VerifierConfig()), mode=mode)


# --------------------------------------------------------------------------
def test_tier_comes_from_actuator_not_agent():
    """T12: an agent claiming tier 1 must not change the resolved tier."""
    assert tier_of("UNLOCK_EXTERIOR_DOOR") == 3
    p = ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM, claimed_tier=1)
    d = guard(home(False)).evaluate(p)
    assert d.tier == 3, d.tier


def test_unknown_command_fails_closed():
    assert tier_of("SOME_NEW_ACTUATOR") == 3


def test_benign_allow():
    d = guard(home(True)).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert d.allowed, d.reasons


def test_single_sensor_spoof_denied():
    h = home(False)
    adv = Adversary(h, k=1, m=1, seed=2)
    adv.select_targets(CLAIM.predicate, CLAIM.target)
    adv.take_over(CLAIM)
    d = guard(h).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert not d.allowed


def test_replay_is_stale():
    h = home(False)
    adv = Adversary(h, k=8, m=4, seed=2)
    adv.select_targets(CLAIM.predicate, CLAIM.target)
    T3_rf_replay(h, adv, CLAIM)
    d = guard(h).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert not d.allowed
    assert d.result.rejected_stale > 0


def test_future_timestamps_rejected():
    """T14: a reading dated in the future must not count as fresh."""
    h = home(False)
    adv = Adversary(h, k=8, m=4, seed=2)
    adv.select_targets(CLAIM.predicate, CLAIM.target)
    T14_time_sync_attack(h, adv, CLAIM)
    d = guard(h).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert not d.allowed


def test_unsigned_calibration_rejected():
    h = home(False)
    adv = Adversary(h, k=8, m=4, seed=2)
    adv.select_targets(CLAIM.predicate, CLAIM.target)
    T9_calibration_poisoning(h, adv, CLAIM)
    d = guard(h).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert not d.allowed
    assert d.result.rejected_calibration > 0


def test_independence_cap_holds():
    """One class contributes at most class_cap, however many sensors it holds."""
    h = SmartHome(seed=3).commission_vendor_heavy(6)
    h.state.occupancy["living_room"] = False
    h.tick(40.0)
    adv = Adversary(h, k=6, m=1, seed=4)
    adv.select_targets(CLAIM.predicate, CLAIM.target, prefer_largest=True)
    adv.take_over(CLAIM)
    adv.silence_others(CLAIM)
    res = VerificationLayer(h, VerifierConfig()).verify(CLAIM, max_age=30.0)
    assert res.positive_classes == 1
    assert max(g.capped for g in res.groups) <= 1.0 + 1e-9
    assert res.C <= 1.0 / res.total_classes + 1e-9


def test_tier0_needs_no_evidence():
    d = guard(home(False)).evaluate(ActionProposal("READ_TEMPERATURE", CLAIM))
    assert d.allowed


def test_empty_bus_fails_closed():
    h = home(False)
    h.bus.clear()
    d = guard(h).evaluate(ActionProposal("UNLOCK_EXTERIOR_DOOR", CLAIM))
    assert not d.allowed


def test_escalation_is_rate_limited():
    g = guard(home(True))
    now = 1_000_000.0
    approvals = [g.escalate(now + i) for i in range(6)]
    assert sum(approvals) == g.cfg.escalation_budget


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
