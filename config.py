"""Tiers, thresholds and verifier knobs, kept in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Tier is resolved from the actuator command, never from agent text (S7.1, T12).
ACTUATOR_TIER: Dict[str, int] = {
    # Tier 0 - read-only
    "READ_TEMPERATURE": 0,
    "READ_SENSOR_STATUS": 0,
    "LOG_EVENT": 0,
    # Tier 1 - trivially reversible
    "LIGHT_ON": 1,
    "LIGHT_OFF": 1,
    "SET_SCENE": 1,
    # Tier 2 - reversible but consequential
    "THERMOSTAT_SET": 2,
    "OPEN_BLINDS": 2,
    "START_VACUUM": 2,
    # Tier 3 - irreversible / safety-critical
    "UNLOCK_EXTERIOR_DOOR": 3,
    "DISARM_ALARM": 3,
    "OPEN_GARAGE": 3,
    "OVEN_ON": 3,
    "WATER_VALVE_OPEN": 3,
}

# Unknown commands fail closed at the highest tier.
DEFAULT_TIER = 3


@dataclass(frozen=True)
class TierPolicy:
    """Evidence requirement for one risk tier."""

    theta: float              # minimum corroboration score C(p)
    min_classes: int          # minimum number of independent supporting classes
    require_probe: bool       # predicate-bound active challenge required?
    max_evidence_age_s: float # freshness horizon for admissible evidence


TIER_POLICY: Dict[int, TierPolicy] = {
    0: TierPolicy(theta=0.00, min_classes=0, require_probe=False, max_evidence_age_s=1e9),
    1: TierPolicy(theta=0.35, min_classes=1, require_probe=False, max_evidence_age_s=120.0),
    2: TierPolicy(theta=0.60, min_classes=2, require_probe=False, max_evidence_age_s=60.0),
    3: TierPolicy(theta=0.85, min_classes=3, require_probe=True, max_evidence_age_s=30.0),
}


@dataclass
class VerifierConfig:
    """Knobs for the verification layer."""

    window_s: float = 120.0           # evidence collection window
    freshness_tau_s: float = 15.0     # exponential freshness decay constant
    class_cap: float = 1.0            # max signed contribution of one independence class
    dropout_penalty: float = 1.0      # how hard silence inside the window is punished
    min_continuity: float = 0.25      # below this a channel contributes nothing
    probe_tolerance: float = 0.25     # relative tolerance on probe response
    probe_nonce_ttl_s: float = 5.0    # a challenge answer older than this is dead
    channel_binding: bool = True      # asserting channel must answer the challenge
    use_independence_grouping: bool = True  # False = ablation, every sensor counts alone
    escalation_budget: int = 3        # human confirmations allowed per hour (T13)


@dataclass
class AdversaryBound:
    """
    Attacker controls at most k sensors, spanning at most m of M independence
    classes. Safety condition (S7.2):

        theta(r) > m / M

    since an attacker owning m of M classes can lift C(p) by at most m/M.
    Checked empirically in experiments/run_sweep.py.
    """

    k: int = 1
    m: int = 1
    M: int = 4


def tier_of(actuator_command: str) -> int:
    """Resolve risk tier from the actuator command. Fails closed."""
    return ACTUATOR_TIER.get(actuator_command.upper(), DEFAULT_TIER)


def policy_of(tier: int) -> TierPolicy:
    return TIER_POLICY.get(tier, TIER_POLICY[3])
