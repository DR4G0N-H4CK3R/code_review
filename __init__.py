"""Cross-Sensor Physical State Verification for Smart-Home LLM Agents."""
from .config import ACTUATOR_TIER, TIER_POLICY, VerifierConfig, policy_of, tier_of
from .evidence import Claim, Evidence, EvidenceSet, EvidenceCollector
from .home import SmartHome, SensorSpec, SensorBus, PhysicalState
from .verifier import VerificationLayer, VerificationResult, ProbeVerdict
from .guardrail import GuardrailEngine, ActionProposal, Decision, MockSmartHomeAgent, ALLOW, DENY

__all__ = [
    "ACTUATOR_TIER", "TIER_POLICY", "VerifierConfig", "policy_of", "tier_of",
    "Claim", "Evidence", "EvidenceSet", "EvidenceCollector",
    "SmartHome", "SensorSpec", "SensorBus", "PhysicalState",
    "VerificationLayer", "VerificationResult", "ProbeVerdict",
    "GuardrailEngine", "ActionProposal", "Decision", "MockSmartHomeAgent",
    "ALLOW", "DENY",
]
__version__ = "0.7.0"
