"""Lightweight Federated GNN for IoT Device-Relationship Anomaly Detection."""
from .datasets import FlowRecord, SyntheticSource, load_source, BENIGN, MALICIOUS
from .graphs import GraphBuilder, GraphSnapshot, NODE_FEATURES, EDGE_FEATURES
from .model_numpy import GCNAutoencoder, GCNConfig
from .scoring import evaluate, score_snapshot, alerts_for_snapshot, temporal_consensus, roc_auc
from .aggregation import aggregate, clip_and_noise, validate_update, STRATEGIES
from .federated import Site, build_sites, run_federated, train_local_only, train_centralised, personalise

__all__ = [
    "FlowRecord", "SyntheticSource", "load_source", "BENIGN", "MALICIOUS",
    "GraphBuilder", "GraphSnapshot", "NODE_FEATURES", "EDGE_FEATURES",
    "GCNAutoencoder", "GCNConfig",
    "evaluate", "score_snapshot", "alerts_for_snapshot", "temporal_consensus", "roc_auc",
    "aggregate", "clip_and_noise", "validate_update", "STRATEGIES",
    "Site", "build_sites", "run_federated", "train_local_only", "train_centralised", "personalise",
]
__version__ = "0.7.0"
