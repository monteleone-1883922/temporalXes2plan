from dataclasses import dataclass, field
from typing import Dict, Set

from pm4py import PetriNet, Marking
from pm4py.objects.powl.obj import Transition


@dataclass
class PetriNetModel:
    """Complete structural description of a discovered Petri net.

    Produced by ModelDiscoverer.discover() and used as the authoritative
    structural representation throughout the parsing pipeline.
    """
    petrinet: PetriNet
    initial_marking: Marking
    final_marking: Marking
    activities: Set[str]
    silent_transitions: Dict[Transition, str]


@dataclass
class AnalysisConfig:
    """Global configuration for the XES log analysis pipeline.

    Controls pre-discretization (k-means++), decision tree cascade thresholds,
    and statistical fallback modes for XOR splits and conditional effects.
    """

    # --- Pre-discretization (k-means++) ---
    kmeans_max_k: int = 5
    kmeans_min_cluster_fraction: float = 0.05
    kmeans_silhouette_threshold: float = 0.10
    kmeans_n_init: int = 10

    # --- Decision tree cascade (shared for XOR splits and conditional effects) ---
    dt_min_samples: int = 30
    dt_min_accuracy: float = 0.75

    # --- XOR split statistical fallback (Level 2) ---
    # majority_only | weighted | pruned_weighted
    xor_statistical_mode: str = "pruned_weighted"
    xor_prune_threshold: float = 0.10

    # --- Conditional effect statistical fallback (Level 2) ---
    # appearance (2A): threshold | always | duplicate
    effect_appearance_mode: str = "duplicate"
    effect_appearance_threshold: float = 0.20
    # value (2B): majority_only | duplicate
    effect_value_mode: str = "duplicate"

    # --- Logging ---
    log_removed_effects: bool = True
