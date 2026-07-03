"""Single-pass log preprocessor for downstream analysis.

Reads a PetriNetLog exactly once and materialises, for every firing of every
labeled transition, the accumulated attribute state before the firing
(pre_state) and the attributes that actually changed (changed_attrs).

The result (PreprocessedLog) provides two access indexes over the same data:
  - transition_firings: per-transition  (used by effect statistics and DT)
  - xor_firings:        per-XOR-place   (used by XOR branch stats and DT)

No other module should need to iterate the raw PetriNetLog for attribute or
structural analysis after preprocessing.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from pm4py import PetriNet
from pm4py.objects.powl.obj import Transition

import core_utils as utils
from models import (
    AnalysisConfig,
    PetriNetLog,
    PreprocessedLog,
    TransitionFiringData,
)
from parsing.discretizer import Discretizer

logger = utils.get_logger(__name__)


class LogPreprocessor:
    """Builds a PreprocessedLog from a PetriNetLog in a single pass.

    Responsibilities:
    - Maintain a per-execution running attribute state (sanitized + discretized).
    - For every non-tau firing, snapshot the state before the firing, compute
      which attributes changed, and emit a TransitionFiringData.
    - Index each TransitionFiringData both by transition name and by any
      XOR-split places it traverses.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        discretizer: Optional[Discretizer] = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self._discretizer = discretizer

    def preprocess(
        self,
        pn_log: PetriNetLog,
        xor_splits: Dict[PetriNet.Place, List[Transition]],
    ) -> PreprocessedLog:
        """Run a single pass over pn_log and return a PreprocessedLog.

        Args:
            pn_log: Replay-filtered PetriNetLog.
            xor_splits: XOR-split places mapped to their outgoing branch
                transitions, as found in PetriNetModel.xor_splits.

        Returns:
            PreprocessedLog with both per-transition and per-XOR-place indexes.
        """
        transition_firings: Dict[str, List[TransitionFiringData]] = defaultdict(list)
        xor_firings: Dict[str, List[TransitionFiringData]] = defaultdict(list)

        # Pre-build a set of valid branch transitions per XOR place for O(1)
        # membership checks inside the inner loop.
        valid_branches: Dict[PetriNet.Place, Set[Transition]] = {
            place: set(transitions) for place, transitions in xor_splits.items()
        }

        for execution in pn_log.executions:
            # Per-execution running state: sanitized attr name → latest value.
            # Reset for every execution so traces remain independent.
            state: Dict[str, Any] = {}

            for step in execution.steps:
                # Tau transitions carry no observable activity and are never
                # targets of effect analysis.  They do not update the running
                # state and produce no transition_firings entry, but they do
                # traverse XOR split places and must be counted there.
                if step.is_tau:
                    for place in step.from_places:
                        if place in valid_branches and step.transition in valid_branches[place]:
                            fd = TransitionFiringData(
                                activity_name=step.activity_name,
                                pre_state=dict(state),
                                changed_attrs={},
                                from_places=frozenset(step.from_places),
                            )
                            xor_firings[place.name].append(fd)
                    continue

                act = utils.sanitize_name(step.activity_name)

                # --- 1. Snapshot the state BEFORE this step's attributes ---
                # This enforces causal ordering: features represent what was
                # known before the transition fired, not what it writes.
                pre_state = dict(state)

                # --- 2. Compute which attributes this step changes ---
                changed: Dict[str, Any] = {}
                for attr, val in step.attributes.items():
                    if attr in self.config.ignored_attributes or val is None:
                        continue
                    sanitized = utils.sanitize_name(attr)
                    if (self._discretizer is not None
                            and sanitized in self._discretizer.boundaries):
                        val = self._discretizer.transform_value(sanitized, val)
                    if val is None:
                        continue
                    # An attribute "changed" if its new value differs from the
                    # last seen value in this execution, or if it appears for
                    # the first time.
                    if sanitized not in state or state[sanitized] != val:
                        changed[sanitized] = val

                # --- 3. Create the TransitionFiringData and index it ---
                fd = TransitionFiringData(
                    activity_name=act,
                    pre_state=pre_state,
                    changed_attrs=changed,
                    from_places=frozenset(step.from_places),
                )
                transition_firings[act].append(fd)

                # If this step traverses one or more XOR-split places, index
                # the same object under each place it crosses.
                for place in step.from_places:
                    if place in valid_branches and step.transition in valid_branches[place]:
                        xor_firings[place.name].append(fd)

                # --- 4. Update the running state ---
                # All non-ignored attributes written by this step (even if they
                # did NOT change) are recorded so that future steps can compare
                # against the latest known value.  We update with the full
                # attribute set, not just `changed`, to ensure the state stays
                # in sync even if an attribute is rewritten with the same value.
                for attr, val in step.attributes.items():
                    if attr in self.config.ignored_attributes or val is None:
                        continue
                    sanitized = utils.sanitize_name(attr)
                    if (self._discretizer is not None
                            and sanitized in self._discretizer.boundaries):
                        val = self._discretizer.transform_value(sanitized, val)
                    if val is not None:
                        state[sanitized] = val

        logger.debug(
            "Preprocessing complete: %d transitions, %d XOR places, %d total firings.",
            len(transition_firings),
            len(xor_firings),
            sum(len(v) for v in transition_firings.values()),
        )

        return PreprocessedLog(
            transition_firings=dict(transition_firings),
            xor_firings=dict(xor_firings),
        )
