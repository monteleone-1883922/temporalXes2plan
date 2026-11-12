"""Single-pass log preprocessor for downstream analysis.

Reads a PetriNetLog exactly once and materialises, for every firing of every
labeled transition, the accumulated attribute state before the firing
(pre_state) and the attributes that actually changed (changed_attrs).

A first pass finds the start-of-trace ("static") attributes: set by the first visible event of
a trace and never changed afterwards, in at least AnalysisConfig.static_attr_probability of the
traces where they appear. They are not effects (the probability estimator leaves them out), but
they stay in the state like any other attribute: guards can read them, and in the few traces
where one of them does change later the state follows the log.

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
    TransitionFiringData, StaticAttributeInfo,
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
        attribute_categories: Optional[Dict[str, str]] = None,
        ignored_attributes: Optional[Set[str]] = None,
    ) -> None:
        self.config = config or AnalysisConfig()
        self._discretizer = discretizer
        self.attribute_categories = attribute_categories or {}
        # Attributes that must never enter the running state or a firing. The
        # Parser passes config.ignored_attributes plus whatever the noise
        # filter found; without it the config's own list is used.
        self.ignored_attributes = (
            ignored_attributes if ignored_attributes is not None else self.config.ignored_attributes
        )

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
        tau_seen: set = set()

        # Pre-build a set of valid branch transitions per XOR place for O(1)
        # membership checks inside the inner loop.
        valid_branches: Dict[PetriNet.Place, Set[Transition]] = {
            place: set(transitions) for place, transitions in xor_splits.items()
        }
        static_attributes_info = self._static_attribute_info(pn_log)
        static_attrs = {
            info.name for info in static_attributes_info.values()
            if info.is_static(self.config.static_attr_probability,
                              self.config.static_appearances_min_samples)
        }

        for execution in pn_log.executions:
            # Per-execution running state: sanitized attr name → latest value.
            # Reset for every execution so traces remain independent.
            # Categorical and numerical attributes are pre-seeded with the NONE
            # sentinel so pre_state always carries an explicit value for every
            # known attribute, avoiding NaN in the DT feature matrix.
            state: Dict[str, Any] = {
                attr: "none"
                for attr, t in self.attribute_categories.items()
                if t in ('categorical', 'numerical')
            }
            # True until the first visible (non-tau) step of this execution.
            at_first_event = True

            for step in execution.steps:
                # Tau transitions carry no observable activity and are never
                # targets of effect analysis.  They do not update the running
                # state and produce no transition_firings entry, but they do
                # traverse XOR split places and must be counted there.
                if step.is_tau:
                    tau_seen.add(step.transition)
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

                act = step.activity_name

                # --- 1. Snapshot the state BEFORE this step's attributes ---
                # This enforces causal ordering: features represent what was
                # known before the transition fired, not what it writes.
                pre_state = dict(state)

                # --- 2. Compute which attributes this step changes ---
                changed: Dict[str, Any] = {}
                for attr, val in step.attributes.items():
                    if attr in self.ignored_attributes or val is None:
                        continue
                    sanitized = attr
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
                if self.config.initial_event_as_input and at_first_event:
                    # The first event's attributes are the case's input, not a
                    # modelled effect: keep it out of effect/guard mining. The
                    # key is still created so the activity counts as fired
                    # (an activity seen only first ends up with no firings).
                    # Its XOR firing below is kept: it still says which branch
                    # was taken. The state update (step 4) still runs, so
                    # later pre_states include these attributes.
                    transition_firings.setdefault(act, [])
                else:
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
                    if attr in self.ignored_attributes or val is None:
                        continue
                    sanitized = attr
                    if (self._discretizer is not None
                            and sanitized in self._discretizer.boundaries):
                        val = self._discretizer.transform_value(sanitized, val)
                    if val is not None:
                        state[sanitized] = val
                at_first_event = False

        logger.info(
            "Preprocessing complete: %d labeled transitions fired, "
            "%d distinct tau transitions seen, %d XOR places, %d total firings.",
            len(transition_firings),
            len(tau_seen),
            len(xor_firings),
            sum(len(v) for v in transition_firings.values()),
        )

        return PreprocessedLog(
            transition_firings=dict(transition_firings),
            xor_firings=dict(xor_firings),
            static_attributes=static_attrs,
            static_attribute_info=static_attributes_info,
        )

    def _normalized_value(self, attr: str, val: Any) -> Any:
        """The value of attr as the state keeps it (discretized for numerical attributes), or
        None for an ignored attribute or a missing value."""
        if attr in self.ignored_attributes or val is None:
            return None
        if self._discretizer is not None and attr in self._discretizer.boundaries:
            return self._discretizer.transform_value(attr, val)
        return val

    def _static_attribute_info(self, pn_log: PetriNetLog) -> Dict[str, StaticAttributeInfo]:
        """First pass: how often each attribute is set by the first visible event of a trace and
        never changed afterwards.

        For every execution, an attribute counts as an appearance when some visible step writes
        it, and as a static appearance when the FIRST visible step writes it and no later visible
        step writes a different (discretized) value. A trace whose first event does not write the
        attribute is an appearance that is not static. Tau steps carry no attributes."""
        info: Dict[str, StaticAttributeInfo] = {}
        for execution in pn_log.executions:
            first_values: Dict[str, Any] = {}
            seen: Set[str] = set()
            changed_later: Set[str] = set()
            at_first_event = True
            for step in execution.steps:
                if step.is_tau:
                    continue
                for attr, raw in step.attributes.items():
                    val = self._normalized_value(attr, raw)
                    if val is None:
                        continue
                    seen.add(attr)
                    if at_first_event:
                        first_values[attr] = val
                    elif attr in first_values and first_values[attr] != val:
                        changed_later.add(attr)
                at_first_event = False
            for attr in seen:
                entry = info.setdefault(attr, StaticAttributeInfo(attr, 0, 0))
                entry.appearances += 1
                if attr in first_values and attr not in changed_later:
                    entry.static_appearances += 1
        return info
