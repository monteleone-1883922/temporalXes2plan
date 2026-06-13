# Parsing Module — Technical Reference

> **Status:** current as of the `Parser` rewrite (typed pipeline facade).
> Previous entries referencing `StructureAnalyzer`, `CorrelationMiner`,
> module-level `decision_mining` functions are obsolete; this document
> supersedes them.

---

## Overview

The `src/parsing/` package converts a raw XES event log into a
`ParseResult` — an encoder-ready representation of the process model
with probabilistic routing, conditional effects, and duration statistics.

```
XES file
   │
   ▼
Parser.__init__()   ← single public entry point
   │
   ├─ Step 1  LogProcessor          load + lifecycle filter + variant filter + 80/20 split
   ├─ Step 2  ModelDiscoverer       Petri net discovery + structural analysis
   ├─ Step 3  Discretizer           k-means++ boundaries for numeric attributes
   ├─ Step 4  PetriNetLogBuilder    token replay → PetriNetLog
   ├─ Step 5  LogPreprocessor       single-pass → PreprocessedLog (pre_state, changed_attrs)
   ├─ Step 6  ProbabilityEstimator  XOR branch probs + attribute effect probs
   ├─ Step 7  DecisionMiner         screen_xor_splits() → XorSplitScreening
   ├─ Step 8  DecisionMiner         screen_effects()    → TransitionScreening
   ├─ Step 9  DecisionMiner         mine_xor_splits()   → XorSplitGuards (DT)
   ├─ Step 10 DecisionMiner         mine_effects()      → TransitionEffects (DT)
   │
   ├─ _save_snapshot()       write PNML + JSON to snapshot_dir
   ├─ _filter_xor_splits()   prune Petri net branches; build XorBranchInfo
   ├─ _filter_effects()      build EffectInfo per (transition, attribute)
   └─ _build_parse_result()  assemble ParseResult
```

All modules below the `Parser` facade are internal to the pipeline.
Consumers interact only with `Parser` and its `parse_result` attribute.

---

## Entry point: `xes_parser.py`

### `class Parser`

Facade that owns and orchestrates all pipeline steps. Instantiating `Parser`
runs the full pipeline; the final encoder-ready result is available as
`parser.parse_result`.

**Constructor signature:**

```python
Parser(
    log_path: str,
    coverage_percentage: float,
    discovery_algorithm: str = 'inductive',   # 'alpha' | 'inductive' | 'heuristics' | 'ilp'
    use_activity_classifier: bool = False,
    config: Optional[AnalysisConfig] = None,
)
```

**Key intermediate attributes** (available after `__init__` completes):

| Attribute | Type | Source |
|---|---|---|
| `log` | `EventLog` (train split) | LogProcessor |
| `full_log` | `EventLog` (full) | LogProcessor |
| `attributes` | `Set[str]` | LogProcessor |
| `attribute_categories` | `Dict[str, str]` | LogProcessor |
| `petri_net_model` | `PetriNetModel` | ModelDiscoverer (pruned in-place) |
| `discretizer` | `Discretizer` | Discretizer |
| `pn_log` | `PetriNetLog` | PetriNetLogBuilder |
| `preprocessed_log` | `PreprocessedLog` | LogPreprocessor |
| `xor_stats` | `Dict[str, XorSplitStats]` | ProbabilityEstimator |
| `attribute_effects` | `Dict[str, AttributeEffect]` | ProbabilityEstimator |
| `xor_screening` | `Dict[str, XorSplitScreening]` | DecisionMiner |
| `effect_screening` | `Dict[str, TransitionScreening]` | DecisionMiner |
| `xor_guards` | `Dict[str, XorSplitGuards]` | DecisionMiner |
| `transition_effects` | `Dict[str, TransitionEffects]` | DecisionMiner |
| `parse_result` | `ParseResult` | `_build_parse_result()` |

**Private methods:**

| Method | Purpose |
|---|---|
| `_save_snapshot()` | Write PNML + JSON to `config.snapshot_dir` before pruning |
| `_filter_xor_splits()` | Prune Petri net branches; populate `_xor_branch_info` |
| `_apply_dt_level()` | Cascade level 1 — DT guards available |
| `_apply_deterministic()` | Cascade level 1 — single certain branch |
| `_apply_fallback()` | Cascade level 2/3 — statistical fallback |
| `_filter_effects()` | Build `_transition_effect_info` from screening results |
| `_build_parse_result()` | Assemble `ParseResult` from pruned net + info maps |

### XOR cascade levels

`_filter_xor_splits()` assigns a `cascade_level` to every surviving branch:

| Level | Trigger | Meaning |
|---|---|---|
| 1 | `screening.action == "dt"` | DT guards available for routing |
| 1 | `screening.action == "deterministic"` | Single certain branch; no DT needed |
| 2 | `screening.action == "fallback"` and `total_samples >= min_samples` | Probability-based routing (configurable via `xor_statistical_mode`) |
| 3 | `screening.action == "fallback"` and `total_samples < min_samples` | Too few samples; all branches kept with equal weight |

`xor_statistical_mode` controls level-2 behaviour:
- `"majority_only"` — keep only the most probable branch
- `"weighted"` — keep all active branches; encoder assigns cost via `-log(p)`
- `"pruned_weighted"` (default) — remove pruned branches; keep active ones with costs

### Effect cascade levels

`_filter_effects()` applies two independent level-3 gates:

| Gate | Condition | Consequence |
|---|---|---|
| Transition gate | `total_firings < min_samples` | Entire transition excluded |
| Attribute gate | `appearance_samples < min_samples` and `appearance_action == "fallback"` | Only that attribute excluded |

Surviving `(transition, attribute)` pairs produce an `EffectInfo` with:
- `appearance_level` 1 (DT or deterministic) or 2 (fallback)
- `value_level` 1 (DT or deterministic) or 2 (fallback)
- `value_probabilities` with pruned values removed

---

## Step 1 — `log_processor.py`

### `class LogProcessor`

Loads, filters, and categorizes attributes from a pm4py XES event log.

**Constructor:**
```python
LogProcessor(log_path: str, use_activity_classifier: bool = False)
```

**Ignored attributes** (class constant):
```
case:concept:name, concept:name, time:timestamp, lifecycle:transition,
org:resource, org:group, variant-index, Resource, org:role
```

**Methods:**

`load_and_filter_log(coverage_percentage) → Tuple[EventLog, EventLog, EventLog]`
- Loads via `pm4py.objects.log.importer.xes.importer.apply()`
- If `use_activity_classifier=True`: concatenates `concept:name + lifecycle:transition`
  (e.g. `"Accepted (In Progress)"`) on every event — no lifecycle filtering
- Otherwise: filters events to `lifecycle:transition == 'complete'` only
- Applies `pm4py.filter_variants_by_coverage_percentage()` to retain the most
  frequent variants; falls back to the unfiltered log if the filter removes everything
- Returns: `(filtered_log, full_log, full_lifecycle_log)`

`split_train_test(log) → Tuple[EventLog, EventLog]`
- 80 / 20 split via `pm4py.split_train_test(train_percentage=0.8)`

`initialize_attributes(full_log) → Tuple[Set[str], Dict[str, str]]`
- Extracts all event-level attribute names (excluding ignored) as sanitized strings
- Returns `(attributes_set, attribute_categories_dict)`

`categorize_attributes(full_log) → Dict[str, str]`
- Includes both trace-level (`case:*`) and event-level attributes
- Classification: `'boolean'` if all values are `bool`; `'numerical'` if all
  values are `int` or `float` (not `bool`); `'categorical'` otherwise
- Mixed-type attributes are forced to `'categorical'` with a warning

---

## Step 2 — `model_discoverer.py`

### `class ModelDiscoverer`

Wraps pm4py discovery algorithms and performs full structural analysis of the
discovered net in a single pass.

**Supported algorithms:**

| Key | Function |
|---|---|
| `'alpha'` | `pm4py.discovery.discover_petri_net_alpha` |
| `'inductive'` | `pm4py.discovery.discover_petri_net_inductive` (default) |
| `'heuristics'` | `pm4py.discovery.discover_petri_net_heuristics` |
| `'ilp'` | `pm4py.discovery.discover_petri_net_ilp` |

**Methods:**

`discover(train_log) → PetriNetModel`
- Runs the selected algorithm → `(petrinet, initial_marking, final_marking)`
- Calls structural helpers to build arc maps and identify XOR splits
- Returns a `PetriNetModel` with all indexes pre-computed

**Private structural helpers:**

| Method | Output |
|---|---|
| `_extract_activities_and_silent()` | `(Set[str], Dict[Transition, str])` — activity names; tau names `tau_1`, `tau_2`, … |
| `_build_arc_maps()` | `(trans_inputs, trans_outputs)` — transition → input/output place sets |
| `_identify_xor_splits()` | `Dict[Place, List[Transition]]` — places with >1 outgoing transition |
| `_build_place_inputs()` | `Dict[Place, List[Transition]]` — inverse of `trans_outputs` |

**`PetriNetModel`** (returned dataclass, defined in `models.py`):

| Field | Type | Description |
|---|---|---|
| `petrinet` | `PetriNet` | pm4py net object |
| `initial_marking` | `Marking` | Source place(s) |
| `final_marking` | `Marking` | Sink place(s) |
| `activities` | `Set[str]` | All sanitized activity names incl. tau |
| `silent_transitions` | `Dict[Transition, str]` | Tau transitions with assigned names |
| `trans_inputs` | `Dict[Transition, Set[Place]]` | Input places per transition |
| `trans_outputs` | `Dict[Transition, Set[Place]]` | Output places per transition |
| `xor_splits` | `Dict[Place, List[Transition]]` | XOR-split places |
| `place_inputs` | `Dict[Place, List[Transition]]` | Transitions that produce into each place |

`PetriNetModel.prune_transitions(to_remove)` returns a new `PetriNetModel`
with the specified transitions (and their arcs) removed and all index maps
rebuilt.

---

## Step 3 — `discretizer.py`

### `class Discretizer`

Pre-computes k-means++ cluster boundaries for all numeric attributes.
Must be `fit()` before any DT training so all downstream components share
the same fixed interval vocabulary.

**Constructor:**
```python
Discretizer(config: Optional[AnalysisConfig] = None)
```

**Public API:**

`fit(log: EventLog, numeric_attributes: List[str]) → None`
- Converts the log to a DataFrame via `pm4py.convert_to_dataframe()`
- For each attribute: extracts non-null float values; calls `_find_best_boundaries()`
- Skips attributes that are constant, have `< 2` non-null values, or whose best
  silhouette score falls below `config.kmeans_silhouette_threshold`
- Populates `self.boundaries`

`transform_value(attr: str, value: Any) → str`
- Maps a numeric value to its interval label, e.g. `"lte_10_0"`,
  `"gte_10_0_lte_97_5"`, `"gte_97_5"`
- Delegates to `core_utils.discretize_value(attr, value, self.boundaries)`
- Falls back to `str(value)` for attributes without boundaries or `None` values

`boundaries: Dict[str, List[float]]`
- Format: `{ "crp": [10.0, 97.5] }` — sorted split points
- Absent for attributes that were skipped

**Private helper:**

`_find_best_boundaries(attr, values, n_unique) → List[float]`
- Runs `KMeans(init='k-means++', n_init=config.kmeans_n_init)` for
  `k ∈ [2, min(config.kmeans_max_k, n_unique - 1)]`
- Rejects k values where any cluster fraction is below
  `config.kmeans_min_cluster_fraction`
- Selects k with best silhouette score; returns midpoints between sorted centroids
- Returns `[]` if no valid k achieves `silhouette >= config.kmeans_silhouette_threshold`

**Relevant `AnalysisConfig` fields:**

| Field | Default | Meaning |
|---|---|---|
| `kmeans_max_k` | 5 | Maximum number of clusters to try |
| `kmeans_n_init` | 10 | Number of k-means random restarts |
| `kmeans_min_cluster_fraction` | 0.05 | Minimum cluster size fraction |
| `kmeans_silhouette_threshold` | 0.3 | Minimum silhouette score to accept a split |

---

## Step 4 — `petri_net_log_builder.py`

### `class PetriNetLogBuilder`

Builds a `PetriNetLog` by aligning each trace in the event log with the
activated transitions produced by pm4py token-based replay. Supports
lifecycle-aware logs (start/complete event pairs).

**Constructor:**
```python
PetriNetLogBuilder(
    petrinet, initial_marking, final_marking,
    trans_inputs, trans_outputs,
    silent_transitions,
    config: AnalysisConfig,
)
```

**Public method:**

`build(log) → PetriNetLog`
- Detects lifecycle logs via `_has_lifecycle_start_events()`
- If lifecycle: calls `_filter_and_annotate_lifecycle()` to remove start events
  and inject `__duration_seconds__` into each complete event
- Runs `pm4py.conformance_diagnostics_token_based_replay()`
- Filters traces below `config.replay_min_fitness`
- For each accepted trace, calls `_build_execution()` → `TraceExecution`

**Lifecycle handling (`_filter_and_annotate_lifecycle`):**
- Single-pass over each trace; `start` events are buffered by activity name
- On each `complete` event: pops the earliest matching start timestamp,
  computes `delta = complete_ts - start_ts` (negative deltas discarded)
- Attaches `delta` to the event as `__duration_seconds__` (shallow copy only;
  original log is not mutated)

**Alignment (`_align_trace`):**
- Lockstep: each labeled transition in `activated_transitions` consumes the
  next log event (attributes become a fresh `dict` copy)
- Tau transitions receive `{}`
- Override point for Approach B (optimal alignment):
  replace with `pm4py.conformance_diagnostics_alignments`

**Marking simulation (`_build_execution`):**
- Tracks a `Dict[Place, int]` marking, consuming/producing tokens per arc
- Records `from_places` (the set of input places that held tokens at firing time)
  on each `FiringStep`
- Extracts and removes `__duration_seconds__` from the event dict into
  `FiringStep.duration_seconds`

**`PetriNetLog` / `TraceExecution` / `FiringStep`** (models):

| Field | Type | Description |
|---|---|---|
| `FiringStep.activity_name` | `str` | Sanitized transition label or tau name |
| `FiringStep.is_tau` | `bool` | Whether this is a silent transition |
| `FiringStep.from_places` | `Set[Place]` | Input places that had tokens |
| `FiringStep.attributes` | `Dict[str, Any]` | Event attributes (without `__duration_seconds__`) |
| `FiringStep.duration_seconds` | `Optional[float]` | Lifecycle delta, or `None` |

---

## Step 5 — `log_preprocessor.py`

### `class LogPreprocessor`

Reads a `PetriNetLog` exactly once and materialises two indexed views of
the same data, ready for all downstream statistical and DT-based analysis.

**Constructor:**
```python
LogPreprocessor(
    config: Optional[AnalysisConfig] = None,
    discretizer: Optional[Discretizer] = None,
)
```

**Public method:**

`preprocess(pn_log: PetriNetLog, xor_splits: Dict[Place, List[Transition]]) → PreprocessedLog`

For every non-tau `FiringStep` in every `TraceExecution`:
1. Snapshots the running attribute state *before* this step → `pre_state`
2. Computes `changed_attrs` — attributes whose value differs from the last
   seen value in this trace (or appears for the first time)
3. Applies `discretizer.transform_value()` to numeric attributes in boundaries
4. Emits a `TransitionFiringData` indexed under:
   - `transition_firings[activity_name]`
   - `xor_firings[place.name]` for every XOR-split place the step crosses
5. Updates the running state with the full attribute set (not just changes),
   keeping the state in sync even when a value is rewritten unchanged

**`PreprocessedLog`** fields:

| Field | Type | Description |
|---|---|---|
| `transition_firings` | `Dict[str, List[TransitionFiringData]]` | All firings indexed by activity name |
| `xor_firings` | `Dict[str, List[TransitionFiringData]]` | Firings indexed by XOR-split place name |

**`TransitionFiringData`** fields:

| Field | Type | Description |
|---|---|---|
| `activity_name` | `str` | Sanitized activity name |
| `pre_state` | `Dict[str, Any]` | Attribute state *before* this firing |
| `changed_attrs` | `Dict[str, Any]` | Attributes that changed at this firing |
| `from_places` | `FrozenSet[Place]` | Input places at firing time |

**Design invariant:** tau steps are skipped entirely — they do not update
the running state and produce no `TransitionFiringData`.

---

## Step 6 — `probability_estimator.py`

### `class ProbabilityEstimator`

Pure counter and normalizer operating on a `PreprocessedLog`.
All sanitization and discretization has already been performed; this module
only counts and divides.

**Constructor:**
```python
ProbabilityEstimator(
    silent_transitions: Dict[Transition, str],
    config: Optional[AnalysisConfig] = None,
)
```

**Methods:**

`compute_xor_probabilities(preprocessed_log, decision_points) → Dict[str, XorSplitStats]`
- For each XOR-split place, counts branch activations from `xor_firings`
- Normalizes to probabilities; falls back to equal weights if `total == 0`
- Returns `{ place_name: XorSplitStats(probabilities, total_executions) }`

`compute_attribute_effect_probabilities(preprocessed_log) → Dict[str, AttributeEffect]`
- For each labeled transition T:
  - `P(attr X changes | T fires) = |firings where X in changed_attrs| / |firings|`
  - `P(X = v | X changes after T) = |firings where X → v| / |firings where X changed|`
- Returns `{ activity: AttributeEffect(presence_probabilities, value_probabilities, total_firings) }`

---

## Steps 7–10 — `decision_mining.py`

### `class DecisionMiner`

Mines decision logic for XOR-split routing and conditional attribute effects
using `sklearn.tree.DecisionTreeClassifier`. Operates entirely on the
`PreprocessedLog` — no raw log scanning.

**Constructor:**
```python
DecisionMiner(
    silent_transitions: Dict[Transition, str],
    config: Optional[AnalysisConfig] = None,
)
```

### Screening phase (Steps 7–8)

**`screen_xor_splits(xor_splits, xor_stats) → Dict[str, XorSplitScreening]`**

For each XOR-split place, classifies branches and assigns an `action`:

| Condition | `action` |
|---|---|
| 0 active branches | `"fallback"` |
| 1 active branch | `"deterministic"` |
| `total_samples < dt_min_samples` | `"fallback"` |
| ≥ 2 active branches, enough samples | `"dt"` |

Branch status:
- `"pruned"` if `probability < config.xor_prune_threshold`
- `"certain"` if sole remaining active branch
- `"active"` otherwise

**`screen_effects(attribute_effects) → Dict[str, TransitionScreening]`**

For each `(transition, attribute)` pair, classifies two sub-tasks:

*Appearance (2A):*

| Condition | `appearance_action` |
|---|---|
| `presence_prob < effect_never_threshold` | `"never"` |
| `presence_prob > effect_always_threshold` | `"deterministic"` |
| `total_firings < dt_min_samples` | `"fallback"` |
| otherwise | `"dt"` |

*Value (2B):*

| Condition | `value_action` |
|---|---|
| `≤ 1` distinct values | `"deterministic"` |
| `< 2` active values after pruning | `"deterministic"` |
| `estimated_value_samples < dt_min_samples` | `"fallback"` |
| otherwise | `"dt"` |

Individual values are tagged `"pruned"` / `"certain"` / `"active"` by
`effect_value_prune_threshold` and `effect_value_certain_threshold`.

### Mining phase (Steps 9–10)

**`mine_xor_splits(preprocessed_log, xor_screening) → Dict[str, XorSplitGuards]`**
- Processes only places with `screening.action == "dt"`
- Builds feature matrix from `xor_firings[place_name]`: `pre_state → X`, `activity_name → y`
- Excludes firings of pruned branches from training data
- Falls back to `screening.action = "fallback"` if:
  - feature matrix is empty or has only one class
  - DT accuracy < `config.dt_min_accuracy`
- Returns `{ place_name: XorSplitGuards(guards, total_samples, dt_accuracy) }`

**`mine_effects(preprocessed_log, effect_screening) → Dict[str, TransitionEffects]`**
- For each `(transition, attribute)` marked `"dt"`:
  - **2A (appearance)**: `y = "appears" / "not_appears"` → binary DT
  - **2B (value)**: `y = changed value` (rows where attr is `NA` excluded) → multiclass DT
  - Pruned values excluded from 2B training
- Returns `{ activity: TransitionEffects(transition_name, total_firings, effects) }`

### Feature encoding (`_train_and_extract`)

| Column type | Encoding | Guard type |
|---|---|---|
| `bool` dtype or binary `{0,1}` | cast to `int` | `Guard(attr, None, negated)` |
| `object` / string | one-hot via `pd.get_dummies` | `Guard(attr, value, negated=False)` on `>` side |
| Numeric (not binary) | **dropped** before training | no guard generated |

Discretized numeric columns (e.g. `"crp_gte_10_0_lte_97_5"`) arrive as
strings and are treated as categorical, producing `Guard("crp", "gte_10_0_lte_97_5", False)`.

### SOP guard structure

Guards are returned in **Sum-of-Products** form:

```
Dict[activity_name, List[List[Guard]]]
  outer list = OR  (different root-to-leaf paths predicting the same class)
  inner list = AND (conditions along one path)
```

### Leaf pruning (`_build_sop_guards`)

After walking all root-to-leaf paths, leaves are filtered by:
- `n_leaf >= config.dt_prune_min_leaf_samples`
- `purity >= config.dt_prune_min_purity`

Activities whose *all* leaves are pruned are handled by
`config.dt_prune_orphan_mode`:
- `"keep_best"` — retain the single highest-purity leaf for that activity (default)
- `"drop"` — remove the activity from the guards entirely

**Side effect:** `mine_xor_splits` and `mine_effects` **mutate** the
`screening` objects by updating `action` / `appearance_action` /
`value_action` to `"fallback"` when DT training fails. This ensures
`_filter_xor_splits` and `_filter_effects` see the actual post-DT outcome.

### Relevant `AnalysisConfig` fields (DT)

| Field | Meaning |
|---|---|
| `dt_min_samples` | Min samples to attempt DT (XOR or effect) |
| `dt_min_accuracy` | Min training accuracy to accept a DT |
| `dt_max_depth` | Max depth of `DecisionTreeClassifier` |
| `dt_prune_min_leaf_samples` | Min samples at a leaf to keep it |
| `dt_prune_min_purity` | Min purity at a leaf to keep it |
| `dt_prune_orphan_mode` | `"keep_best"` or `"drop"` |
| `xor_prune_threshold` | Branch probability below which it is pruned |
| `xor_statistical_mode` | `"majority_only"`, `"weighted"`, `"pruned_weighted"` |
| `effect_never_threshold` | Presence probability below which effect is ignored |
| `effect_always_threshold` | Presence probability above which effect is deterministic |
| `effect_value_prune_threshold` | Value probability below which value is pruned |
| `effect_value_certain_threshold` | Value probability above which value is certain |
| `probability_min_samples` | Min firings/samples for statistical reliability |

---

## Independent module — `temporal_extractor.py`

### `class TemporalExtractor`

Extracts action duration statistics for durative PDDL actions. **Not** called
by `Parser.__init__()`; used separately when temporal planning is required.

**Three extraction strategies:**

| Strategy | Method | Source |
|---|---|---|
| Lifecycle | `extract_from_lifecycle_pn(pn_log)` | `FiringStep.duration_seconds` (from lifecycle annotation) |
| Inter-event | `estimate_from_inter_event_times_pn(pn_log)` | `timestamp[i+1] - timestamp[i]` between consecutive labeled steps |
| External | `from_external(durations)` | User-supplied `ExternalDuration` bounds |

**Main entry point:**

`extract(external_durations, fallback_to_inter_event, petri_net_log) → Dict[str, ActionDurationStats]`

Priority per activity: **external > lifecycle > inter-event**.
Activities not covered by any strategy are omitted.

**Duration formula** (for log-derived strategies):

```
effective_min = max(observed_min, mean - std_dev)
effective_max = min(observed_max, mean + std_dev)
```

Falls back to `(observed_min, observed_max)` if `effective_min > effective_max`.

**Dataclasses:**

`ExternalDuration(min_duration: float, max_duration: float)`
— validated: `min_duration <= max_duration`

`ActionDurationStats(effective_min, effective_max, source, mean?, std_dev?, observed_min?, observed_max?, count?)`
— statistical fields are `None` for `source='external'`

---

## Test coverage summary

| Test file | Scope | Strategy |
|---|---|---|
| `test_parser.py` | `Parser` integration | Module-scoped fixture; 120-trace synthetic log |
| `test_parser.py::TestFilterEffectsLevel3` | `_filter_effects` | Unit tests via `object.__new__(Parser)` stub |
| `test_decision_mining.py` | `DecisionMiner` (all methods) | Synthetic `PreprocessedLog` / `TransitionFiringData` |
| `test_discretizer.py` | `Discretizer.fit` + `transform_value` | Synthetic pm4py `EventLog` objects in-memory |
| `test_log_preprocessor.py` | `LogPreprocessor.preprocess` | Synthetic `PetriNetLog` |
| `test_probability_estimator.py` | `ProbabilityEstimator` | Synthetic `PreprocessedLog` |
| `test_petri_net_log_builder.py` | `PetriNetLogBuilder` | Synthetic pm4py log + replay |
| `test_model_discoverer.py` | `ModelDiscoverer` structural helpers | Synthetic net arcs |
| `test_temporal_extractor.py` | `TemporalExtractor` (all three strategies) | Synthetic `PetriNetLog` |

Integration tests are marked `@pytest.mark.integration` and skipped by default.
Run with: `conda run -n temporalXes2plan python -m pytest -m integration`

---

## Snapshot output (`_save_snapshot`)

Written to `config.snapshot_dir` before any branch pruning:

| File | Format | Contents |
|---|---|---|
| `petri_net.pnml` | PNML (XML) | Original discovered Petri net |
| `xor_screening.json` | JSON | `XorSplitScreening.to_dict()` per place |
| `effect_screening.json` | JSON | `TransitionScreening.to_dict()` per activity |

The directory is created if it does not exist; existing files are overwritten.
