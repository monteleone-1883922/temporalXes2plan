import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import pm4py
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set, Union
import warnings
import os
import logging
from tqdm import tqdm
import core_utils as utils

logger = utils.get_logger(__name__)


DEFAULT_CONFIG = {
    'min_instances': 100,  # minimum number of occurrences for an activity to be considered a decision point
    'max_depth': 2,    # depth of the decision tree
    'min_samples_leaf': 150,  # minimum number of samples required in a leaf node
    'top_features': 4  # Number of top features to select globally (None for all)
}

META_COLUMNS_IGNORED = [
    'case:concept:name', 
    'concept:name', 
    'time:timestamp',
    'org:resource',
    'lifecycle:transition',
    'org:group',
    'variant-index',
    'Resource',
    'org:role'
]

OPTIONAL_DATA_ACTIVITIES_TO_IGNORE = []
all_categorical_cols = set()
all_bool_cols = set()


def get_base_feature(col_name: str) -> str:
    """
    Find the base feature name from a potentially encoded column name.
    Useful for mapping one-hot encoded or discretized columns back to their original attribute.

    Args:
        col_name: The name of the column (e.g., 'attribute_value', 'attr_True').

    Returns:
        The base attribute name in lowercase.
    """
    col_name_lower = col_name.lower()
    if col_name_lower.endswith("_false") or col_name_lower.endswith("_true"):
        base = col_name.rsplit('_', 1)[0]
        return base.lower() if base else base
    parts = col_name.split('_')
    if len(parts) > 1:
        # Try progressively shorter prefixes to find the base categorical attribute
        for i in range(1, len(parts)):
            base = "_".join(parts[:-i])
            if base.lower() in all_categorical_cols:
                return base.lower()
    
    logger.debug(f"Could not resolve base feature for column: {col_name}, defaulting to lowercase name")
    return col_name_lower

def extract_simple_guards(tree: DecisionTreeClassifier, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extract simple logical guards (conditions) from a trained decision tree.

    Args:
        tree: The trained scikit-learn DecisionTreeClassifier.
        feature_names: List of feature names corresponding to the tree's input.

    Returns:
        A list of dictionaries, each containing an 'activity' and its 'guard' condition string.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    # function to recursively read conditions of decision tree for each activity
    # Example tree:

    #       amount <= 500?
    #       /           \
#   priority > 3?    "Reject"
#    /        \
# "Approve"  "Review"
#
# would yield:
#     [
#         ("Approve", [("amount", "<=", 500), ("priority", ">", 3)]),
#         ("Review", [("amount", "<=", 500), ("priority", "<=", 3)]),
#         ("Reject", [("amount", ">", 500)])
#     ]

    def recurse(node, conditions):
        """Recursively walks the tree, building conditions with thresholds."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], conditions + [(name, ">", threshold)])
        else:
            value = tree_.value[node]
            activity_index = np.argmax(value)
            activity_name = tree.classes_[activity_index]
            paths.append((activity_name, conditions))

    recurse(0, [])
    # Example of guards content:
    # {
    #     'activity': 'Approve',
    #     'guard': '(amount lte_500) and (priority gte_3) and (urgent)'
    # }
    guards = []
    for activity, conds in paths:

        guard_conditions = []
        numeric_conds = []
        numeric_per_base = defaultdict(list)
        for name, op, thresh in conds:
            base = get_base_feature(name)
            sanitized_base = utils.sanitize_name(base)
            if name.endswith("_True"):
                guard_conditions.append(f"({sanitized_base})")
            elif name.endswith("_False"):
                guard_conditions.append(f"(not ({sanitized_base}))")
            elif base in all_categorical_cols:
                value = name[len(sanitized_base) + 1:]
                guard_conditions.append(f"({sanitized_base} e{value})") #TODO use = instead?
            else:
                # for bool converted to integers 0/1
                if base in all_bool_cols:
                    if op == "<=":
                        guard_conditions.append(f"(not ({sanitized_base}))")
                    elif op == ">":
                        guard_conditions.append(f"({sanitized_base})")
                    else:
                        guard_conditions.append(f"({sanitized_base})")  # fallback
                else:
                    numeric_conds.append((sanitized_base, op, thresh))
                    numeric_per_base[base].append((op, thresh))
        
        # Process numeric conditions into intervals
        for base, cond_list in numeric_per_base.items():
            lower = float('-inf')
            upper = float('inf')
            for op, thresh in cond_list:
                if op == "<=":
                    upper = min(upper, thresh)
                elif op == ">":
                    lower = max(lower, thresh)
            
            if lower != float('-inf'):
                lower = round(lower, 2)
            if upper != float('inf'):
                upper = round(upper, 2)
            
            if lower == float('-inf') and upper != float('inf'):
                interval_str = f"(-inf-{upper}]"
            elif lower != float('-inf') and upper == float('inf'):
                interval_str = f"({lower}-inf)"
            elif lower != float('-inf') and upper != float('inf'):
                interval_str = f"({lower}-{upper}]"
            else:
                interval_str = "(-inf-inf)"
            
            converted = utils.discretize_value(None, interval_str)
            guard_conditions.append(f"({base} {converted})")
        if guard_conditions:
            guard = " and ".join(guard_conditions) 
            guards.append({'activity': activity, 'guard': guard})
    
    return guards

def load_and_prepare_log(
    log_path: str, 
    meta_cols: List[str], 
    log: Optional[Any] = None
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    """Loads log, auto-detects features, and prepares DataFrame.
    
    Args:
        log_path: Path to the XES log file (used if log is None)
        meta_cols: Columns to ignore
        log: Optional pre-loaded pm4py log object (already transformed with Activity classifier if needed)
    """
    if log is None:
        logger.info(f"Loading log: {log_path}")
        log = pm4py.objects.log.importer.xes.importer.apply(log_path)
    else:
        logger.info(f"Using pre-loaded log")
    df = pm4py.convert_to_dataframe(log)

    #add column next activity for every activity
    df['next_activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)
    all_cols = set(df.columns)
    meta_cols_present = set(meta_cols) | {'next_activity', 'Activity', 'concept:name'}
    
    # Also filter out activity names which shouldn't be used for decision mining
    all_feature_cols = [col for col in all_cols
                       if not col.startswith('case:variant') and col not in meta_cols_present]
    # obtain columns by datatype
    numeric_cols = set(df[all_feature_cols].select_dtypes(include=np.number).columns)
    bool_cols = list(df[all_feature_cols].select_dtypes(include='bool').columns)
    object_cols = set(df[all_feature_cols].select_dtypes(include='object').columns)

    all_feature_cols = set(all_feature_cols)

    for col in bool_cols:
        if col in df.columns and col in all_feature_cols:
            df[col] = df[col].map({True: 1, False: 0})
    final_bool_cols = set(bool_cols)
    # tries to convert to bool columns all columns with only 2 or less categorical values if they contain boolean values
    for col in object_cols:
        unique_vals_no_na = df[col].dropna().unique()
        if len(unique_vals_no_na) <= 2 and all(str(v).lower() in ['true', 'false', '1', '0'] for v in unique_vals_no_na):
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0, 
                                   '1': 1, '0': 0, 1: 1, 0: 0})
            final_bool_cols.add(col)
        # Ignore columns with always same value
        elif len(unique_vals_no_na) < 2 and col in all_feature_cols:
            all_feature_cols.remove(col)
            logger.info(f"Ignoring column with single value: {col}")

             
    # Try to convert object columns that are numeric to numeric
    for col in list(object_cols):
        if col not in final_bool_cols and col in all_feature_cols:
            try:
                temp = pd.to_numeric(df[col], errors='coerce')
                if temp.notna().sum() > 0 and temp.dtype in [np.int64, np.float64]:
                    nan_count = temp.isna().sum()
                    if nan_count > len(temp) * 0.5:
                        logger.warning(f"Conversion of column {col} to numeric resulted in {nan_count}/{len(temp)} NaNs. Data might be noisy.")
                    df[col] = temp
                    numeric_cols.add(col)
                    object_cols.remove(col)
            except:
                continue
             
    categorical_cols = [c for c in all_feature_cols if c not in numeric_cols and c not in final_bool_cols]

    logger.info(f"Auto-detected features: Numeric: {len(numeric_cols)}, Boolean: {len(final_bool_cols)}, Categorical: {len(categorical_cols)}")
    logger.debug(f"Numeric: {numeric_cols}")
    logger.debug(f"Boolean: {final_bool_cols}")
    logger.debug(f"Categorical: {categorical_cols}")
    all_feature_cols = list(all_feature_cols)
    if not all_feature_cols:
        logger.warning(f"No features detected in log {log_path} after filtering.")
    df[all_feature_cols] = df.groupby('case:concept:name')[all_feature_cols].ffill()
    return df, all_feature_cols, list(numeric_cols), list(final_bool_cols), list(categorical_cols)

def find_decision_points(df: pd.DataFrame, min_instances: int) -> List[str]:
    """
    Identify activities that act as decision points in the process.

    Args:
        df: Prepared DataFrame containing 'concept:name' and 'next_activity'.
        min_instances: Minimum frequency for an activity to be considered.

    Returns:
        List of activity names that are decision points.
    """
    """Finds all activities with more than one unique next step."""
    if 'next_activity' not in df.columns or 'concept:name' not in df.columns:
        return []
    decision_counts = df.groupby('concept:name')['next_activity'].nunique()
    potential_decision_points = decision_counts[decision_counts > 1].index.tolist()
    instance_counts = df['concept:name'].value_counts()
    final_decision_points = [
        p for p in potential_decision_points 
        if instance_counts.get(p, 0) >= min_instances
    ]
    logger.info(f"Found {len(final_decision_points)} potential decision points (>{min_instances} instances): {final_decision_points}")
    return final_decision_points

def run_analysis_for_decision_point(
    df: pd.DataFrame, 
    decision_point: str, 
    all_features: List[str], 
    booleans: List[str], 
    categoricals: List[str], 
    config: Dict[str, Any], 
    global_top_features: Optional[List[str]] = None, 
    skip_final_training: bool = False
) -> Tuple[Optional[Dict[str, List[str]]], Dict[str, Set[str]], List[Tuple[str, float]]]:
    """
    Analyzes a single decision point and extracts decision rules.
    
    Returns:
        grouped_guards: Dictionary mapping activities to lists of guard conditions
        value_types: Dictionary mapping attributes to sets of value types used (e.g., 'lte_28_5', 'gte_57_5')
        sorted_features: List of (feature, importance) tuples sorted by importance
    """
    df_decision = df[df["concept:name"] == decision_point].copy()
    # Remove final activities
    df_decision = df_decision.dropna(subset=['next_activity'])
    # Remove activities considered as noise
    heuristic_noise = [c for c in all_features if c in df_decision['next_activity'].unique()]
    all_noise = set(heuristic_noise) | set(OPTIONAL_DATA_ACTIVITIES_TO_IGNORE)
    df_decision = df_decision[~df_decision['next_activity'].isin(all_noise)]
    
    if len(df_decision) < config['min_instances']:
        logger.warning(f"  Skipping {decision_point}: Not enough valid instances after filtering (found {len(df_decision)}).")
        return None, {}, []
    outcomes = df_decision['next_activity'].nunique()
    if outcomes < 2:
        logger.warning(f"  Skipping {decision_point}: Only one outcome ({df_decision['next_activity'].unique()}) after filtering.")
        return None, {}, []
        
    logger.info(f"Analyzing {decision_point}: {len(df_decision)} instances, {outcomes} unique outcomes, min_samples_leaf: {config['min_samples_leaf']}")
    dynamic_min_samples = config['min_samples_leaf']

    X = df_decision[all_features]
    y = df_decision['next_activity'].astype(str)
    default_activity = y.mode()[0]
    for col in booleans:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)
    X_encoded = pd.get_dummies(X.fillna(0), columns=categoricals, dummy_na=False)
    # Sanitize column names to avoid issues with spaces and special characters
    X_encoded = X_encoded.rename(columns={col: utils.sanitize_name(col) for col in X_encoded.columns})
    encoded_feature_names = list(X_encoded.columns)
    
    clf = DecisionTreeClassifier(
        max_depth=config['max_depth'], 
        min_samples_leaf=dynamic_min_samples,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_encoded, y)
    # Collect importance of each feature for this decision point
    importances = clf.feature_importances_
    feature_importance_dict = dict(zip(encoded_feature_names, importances))
    base_importance = defaultdict(float)
    for name, imp in feature_importance_dict.items():
        base = get_base_feature(name)
        base_importance[base] += imp
    
    if importances.sum() == 0:
        logger.warning(f"Decision point {decision_point} result in 0 feature importance for all features.")
    
    sorted_features = sorted(base_importance.items(), key=lambda x: x[1], reverse=True)
    
    if skip_final_training:
        return None, {}, sorted_features
    
    if config['top_features'] is not None and config['top_features'] > 0:
        available_features = list(set(get_base_feature(f) for f in encoded_feature_names))
        if global_top_features is not None:
            top_features = [f for f in global_top_features if f in available_features]
            logger.info(f"  Selecting top {len(top_features)} features from global: {top_features}")
        else:
            num_available = len(available_features)
            if config['top_features'] > num_available:
                logger.warning(f"  Requested {config['top_features']} top features, but only {num_available} available. Using all.")
                top_features = available_features
            else:
                top_features = [f[0] for f in sorted_features[:config['top_features']]]
            logger.info(f"  Selecting top {len(top_features)} features: {top_features}")
        
        X_filtered = X_encoded[[col for col in encoded_feature_names if get_base_feature(col) in top_features]]
        
        # Retrain with filtered features
        clf = DecisionTreeClassifier(
            max_depth=config['max_depth'], 
            min_samples_leaf=dynamic_min_samples,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_filtered, y)
        encoded_feature_names = list(X_filtered.columns)
    
    guards = extract_simple_guards(clf, encoded_feature_names)
    if not guards:
        logger.debug(f"No guards extracted for decision point {decision_point} (tree might be root only).")

    grouped_guards = defaultdict(list)
    
    # Extract value types from guards instead of using split points
    # This ensures we only create types that are actually used in the decision rules
    value_types = defaultdict(set)
    for g in guards:

        guard = g['guard']
        grouped_guards[g['activity']].append(guard)
        # Parse conditions to extract value types
        conditions = guard.split(" and ")
        for cond in conditions:
            cond = cond.strip()
            if cond.startswith('(') and cond.endswith(')'):
                inner = cond[1:-1]
                # Check for numeric value types (lte_X, gte_X, gte_X_lte_Y)
                parts = inner.split(' ')
                if len(parts) == 2:
                    attr, value_type = parts
                    # Only add numeric value types (not categorical or boolean)
                    if value_type.startswith('lte_') or value_type.startswith('gte_'):
                        value_types[attr].add(value_type)

    return grouped_guards, value_types, sorted_features

def discover_all_decision_rules(
    log_path: str, 
    config_overrides: Dict[str, Any] = {}, 
    log: Optional[Any] = None
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[float]], List[Dict[str, Union[str, Set[str]]]]]:
    """
    Main entry point. Loads a log, runs mining on all decision points,
    and returns a dictionary of all rules and consolidated intervals.
    
    Args:
        log_path: Path to the XES log file
        config_overrides: Optional config overrides
        log: Optional pre-loaded pm4py log object (already transformed with Activity classifier if needed)
    
    Returns:
        all_decision_rules: Dictionary mapping decision points to their guards
        all_intervals: Dictionary mapping attributes to lists of split points (thresholds)
                      consolidated from all decision trees to create non-overlapping intervals
        samples: List of sample preconditions for each activity
    """
    global all_categorical_cols, all_bool_cols
    config = DEFAULT_CONFIG.copy()
    config.update(config_overrides)
    
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    all_decision_rules = {} 
    samples = [] 

    try:
        df, all_features, numericals, booleans, categoricals = load_and_prepare_log(
            log_path, 
            META_COLUMNS_IGNORED,
            log=log
        )

        all_categorical_cols = {utils.sanitize_name(c) for c in categoricals}
        all_bool_cols = {utils.sanitize_name(b) for b in booleans}

        decision_points = find_decision_points(df, config['min_instances'])        
        if not decision_points:
            logger.warning("No decision points found meeting the criteria.")
            return {}, {}, []

        # First pass: collect feature importances globally
        global_feature_importance = defaultdict(float)
        for i, point in enumerate(tqdm(decision_points, desc="Decision Mining (Pass 1: Feature Importance)")):
            formatted_point = utils.sanitize_name(point)
            logger.info(f"ANALYZING DECISION POINT {i+1}/{len(decision_points)}: 'exec_{formatted_point}' (collecting features)")
            
            _, _, sorted_features = run_analysis_for_decision_point(
                df, point, 
                all_features, booleans, categoricals,
                config, skip_final_training=True
            )
            
            for f, score in sorted_features:
                global_feature_importance[f] += score
        
        # Select global top features
        global_top_features_sorted = sorted(global_feature_importance.items(), key=lambda x: x[1], reverse=True)
        global_top_features = [f for f, s in global_top_features_sorted[:config['top_features']]]
        logger.info(f"Global top {len(global_top_features)} features selected: {global_top_features}")

        # Second pass: train with global top features
        all_decision_rules = {} 
        all_value_types = defaultdict(set)
        for i, point in enumerate(tqdm(decision_points, desc="Decision Mining (Pass 2: Training)")):
            formatted_point = utils.sanitize_name(point)
            logger.info(f"ANALYZING DECISION POINT {i+1}/{len(decision_points)}: 'exec_{formatted_point}'")
            
            rules_for_point, value_types_for_point, _ = run_analysis_for_decision_point(
                df, point, 
                all_features, booleans, categoricals,
                config, global_top_features=global_top_features, skip_final_training=False
            )
            
            if rules_for_point:
                all_decision_rules[point] = rules_for_point
                for feature, types in value_types_for_point.items():
                    all_value_types[feature].update(types)
                
                # Add samples
                current_formatted = 'exec_' + utils.sanitize_name(point) #point is an activity not a place
                for activity_name, guards_list in rules_for_point.items():
                    formatted_activity = 'exec_' + utils.sanitize_name(activity_name)
                    for guard in guards_list:
                        conditions = guard.split(" and ")
                        preconditions = set(conditions) | {f"(completed {current_formatted})"}
                        samples.append({'activity': formatted_activity, 'preconditions': preconditions})
            
        logger.info("Decision mining complete.")
        
        # Convert value types to split points for non-overlapping interval generation
        all_intervals = defaultdict(list)
        value_type_mappings = {}
        
        for attr, value_types_set in all_value_types.items():
            split_points = set()
            for vtype in value_types_set:
                # Extract numeric thresholds from value type names
                # e.g., 'lte_28_5' -> 28.5, 'gte_147_5' -> 147.5, 'gte_28_5_lte_147_5' -> 28.5, 147.5
                parts = vtype.replace('gte_', '').replace('lte_', '').split('_')
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):
                        try:
                            threshold = float(f"{parts[i]}.{parts[i+1]}")
                            split_points.add(threshold)
                        except ValueError:
                            pass
            
            all_intervals[attr] = sorted(split_points)
            
            # Build mapping from old value types to new consolidated intervals
            value_type_mappings[attr] = {}
            for old_vtype in value_types_set:
                matching_intervals = _find_matching_intervals(old_vtype, all_intervals[attr])
                value_type_mappings[attr][old_vtype] = matching_intervals
        
        # Remap guards in samples to use consolidated intervals
        remapped_samples = _remap_samples_to_consolidated_intervals(samples, value_type_mappings)
        # all_decision_rules example:
        """{
            'Review_Application': {
                'Approve': [
                    '(amount lte_500_0) and (credit_score gte_700_0)',
                    '(amount lte_200_0)'
                ],
                'Reject': [
                    '(amount gte_500_0) and (not (has_guarantor))'
                ],
                'Request_Info': [
                    '(amount gte_500_0) and (has_guarantor) and (credit_score lte_700_0)'
                ]
            },
            'Check_Documents': {
                'Accept': [
                    '(credit_score gte_600_0) and (urgent)'
                ],
                'Return': [
                    '(credit_score lte_600_0)'
                ]
            }
        }"""
        # all_intervals example:
        """{
            'amount': [200.0, 500.0],
            'credit_score': [600.0, 700.0]
        }"""
        # remapped_samples example:
        """[
            {
                'activity': 'exec_Approve',
                'preconditions': {
                    '(amount lte_200_0)',
                    '(completed exec_Review_Application)'
                }
            },
            {
                'activity': 'exec_Approve',
                'preconditions': {
                    '(amount gte_200_0_lte_500_0)',
                    '(completed exec_Review_Application)'
                }
            }
        ]"""
        return all_decision_rules, all_intervals, remapped_samples

    except FileNotFoundError:
        logger.error(f"Log file not found at '{log_path}'")
        return {}, {}, []
    except Exception as e:
        logger.error(f"An unexpected error occurred in decision mining: {e}")
        return {}, {}, []

def _find_matching_intervals(old_value_type: str, thresholds: List[float], thresholds_sort: bool = False) -> List[str]:
    """
    Find which consolidated intervals match the semantics of an old value type.
    old value type has to be a float converted to string, this doesn't work on integers
    
    For example, if old_value_type is 'gte_57_5' and thresholds are [28.5, 40.5, 57.5, 147.5],
    it should match both 'gte_57_5_lte_147_5' and 'gte_147_5'.
    """
    if not thresholds:
        return [old_value_type]
    
    matching = []
    
    # Parse the old value type
    is_lower_bound = 'gte_' in old_value_type
    is_upper_bound = 'lte_' in old_value_type
    #TODO: tieni a mente che gli intervalli sono contigui quindi forse per quello legge solo il primo
    # check that all numbers are float
    # Extract the threshold value(s)
    parts = old_value_type.replace('gte_', '').replace('lte_', '').split('_')
    try:
        if len(parts) >= 2:
            threshold_val = float(f"{parts[0]}.{parts[1]}")
        else:
            logger.warning(f"Could not extract threshold value from type: {old_value_type}")
            return [old_value_type]
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing threshold from {old_value_type}: {e}")
        return [old_value_type]

    upper_threshold_val = None

    if is_lower_bound and is_upper_bound:
        try:
            upper_threshold_val = float(f"{parts[2]}.{parts[3]}")
        except:
            pass


    thresholds_sorted = sorted(thresholds) if thresholds_sort else thresholds
    
    # Generate consolidated interval names
    for i, thresh in enumerate(thresholds_sorted):
        thresh_str = str(thresh).replace('.', '_').replace('-', 'neg')
        
        # First interval: lte_X
        if i == 0:
            interval_name = f'lte_{thresh_str}'
            # Check if this interval matches the old value type
            if is_upper_bound and not is_lower_bound:
                # Old type is lte_Y, matches if thresh >= threshold_val
                if thresh >= threshold_val:
                    matching.append(interval_name)
            #  Fixme: add logs, this should never happen
            elif not is_upper_bound and not is_lower_bound:
                logger.warning(f"Value type {old_value_type} has neither gte nor lte. This should not happen.")
                matching.append(interval_name)
        
        # Middle intervals: gte_X_lte_Y
        if i < len(thresholds_sorted) - 1:
            lower_str = thresh_str
            upper_str = str(thresholds_sorted[i + 1]).replace('.', '_').replace('-', 'neg')
            interval_name = f'gte_{lower_str}_lte_{upper_str}'
            
            # Check if this interval matches
            if is_lower_bound and not is_upper_bound:
                # Old type is gte_Y, matches if thresh >= threshold_val
                if thresh >= threshold_val:
                    matching.append(interval_name)
            elif not is_lower_bound and is_upper_bound:
                # Old type is lte_Y, matches if upper threshold <= threshold_val
                if thresholds_sorted[i + 1] <= threshold_val:
                    matching.append(interval_name)
            elif is_lower_bound and is_upper_bound:
                # Old type is gte_X_lte_Y, check overlap
                if not upper_threshold_val or \
                    (thresh >= threshold_val and thresholds_sorted[i + 1] <= upper_threshold_val):
                    matching.append(interval_name)
        
        # Last interval: gte_X
        if i == len(thresholds_sorted) - 1:
            interval_name = f'gte_{thresh_str}'
            # Check if this interval matches
            if is_lower_bound:
                # Old type is gte_Y, matches if thresh >= threshold_val
                if thresh >= threshold_val:
                    matching.append(interval_name)
    
    return matching if matching else [old_value_type]

def _remap_samples_to_consolidated_intervals(
    samples: List[Dict[str, Union[str, Set[str]]]],
    value_type_mappings: Dict[str, Dict[str, List[str]]]
) -> List[Dict[str, Union[str, Set[str]]]]:
    """
    Remap guards in samples to use consolidated non-overlapping intervals.
    If a guard maps to multiple intervals, create duplicate samples for each.
    """
    remapped_samples = []
    
    for sample in tqdm(samples, desc="Remapping samples to consolidated intervals"):
        activity = sample['activity']
        preconditions = sample['preconditions']
        
        # Find attributes with multiple matching intervals
        attr_alternatives = {}
        fixed_preconditions = []
        
        for precond in preconditions:
            if precond.startswith('(completed '):
                fixed_preconditions.append(precond)
                continue
            
            # Parse the precondition
            if precond.startswith('(') and precond.endswith(')'):
                inner = precond[1:-1]
                parts = inner.split(' ')
                
                if len(parts) == 2:
                    attr, value_type = parts
                    # Check if this needs remapping
                    if attr in value_type_mappings and value_type in value_type_mappings[attr]:
                        matching_intervals = value_type_mappings[attr][value_type]
                        if len(matching_intervals) > 1:
                            # Multiple intervals match - store for creating variants
                            # works because no attr appear more than once in the condition for a branch of the decision tree
                            attr_alternatives[attr] = matching_intervals
                        elif len(matching_intervals) == 1:
                            # Single match - remap directly
                            fixed_preconditions.append(f'({attr} {matching_intervals[0]})')
                        else:
                            # No match - keep original
                            fixed_preconditions.append(precond)
                    else:
                        fixed_preconditions.append(precond)
                else:
                    fixed_preconditions.append(precond)
            else:
                fixed_preconditions.append(precond)
        
        # If there are alternatives, create a sample for each combination
        if attr_alternatives:
            # For now, create a sample for each alternative of the first attribute
            # This is a simplification - full expansion would create all combinations

            for attr, new_thresholds_precond in attr_alternatives.items():
                logger.debug(f"Sample for activity {activity} duplicated into {len(new_thresholds_precond)} variants due to overlapping interval {attr}")
                for alt_value in new_thresholds_precond:
                    new_preconditions = set(fixed_preconditions)
                    new_preconditions.add(f'({attr} {alt_value})')
                    remapped_samples.append({
                        'activity': activity,
                        'preconditions': new_preconditions
                    })
        else:
            remapped_samples.append({
                'activity': activity,
                'preconditions': set(fixed_preconditions)
            })
    
    return remapped_samples

def get_attribute_domains(
    samples: List[Dict[str, Union[str, Set[str]]]],
    intervals: Optional[Dict[str, List[float]]] = None
) -> Dict[str, Set[Union[str, bool]]]:
    """
    Get for each attribute in the preconditions the set of possible values.
    
    Args:
        samples: List of samples with preconditions
        intervals: Optional dict mapping attributes to lists of split points (thresholds)
                  If provided, will generate non-overlapping interval types instead of
                  extracting overlapping types from samples
    """
    attr_values = defaultdict(set)
    
    # If intervals are provided, generate non-overlapping interval types
    if intervals:
        for attr, thresholds in intervals.items():
            if not thresholds:
                continue
            thresholds = sorted(thresholds)
            
            # Generate non-overlapping intervals
            # First interval: (-inf, threshold[0]]
            threshold_str = str(thresholds[0]).replace('.', '_').replace('-', 'neg')
            attr_values[attr].add(f'lte_{threshold_str}')
            
            # Middle intervals: (threshold[i-1], threshold[i]]
            for i in range(1, len(thresholds)):
                lower_str = str(thresholds[i-1]).replace('.', '_').replace('-', 'neg')
                upper_str = str(thresholds[i]).replace('.', '_').replace('-', 'neg')
                attr_values[attr].add(f'gte_{lower_str}_lte_{upper_str}')
            
            # Last interval: (threshold[-1], +inf)
            threshold_str = str(thresholds[-1]).replace('.', '_').replace('-', 'neg')
            attr_values[attr].add(f'gte_{threshold_str}')
    
    # Extract other (non-interval) attributes from samples
    for sample in samples:
        for precond in sample['preconditions']:
            if precond.startswith('(completed '):
                continue
            # Remove outer ()
            inner = precond[1:-1]
            if inner.startswith('not ('):
                attr = inner[5:-1]
                # Skip if this attr has intervals (already processed above)
                if not intervals or not attr in intervals:
                    # add value for boolean attr
                    if attr in all_bool_cols:
                        attr_values[attr].add(True)
                        attr_values[attr].add(False)
                    else:
                        attr_values[attr].add(False)
            elif ' = ' in inner:
                attr, value = inner.split(' = ', 1)
                if not intervals or not attr in intervals:
                    attr_values[attr].add(value)
            elif inner in all_bool_cols:
                attr = inner
                if not intervals or not attr in intervals:
                    attr_values[attr].add(True)
                    attr_values[attr].add(False)
            else:
                # numerical or other
                parts = inner.split(' ', 1)
                if len(parts) == 2:
                    attr, value = parts
                    # Skip if this attr has intervals (already processed above)
                    if not intervals or not attr in intervals:
                        attr_values[attr].add(value)


    return attr_values


if __name__ == "__main__":
    TEST_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "sepsis.xes")
    print("*"*70)
    print("Running decision mining test...")
    print(f"Log file: {TEST_LOG_FILE}")
    print("*"*70)
    
    rules, intervals, samples = discover_all_decision_rules(TEST_LOG_FILE)
    print(f"Rules: {rules}")
    print(f"Intervals: {intervals}")
    print(f"Samples: {samples}")
    
    domains = get_attribute_domains(samples)
    print("Attribute domains:")
    for attr, vals in domains.items():
        print(f"  {attr}: {vals}")

