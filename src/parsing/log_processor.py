
import copy
from tqdm import tqdm
from typing import Dict, Any, Tuple, Set
import pm4py
import core_utils as utils
from parsing.csv_loader import csv_to_event_log, load_mapping_file

logger = utils.get_logger(__name__)

class LogProcessor:
    """
    Handles event log loading, filtering, train/test splitting, 
    and attribute categorization.
    """
    IGNORED_ATTRIBUTES = {'case:concept:name', 'concept:name', 'time:timestamp', 'lifecycle:transition', 'org:resource', 'org:group', 'variant-index', 'Resource', 'org:role'}

    def __init__(self, log_path: str, use_activity_classifier: bool = False) -> None:
        """
        Initialize the LogProcessor.

        Args:
            log_path: Path to the XES event log file.
            use_activity_classifier: Whether to combine concept:name and lifecycle:transition as activity names.
        """
        self.log_path = log_path
        self.use_activity_classifier = use_activity_classifier

    def load_and_filter_log(self, coverage_percentage: float) -> Tuple[Any, Any, Any]:
        """
        Load the event log (XES or CSV), apply classifiers, and filter by coverage.

        Args:
            coverage_percentage: Minimum cumulative coverage for variant filtering.

        Returns:
            Tuple containing:
            - log: The loaded and filtered event log.
            - full_log: The full event log (pre-filtering, after classifiers).
            - full_lifecycle_log: Original event log (before filtering for complete events).
        """
        if self.log_path.lower().endswith(".csv"):
            log = self._load_csv_log()
        else:
            log = pm4py.objects.log.importer.xes.importer.apply(self.log_path)
        full_lifecycle_log = copy.deepcopy(log)

        if self.use_activity_classifier:
            # When using Activity classifier, combine concept:name + lifecycle:transition
            # This is needed for logs where events are not uniquely identified by concept:name alone
            for trace in tqdm(log, desc="Applying Activity classifier"):
                for event in trace:
                    lifecycle = event.get('lifecycle:transition', '')
                    if lifecycle:
                        concept_name = event.get('concept:name', '')
                        # Combine as "ActivityName (Lifecycle)" - e.g., "Accepted (In Progress)"
                        event['concept:name'] = f"{concept_name} ({lifecycle})"
            logger.info("Using Activity classifier: combining concept:name + lifecycle:transition")
        else:
            # Filter full log for lifecycle (keep only 'complete' events)
            try:
                lifecycle_values = pm4py.get_event_attribute_values(log, 'lifecycle:transition')
                if 'complete' in lifecycle_values:
                    n_traces_before = len(log)
                    events_before = sum(len(t) for t in log)
                    log = pm4py.filter_event_attribute_values(log, 'lifecycle:transition', ['complete'])
                    n_traces_after = len(log)
                    events_after = sum(len(t) for t in log)
                    logger.info(
                        "Lifecycle filter (kept 'complete' events only): "
                        "%d traces kept (%d removed), %d events kept (%d removed)",
                        n_traces_after, n_traces_before - n_traces_after,
                        events_after, events_before - events_after,
                    )
            except Exception:
                pass

        full_log = log  # Store full log for frequency computation
        
        try:
            n_before_coverage = len(log)
            filtered_log = pm4py.filter_variants_by_coverage_percentage(log, coverage_percentage)
            if len(filtered_log) > 0:
                log = filtered_log
                removed = n_before_coverage - len(log)
                logger.info(
                    "Variant coverage filter (threshold=%.2f): %d traces kept, "
                    "%d traces removed (%.1f%%)",
                    coverage_percentage, len(log), removed,
                    100.0 * removed / n_before_coverage if n_before_coverage else 0.0,
                )
            else:
                logger.warning("Warning: Coverage filter removed all traces, using full log")
        except Exception as e:
            logger.warning(f"Warning: Coverage filtering failed: {e}, using full log")
            
        return log, full_log, full_lifecycle_log

    def _load_csv_log(self) -> Any:
        """Load a CSV event log using the companion .mapping.json file."""
        mapping = load_mapping_file(self.log_path)
        return csv_to_event_log(self.log_path, mapping)

    def initialize_attributes(self, full_log: Any) -> Tuple[Set[str], Dict[str, str]]:
        """
        Extract and categorize event and trace attributes.

        Args:
            full_log: The full event log to analyze attributes from.

        Returns:
            Tuple of (attributes_set, attribute_categories_dict).
        """
        attributes = {attr for attr in pm4py.get_event_attributes(full_log)
                      if attr not in self.IGNORED_ATTRIBUTES}
        attribute_categories = self.categorize_attributes(full_log)
        return attributes, attribute_categories

    def categorize_attributes(self, full_log: Any) -> Dict[str, str]:
        """
        Categorize attributes into 'boolean', 'numerical', or 'categorical'.

        Args:
            full_log: The full event log to analyze.

        Returns:
            Dictionary mapping attribute names to categories.
        """
        attr_categories = {}
        
        # Include trace (case) attributes
        for attr in pm4py.get_trace_attributes(full_log):
            sanitized_name = f"case:{attr}"
            values = [trace.attributes[attr] for trace in tqdm(full_log, desc=f"Categorizing trace attribute {attr}", leave=False) if attr in trace.attributes]
            if values:
                attr_categories[sanitized_name] = 'boolean' \
                    if all(isinstance(val, bool) for val in values) \
                    else 'numerical' \
                    if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values) \
                    else 'categorical'
        
        # Include event attributes
        for attr in pm4py.get_event_attributes(full_log):
            if attr in self.IGNORED_ATTRIBUTES:
                continue 
            sanitized_name = attr
            values = pm4py.get_event_attribute_values(full_log, attr).keys()
            attr_categories[sanitized_name] = 'boolean' \
                if all(isinstance(val, bool) for val in values) \
                else 'numerical' \
                if all(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values) \
                else 'categorical'
            
            # Log if mixed types detected
            if attr_categories[sanitized_name] == 'categorical':
                has_numeric = any(isinstance(val, (int, float)) and not isinstance(val, bool) for val in values)
                has_bool = any(isinstance(val, bool) for val in values)
                if has_numeric or has_bool:
                    logger.warning(f"Attribute {attr} has mixed data types, forcing to 'categorical'")
        
        return attr_categories
