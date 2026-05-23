import logging
import re
from typing import List, Dict, Optional, Any, Tuple, Set, Union

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with the specified name and level.

    Args:
        name: Name of the logger (typically __name__).
        level: Logging level (defaults to logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sanitize_name(name: str) -> str:
    """
    Sanitize an activity or attribute name for use in PDDL syntax.
    Strips 'case:' prefixes and replaces spaces or special characters with underscores.

    Args:
        name: The original name string.

    Returns:
        A PDDL-compatible sanitized lowercase string.

    ESEMPIO:
        "case:concept:name" -> "concept_name"
        "ER Registration" -> "er_registration"
        "CRP (mg/L)" -> "crp__mg_l_"
    """
    if not name:
        return name
    # Strip "case:" prefix for cleaner trace attribute names
    if name.lower().startswith("case:"):
        name = name[5:]  # Remove "case:" prefix
    return name.strip().lower()\
        .replace(" ", "_")\
        .replace(":", "_")\
        .replace("-", "_")\
        .replace("(", "")\
        .replace(")", "")\
        .replace("/", "_")\
        .replace("\\", "_")


def convert_interval_to_lte_gte(interval_str: str) -> str:
    """
    Convert numerical interval range notation into standard lte/gte string format.

    Args:
        interval_str: A string representing interval ranges like "(-inf-6.15]" or "(12.5-inf)".

    Returns:
        A PDDL-compatible string representation.

    ESEMPIO:
        "(-inf-6.15]" -> "lte_6_15"
        "(12.5-inf)" -> "gte_12_5"
        "(10.5-20.0]" -> "gte_10_5_lte_20_0"
    """
    interval = interval_str.strip('()[]')
    if interval.startswith('-inf-'):
        upper_part = interval[5:]  # Remove '-inf-'
        upper_fmt = upper_part.replace('.', '_').replace('-', 'neg')
        return f"lte_{upper_fmt}"
    elif interval.endswith('-inf'):
        lower_part = interval[:-4]  # Remove '-inf'
        lower_fmt = lower_part.replace('.', '_').replace('-', 'neg')
        return f"gte_{lower_fmt}"
    else:
        parts = interval.split('-')
        if len(parts) == 2:
            lower, upper = parts
            lower_fmt = lower.replace('.', '_').replace('-', 'neg')
            upper_fmt = upper.replace('.', '_').replace('-', 'neg')
            return f"gte_{lower_fmt}_lte_{upper_fmt}"
    
    return interval_str


def discretize_value(
    attr: Union[str, None], 
    value: Union[float, str, None], 
    intervals: Optional[Dict[str, List[float]]] = None
) -> str:
    """
    Discretize a numeric or interval value based on defined threshold splits.

    Args:
        attr: The attribute name string.
        value: The raw numerical or interval value to discretize.
        intervals: A dictionary mapping attribute names to lists of floats representing thresholds.

    Returns:
        The discretized value string representation.

    ESEMPIO:
        value="(-inf-6.15]", intervals=None -> "lte_6_15"
        value=5.0, intervals={"crp": [10.0, 50.0]} -> "lte_10_0"
        value=25.0, intervals={"crp": [10.0, 50.0]} -> "gte_10_0_lte_50_0"
        value=60.0, intervals={"crp": [10.0, 50.0]} -> "gte_50_0"
    """
    if (isinstance(value, str) and
            (
                '-inf' in value or
                (
                    '-' in value and
                    not value.replace('-', '').replace('.', '').replace('_', '').isalnum()
                )
            )):
        return convert_interval_to_lte_gte(value)
    else:   # Numeric discretization
        if intervals is None or attr not in intervals or not intervals[attr] or value is None:
            return str(value)
        
        try:
            value = float(value)
            thresholds = sorted(intervals[attr])
            
            if value <= thresholds[0]:
                thresh_str = str(thresholds[0]).replace('.', '_').replace('-', 'neg')
                return f'lte_{thresh_str}'
            elif value >= thresholds[-1]:
                thresh_str = str(thresholds[-1]).replace('.', '_').replace('-', 'neg')
                return f'gte_{thresh_str}'
            else:
                for i in range(len(thresholds) - 1):
                    if thresholds[i] < value <= thresholds[i + 1]:
                        lower_str = str(thresholds[i]).replace('.', '_').replace('-', 'neg')
                        upper_str = str(thresholds[i + 1]).replace('.', '_').replace('-', 'neg')
                        return f'gte_{lower_str}_lte_{upper_str}'
            return str(value)
        except (ValueError, TypeError):
            return str(value)
