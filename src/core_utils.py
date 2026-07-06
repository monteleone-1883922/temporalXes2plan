import logging
import re
from typing import List, Dict, Optional, Any, Tuple, Set, Union

def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits its level and handlers from the root logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger instance with propagation enabled.
    """
    return logging.getLogger(name)


def _replace_special_chars(s: str) -> str:
    """Replace non-alphanumeric characters for PDDL name safety.

    Space becomes '_'; any other non-alphanumeric/non-underscore character
    becomes 'ascii{decimal_code}' (e.g. '#' → 'ascii35', '@' → 'ascii64').
    Consecutive underscores are collapsed and leading/trailing underscores
    are stripped.
    """
    parts = []
    for ch in s:
        if ch.isalpha() or ch.isdigit() or ch == '_':
            parts.append(ch)
        elif ch == ' ':
            parts.append('_')
        else:
            parts.append(f"ascii{ord(ch)}")
    result = ''.join(parts)
    result = re.sub(r'_+', '_', result)
    return result.strip('_')


def sanitize_name(name: str) -> str:
    """Sanitize an activity or attribute name for use in PDDL syntax.

    Strips 'case:' prefixes; spaces become '_'; other special characters
    become 'ascii{code}' to preserve uniqueness and avoid empty results.

    ESEMPIO:
        "case:concept:name" -> "conceptascii58name"
        "ER Registration"   -> "er_registration"
        "#"                 -> "ascii35"
    """
    if not name:
        return name
    if name.lower().startswith("case:"):
        name = name[5:]
    name = _replace_special_chars(name.strip().lower().replace(" ", "_")\
        .replace(":", "_")\
        .replace("-", "_")\
        .replace("(", "")\
        .replace(")", "")\
        .replace("/", "_")\
        .replace("\\", "_"))
    return name or "_"


def sanitize_value(attr_name: str, raw_value: Optional[str]) -> Optional[str]:
    """Produce a PDDL constant for an attribute value as {attr}_val_{value}.

    Args:
        attr_name: Already-sanitized attribute name (e.g. "costo").
        raw_value: Raw string value from the log (e.g. "alto", "lte_6_0", "#").

    Returns:
        A valid PDDL constant name, e.g. "costo_val_alto".

    ESEMPIO:
        sanitize_value("costo",    "alto")   -> "costo_val_alto"
        sanitize_value("expense",  "lte_6_0")-> "expense_val_lte_6_0"
        sanitize_value("dismissal","#")      -> "dismissal_val_ascii35"
        sanitize_value("amount",   "2")      -> "amount_val_2"
    """
    if raw_value is None:
        return None
    if raw_value.startswith(f"{attr_name}_val_"):
        return raw_value
    cleaned = _replace_special_chars(str(raw_value).strip().lower())
    cleaned = cleaned or "empty"
    return f"{attr_name}_val_{cleaned}"

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

#TODO is there a test for this or at least a review to see if can be improved
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
