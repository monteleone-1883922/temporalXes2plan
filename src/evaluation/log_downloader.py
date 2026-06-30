"""Download and cache event logs from Zenodo based on the BPM 2025 metadata.

Typical usage::

    from evaluation.log_downloader import get_log_selection, download_if_needed
    from pathlib import Path

    df = get_log_selection(Path("evaluation/data/Metadata.csv"), log_ids=["1", "3"])
    for _, row in df.iterrows():
        path, fmt = download_if_needed(row, cache_dir=Path("/tmp/logs"))
        # path is ready to pass to EvalAPI.parse()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from evaluation.analysis_logs import clean_columns


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_log_selection(
    metadata_csv: Path,
    log_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the BPM 2025 metadata CSV and optionally filter by log ID.

    Args:
        metadata_csv: Path to a locally stored Metadata.csv.
        log_ids: Optional list of "Event Log ID" values to retain.
            If None or empty, all rows are returned.

    Returns:
        Filtered DataFrame with the same columns as analysis_logs.filter_logs().
    """
    df = pd.read_csv(metadata_csv)
    df = clean_columns(df)

    if log_ids:
        ids_str = [str(i) for i in log_ids]
        df = df.loc[df["Event Log ID"].astype(str).isin(ids_str)].copy()

    return df.reset_index(drop=True)


def download_if_needed(
    metadata_row: "pd.Series",
    cache_dir: Path,
    force: bool = False,
) -> tuple[Path, str]:
    """Ensure the event log file is available on disk, downloading if necessary.

    Files are cached at ``<cache_dir>/<log_id>/<filename>``.  A cached copy is
    reused on subsequent calls unless *force* is True.

    Args:
        metadata_row: A row from the DataFrame returned by get_log_selection().
            Required columns: ``Event Log ID``, ``Event Log Dataset File Name``,
            ``DOI Number``, ``Dataset Format``.
        cache_dir: Root directory for the file cache.
        force: If True, always re-download even when the file is already cached.

    Returns:
        Tuple (local_path, fmt) where fmt is a lowercase format string
        ("xes" or "csv").

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    log_id = str(metadata_row["Event Log ID"])
    filename = str(metadata_row["Event Log Dataset File Name"])
    doi = str(metadata_row["DOI Number"])
    fmt = _fmt_from_dataset_format(str(metadata_row.get("Dataset Format", "")))

    dest = Path(cache_dir) / log_id / filename

    if dest.exists() and not force:
        return dest, fmt

    url = _zenodo_download_url(doi, filename)
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    dest.write_bytes(response.content)
    return dest, fmt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_from_dataset_format(dataset_format: str) -> str:
    """Derive a lowercase format tag from the 'Dataset Format' column value.

    Args:
        dataset_format: Raw value from the metadata column (e.g. "XES", "CSV").

    Returns:
        Lowercase format string: "xes" or "csv".
    """
    return dataset_format.strip().lower()


def _zenodo_download_url(doi: str, filename: str) -> str:
    """Build a Zenodo direct-download URL from a DOI and filename.

    Args:
        doi: DOI string of the form "10.5281/zenodo.<record_id>" or a plain
            record ID.
        filename: Filename within the Zenodo record.

    Returns:
        Direct download URL.
    """
    m = re.search(r"zenodo\.(\d+)", doi, re.IGNORECASE)
    record_id = m.group(1) if m else doi.strip()
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
