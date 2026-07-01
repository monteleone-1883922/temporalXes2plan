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

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from evaluation.analysis_logs import clean_columns

logger = logging.getLogger(__name__)


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
    filename = str(
        metadata_row["Event Log Dataset File Name"]
        if pd.notna(metadata_row["Event Log Dataset File Name"])
        else metadata_row["Event Log Name"]
    )
    doi = str(metadata_row["DOI Number"])
    fmt = _fmt_from_dataset_format(str(metadata_row.get("Dataset Format", "")))

    dest = Path(cache_dir) / log_id / filename.replace(" ", "_")

    if dest.exists() and not force:
        return dest, fmt

    url = _resolve_download_url(doi, filename)
    logger.debug("[%s] Downloading from URL: %s", log_id, url)
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


def _resolve_download_url(doi: str, filename: str) -> str:
    """Resolve a direct download URL from a DOI by following the redirect and
    scraping the landing page for a link matching *filename*.

    On 4TU pages the file links live inside ``<div id="files">`` and the
    filename appears as the link **text**, not in the href (which is a UUID
    path). The search order is:
    1. Exact text match (or with an added ``.gz`` suffix) inside ``div#files``.
    2. Any link whose text contains the target extension inside ``div#files``.
    3. Generic full-page fallback for ``.xes``/``.csv`` link text.

    Args:
        doi: DOI string (e.g. "10.4121/uuid:..." or "10.5281/zenodo.<id>").
        filename: Filename from the metadata (e.g. "MyLog.xes").

    Returns:
        Direct download URL for the file.

    Raises:
        requests.HTTPError: If the DOI resolution request fails.
        ValueError: If no matching download link is found on the page.
    """
    doi_url = f"https://doi.org/{doi}"
    response = requests.get(doi_url, allow_redirects=True, timeout=30)
    response.raise_for_status()

    page = BeautifulSoup(response.text, "html.parser")
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem

    def _to_absolute(href: str) -> str:
        if href.startswith("http"):
            return href
        pieces = response.url.split("/")
        return f"{pieces[0]}//{pieces[2]}/{href.lstrip('/')}"

    def _is_file_link(a_tag) -> bool:
        return "download-all-files" not in a_tag.get("id", "")

    # --- Primary: <div id="files"> on 4TU pages ---
    files_div = page.find("div", id="files")
    if files_div:
        for a in files_div.find_all("a", href=True):
            if not _is_file_link(a):
                continue
            text = a.get_text(strip=True)
            if _find_filename(text, filename):
                return _to_absolute(a["href"])

        # Extension-level fallback within files section
        for a in files_div.find_all("a", href=True):
            if not _is_file_link(a):
                continue
            if ext in a.get_text(strip=True):
                return _to_absolute(a["href"])

    # --- Generic fallback: any page link whose text contains the filename ---
    for a in page.find_all("a", href=True):
        text = a.get_text(strip=True)
        if filename in text or (ext in (".xes", ".csv") and ext in text):
            return _to_absolute(a["href"])

    raise ValueError(
        f"File '{filename}' not found on landing page for DOI {doi} (resolved to {response.url})"
    )


def _find_filename(link_text: str, filename: str) -> bool:
    return filename.lower() in link_text.lower() or filename.lower().replace(" ", "_") in link_text.lower()
