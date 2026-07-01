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

import gzip
import io
import logging
import zipfile
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
    filename = str(metadata_row["Event Log Name"]) + (
        f"_{str(metadata_row["Event Log Dataset File Name"])}"
        if pd.notna(metadata_row["Event Log Dataset File Name"])
        else ""
    )
    file_names = [str(metadata_row["Event Log Name"])] + (
        [str(metadata_row["Event Log Dataset File Name"])]
        if pd.notna(metadata_row["Event Log Dataset File Name"])
        else []
    )

    doi = str(metadata_row["DOI Number"])
    fmt = _fmt_from_dataset_format(str(metadata_row.get("Dataset Format", "")))

    dest_filename = filename.replace(" ", "_") + ("" if filename.endswith(".xes") or filename.endswith(".csv") else fmt)
    dest = Path(cache_dir) / log_id / dest_filename

    if dest.exists() and not force:
        return dest, fmt

    url, actual_filename = _resolve_file_link(doi, file_names)
    logger.debug("[%s] Downloading from URL: %s", log_id, url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    safe_name = actual_filename.replace(" ", "_")
    if safe_name.endswith(".gz"):
        dest.write_bytes(gzip.decompress(response.content))
    elif safe_name.endswith(".zip"):
        dest.write_bytes(_extract_from_zip(response.content, fmt))
    else:
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


def _extract_from_zip(content: bytes, fmt: str) -> bytes:
    """Extract the first file matching *fmt* extension from a zip archive.

    Args:
        content: Raw bytes of the zip archive.
        fmt: Target format ("xes" or "csv").

    Returns:
        Raw bytes of the extracted file.

    Raises:
        ValueError: If no file with the expected extension is found in the archive.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        matches = [n for n in zf.namelist() if n.lower().endswith(f".{fmt}")]
        if not matches:
            raise ValueError(f"No .{fmt} file found inside zip archive (contents: {zf.namelist()})")
        # Prefer a file not inside a subdirectory; fall back to the first match
        top_level = [n for n in matches if "/" not in n]
        entry = top_level[0] if top_level else matches[0]
        logger.debug("Extracting '%s' from zip archive", entry)
        return zf.read(entry)


def _resolve_file_link(doi: str, filenames: List[str]) -> tuple[str, str]:
    """Resolve a direct download URL and actual filename from a DOI landing page.

    On 4TU pages the file links live inside ``<div id="files">`` and the
    filename appears as the link **text**, not in the href (which is a UUID
    path). The actual filename on the page may differ from the metadata value
    (e.g. the page shows ``MyLog.xes.gz`` while the CSV says ``MyLog.xes``).

    Search order:
    1. Exact text match (or with an added ``.gz`` suffix) inside ``div#files``.
    2. Any link whose text contains the target extension inside ``div#files``.
    3. Generic full-page fallback for ``.xes``/``.csv`` link text.

    Args:
        doi: DOI string (e.g. "10.4121/uuid:..." or "10.5281/zenodo.<id>").
        filename: Filename from the metadata (e.g. "MyLog.xes").

    Returns:
        Tuple (url, actual_filename) where actual_filename is the link text
        from the landing page (e.g. "MyLog.xes.gz").

    Raises:
        requests.HTTPError: If the DOI resolution request fails.
        ValueError: If no matching download link is found on the page.
    """
    doi_url = f"https://doi.org/{doi}"
    response = requests.get(doi_url, allow_redirects=True, timeout=30)
    response.raise_for_status()

    page = BeautifulSoup(response.text, "html.parser")
    def _to_absolute(href: str) -> str:
        if href.startswith("http"):
            return href
        pieces = response.url.split("/")
        return f"{pieces[0]}//{pieces[2]}/{href.lstrip('/')}"

    def _is_file_link(a_tag) -> bool:
        return "download-all-files" not in a_tag.get("id", "")
    for filename in filenames:
        ext = Path(filename).suffix.lower()


        # --- Primary: <div id="files"> on 4TU pages ---
        files_div = page.find("div", id="files")
        if files_div:
            for a in files_div.find_all("a", href=True):
                if not _is_file_link(a):
                    continue
                text = a.get_text(strip=True)
                if _find_filename(text, filename):
                    return _to_absolute(a["href"]), text

            # Extension-level fallback within files section
            for a in files_div.find_all("a", href=True):
                if not _is_file_link(a):
                    continue
                text = a.get_text(strip=True)
                if ext in text:
                    return _to_absolute(a["href"]), text

        # --- Generic fallback: any page link whose text contains the filename ---
        for a in page.find_all("a", href=True):
            text = a.get_text(strip=True)
            if filename in text or (ext in (".xes", ".csv") and ext in text):
                return _to_absolute(a["href"]), text

        raise ValueError(
            f"File '{filename}' not found on landing page for DOI {doi} (resolved to {response.url})"
        )


def _find_filename(link_text: str, filename: str) -> bool:
    return filename.lower() in link_text.lower() or filename.lower().replace(" ", "_") in link_text.lower()
