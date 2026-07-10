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
import re
import pandas as pd
import requests
import tarfile
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
ZENODO_ZIP_URL =  "https://zenodo.org/records/16268743/files/Collection_Event_Logs.zip?download=1"
METADATA_CSV_URL = "https://zenodo.org/records/16268743/files/Metadata.csv?download=1"
_zenodo_zip_cache: zipfile.ZipFile | None = None

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i nomi colonna rimuovendo i ritorni a capo interni."""
    df.columns = df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    return df


def download_metadata_if_needed(metadata_csv: Path, force: bool = False) -> Path:
    """Ensure Metadata.csv is available on disk, downloading it from Zenodo
    if missing.

    Args:
        metadata_csv: Local path where Metadata.csv should live.
        force: If True, re-download even if the file already exists.

    Returns:
        The same path, guaranteed to exist on disk.

    Raises:
        requests.HTTPError: If the download fails.
    """
    metadata_csv = Path(metadata_csv)
    if metadata_csv.exists() and not force:
        return metadata_csv

    logger.info("Downloading Metadata.csv from Zenodo to %s ...", metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(METADATA_CSV_URL, timeout=120)
    response.raise_for_status()
    metadata_csv.write_bytes(response.content)
    return metadata_csv


def get_log_selection(
    metadata_csv: Path,
    log_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the BPM 2025 metadata CSV and optionally filter by log ID.

    Downloads Metadata.csv from Zenodo first if it's not already present at
    the given path (see download_metadata_if_needed()).

    Args:
        metadata_csv: Path to a locally stored Metadata.csv.
        log_ids: Optional list of "Event Log ID" values to retain.
            If None or empty, all rows are returned.

    Returns:
        Filtered DataFrame with the same columns as analysis_logs.filter_logs().
    """
    metadata_csv = download_metadata_if_needed(Path(metadata_csv))
    df = pd.read_csv(metadata_csv)
    df = clean_columns(df)

    if log_ids:
        ids_str = [str(i) for i in log_ids]
        df = df.loc[df["Event Log ID"].astype(str).isin(ids_str)].copy()

    return df.reset_index(drop=True)


def _get_zenodo_zip() -> zipfile.ZipFile:
    """Download and cache the Zenodo ZIP file containing bundled event logs.

    The ZIP is downloaded once per session and kept in memory for subsequent
    lookups.  This avoids re-downloading the archive for every log.

    Returns:
        A ZipFile object open for reading.

    Raises:
        requests.HTTPError: If the download fails.
        zipfile.BadZipFile: If the downloaded content is not a valid ZIP.
    """
    global _zenodo_zip_cache
    if _zenodo_zip_cache is not None:
        return _zenodo_zip_cache

    logger.info("Downloading Zenodo ZIP archive from %s ...", ZENODO_ZIP_URL)
    response = requests.get(ZENODO_ZIP_URL, timeout=300)
    response.raise_for_status()

    _zenodo_zip_cache = zipfile.ZipFile(io.BytesIO(response.content))
    logger.info(
        "Zenodo ZIP loaded successfully (%d entries).",
        len(_zenodo_zip_cache.namelist()),
    )
    return _zenodo_zip_cache

def _extract_from_zenodo_zip(dataset_filenames: List[str], fmt: str) -> bytes:
    """Extract a specific file from the cached Zenodo ZIP.

    Handles the case where the file might be inside a subdirectory within the
    ZIP, or might have a .gz extension inside the archive.

    Args:
        dataset_filename: The value from 'Event Log Dataset File Name' column.
        fmt: Expected format extension ('.xes' or '.csv').

    Returns:
        Raw bytes of the extracted (and decompressed, if .gz) file.

    Raises:
        FileNotFoundError: If the file is not found in the ZIP archive.
    """
    zf = _get_zenodo_zip()
    namelist = zf.namelist()
    candidates = None
    for raw_filename in dataset_filenames:
        filename = raw_filename.replace(" ", "_") + ("" if raw_filename.endswith(fmt) else fmt)
        # Try exact match first
        candidates = [n for n in namelist if n.endswith(filename)]

        # If not found, try with common extensions appended
        if not candidates:
            candidates = [
                n for n in namelist
                if n.endswith(filename + ".gz")
                or n.endswith(filename + ".zip")
            ]

        # Last resort: case-insensitive partial match
        if not candidates:
            lower_target = filename.lower()
            candidates = [n for n in namelist if lower_target in n.lower()]
        else:
            break

    if not candidates:
        raise FileNotFoundError(
            f"Filenames '{dataset_filenames}' not found in Zenodo ZIP. "
            f"Available entries (first 20): {namelist[:20]}"
        )

    # Pick the best match (shortest path = most direct)
    best = min(candidates, key=len)
    logger.debug("Extracting '%s' from Zenodo ZIP.", best)

    content = zf.read(best)

    # Decompress if gzipped
    if best.endswith(".gz"):
        content = gzip.decompress(content)
    elif best.endswith(".zip"):
        content = _extract_from_zip(content, fmt)

    return content

def download_if_needed(
    metadata_row: "pd.Series",
    cache_dir: Path,
    force: bool = False,
) -> tuple[Path, str]:
    """Ensure the event log file is available on disk, downloading if necessary.

    Files are cached at ``<cache_dir>/<log_id>/<filename>``.  A cached copy is
    reused on subsequent calls unless *force* is True.

    Download strategy:
      1. Try resolving the file from the DOI (original behavior).
      2. **Fallback**: If the DOI resolution fails AND the metadata row has a
         non-NA 'Event Log Dataset File Name', attempt extraction from the
         Zenodo ZIP archive bundled with the metadata.

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
        RuntimeError: If both the DOI download and the Zenodo ZIP fallback fail.
    """
    log_id = str(metadata_row["Event Log ID"])
    dataset_file_name = metadata_row.get("Event Log Dataset File Name")
    has_zip_entry = pd.notna(dataset_file_name) and str(dataset_file_name).strip() not in ("", "NA")

    filename = str(metadata_row["Event Log Name"]) + (
        f"_{str(dataset_file_name)}"
        if has_zip_entry
        else ""
    )
    file_names = [str(metadata_row["Event Log Name"])] + (
        [str(dataset_file_name).replace(" ", "_").replace("-", "_"),
         str(dataset_file_name).replace(" ", "_"),
         str(metadata_row["Event Log Name"]) + "_" + str(dataset_file_name).replace(" ", "_").replace("-", "_"),
         str(metadata_row["Event Log Name"]) + "_" + str(dataset_file_name).replace(" ", "_")
         ]
        if has_zip_entry
        else []
    )

    doi = str(metadata_row["DOI Number"])
    fmt = _fmt_from_dataset_format(str(metadata_row.get("Dataset Format", "")))

    dest_filename = filename.replace(" ", "_") + (
        "" if filename.endswith(".xes") or filename.endswith(".csv") else fmt
    )
    dest = Path(cache_dir) / log_id / dest_filename

    if dest.exists() and not force:
        return dest, fmt

    dest.parent.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: Resolve from DOI ---
    doi_error: Exception | None = None
    if pd.notna(metadata_row.get("DOI Number")) and str(doi).strip() not in ("", "N/A", "NA"):
        try:
            url, actual_filename = _resolve_file_link(doi, file_names)
            logger.debug("[%s] Downloading from DOI URL: %s", log_id, url)

            response = requests.get(url, timeout=120)
            response.raise_for_status()

            safe_name = actual_filename.replace(" ", "_")
            if safe_name.endswith(".gz") and not safe_name.endswith(".tar.gz"):
                # Plain gzipped file (e.g. "log.xes.gz")
                dest.write_bytes(gzip.decompress(response.content))
            elif safe_name.endswith(".zip") or safe_name.endswith(".tar.gz"):
                # Archive containing one or more files
                dest.write_bytes(
                    _extract_from_archive(response.content, fmt, filenames=file_names)
                )
            elif safe_name.endswith(fmt):
                dest.write_bytes(response.content)
            else:
                raise ValueError(
                    f"Unexpected file extension for DOI download: {safe_name}"
                )

            return dest, fmt

        except Exception as e:
            doi_error = e
            logger.warning(
                "[%s] DOI download failed (%s). Attempting Zenodo ZIP fallback...",
                log_id,
                e,
            )

    # --- Strategy 2: Fallback to Zenodo ZIP ---

        try:
            content = _extract_from_zenodo_zip(file_names, fmt)
            dest.write_bytes(content)
            logger.info(
                "[%s] Successfully extracted from Zenodo ZIP as '%s'.",
                log_id,
                dest_filename,
            )
            return dest, fmt

        except Exception as zip_error:
            logger.error(
                "[%s] Zenodo ZIP fallback also failed: %s", log_id, zip_error
            )
            raise RuntimeError(
                f"[{log_id}] Both DOI download and Zenodo ZIP fallback failed. "
                f"DOI error: {doi_error}; ZIP error: {zip_error}"
            ) from zip_error

    # --- Neither strategy available ---
    if doi_error:
        raise RuntimeError(
            f"[{log_id}] DOI download failed and no Zenodo ZIP entry available. "
            f"Error: {doi_error}"
        ) from doi_error

    raise RuntimeError(
        f"[{log_id}] No DOI and no Zenodo ZIP entry — cannot download this log."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_target_in_archive(entries: List[str], fmt: str, filenames: List[str] | None = None) -> str:
    """Find the target file among archive entries using a two-step strategy.

    Strategy:
      1. If exactly ONE file with the correct extension exists, return it
         (unambiguous case).
      2. If multiple files match the extension, search for the one whose name
         matches an entry in *filenames* (case-insensitive, with space/underscore
         normalization).

    Args:
        entries: List of file paths inside the archive.
        fmt: Target format extension without dot ("xes" or "csv").
        filenames: Candidate filenames to match against when multiple entries
            share the same extension.

    Returns:
        The archive entry path to extract.

    Raises:
        ValueError: If no file with the expected extension is found, or if
            multiple files match and none corresponds to the provided filenames.
    """
    # Filter entries matching the target format (including .gz variants)
    matches = [
        n for n in entries
        if n.lower().endswith(f"{fmt}") or n.lower().endswith(f"{fmt}.gz") or n.lower().endswith(f"{fmt}.zip")
    ]

    if not matches:
        raise ValueError(
            f"No {fmt} file found inside archive (contents: {entries[:30]})"
        )

    # --- Case 1: Single match — unambiguous ---
    if len(matches) == 1:
        return matches[0]

    # --- Case 2: Multiple matches — find the one matching our filenames ---
    if filenames:

        def _normalize(name: str, add_fmt: bool = True) -> str:
            return name.lower().replace(" ", "_") + ("" if name.endswith(fmt) or not add_fmt else fmt)

        normalized_targets = [
            _normalize(f) for f in filenames if f and str(f).strip()
        ]
        for entry in matches:
            if any(_normalize(entry, False).endswith(filename) or
                   _normalize(entry, False).endswith(filename + ".gz") or
                   _normalize(entry, False).endswith(filename + ".zip")
                   for filename in normalized_targets):
                return entry
    # --- Fallback: prefer top-level files over nested ones ---
    top_level = [n for n in matches if "/" not in n and "\\" not in n]
    entry = top_level[0] if top_level else matches[0]
    logger.warning(
        "Multiple .%s files in archive and none matched filenames %s. "
        "Using '%s' as fallback.",
        fmt,
        filenames,
        entry,
    )
    return entry

def _extract_from_archive(
    content: bytes, fmt: str, filenames: List[str] | None = None
) -> bytes:
    """Extract the target file from a zip or tar.gz archive.

    Detects the archive type automatically and delegates to the appropriate
    library.  Uses _find_target_in_archive() to select which file to extract.

    Args:
        content: Raw bytes of the archive.
        fmt: Target format extension without dot ("xes" or "csv").
        filenames: Candidate filenames to match when the archive contains
            multiple files with the correct extension.

    Returns:
        Raw bytes of the extracted (and decompressed if .gz) file.

    Raises:
        ValueError: If the archive format is unrecognized or the target file
            cannot be found inside.
    """
    data: bytes

    # --- Try ZIP first ---
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            entries = zf.namelist()
            entry = _find_target_in_archive(entries, fmt, filenames)
            logger.debug("Extracting '%s' from zip archive", entry)
            data = zf.read(entry)

    # --- Try tar/tar.gz ---
    else:
        try:
            with tarfile.open(fileobj=io.BytesIO(content)) as tf:
                entries = [m.name for m in tf.getmembers() if m.isfile()]
                entry = _find_target_in_archive(entries, fmt, filenames)
                logger.debug("Extracting '%s' from tar archive", entry)
                member = tf.getmember(entry)
                f = tf.extractfile(member)
                if f is None:
                    raise ValueError(
                        f"Cannot read '{entry}' from tar archive (extractfile returned None)"
                    )
                data = f.read()
        except (tarfile.TarError, Exception) as e:
            raise ValueError(
                f"Content is neither a valid zip nor tar archive: {e}"
            ) from e

    # Decompress if the entry itself is gzipped
    # --- Post-extraction decompression / nested archive handling ---
    # Case: extracted file is itself gzipped (e.g. "log.xes.gz")
    if entry.endswith(".gz") and not entry.endswith(".tar.gz"):
        data = gzip.decompress(data)

    # Case: extracted file is itself a zip archive (nested zip)
    elif entry.endswith(".zip"):
        data = _extract_from_archive(data, fmt, filenames)

    # Case: extracted file is itself a tar.gz archive (nested tar)
    elif entry.endswith(".tar.gz") or entry.endswith(".tgz"):
        data = _extract_from_archive(data, fmt, filenames)

    return data


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
        matches = [n for n in zf.namelist() if n.lower().endswith(f"{fmt}")]
        if not matches:
            raise ValueError(f"No {fmt} file found inside zip archive (contents: {zf.namelist()})")
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
    fallback_files = []
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
                if text.endswith(".gz") or text.endswith(".zip") and not fallback_files:
                    fallback_files.append((_to_absolute(a["href"]), text))

            if fallback_files:
                return fallback_files[0]


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
