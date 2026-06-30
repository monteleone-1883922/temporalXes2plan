"""Unit tests for evaluation.log_downloader."""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.log_downloader import (
    download_if_needed,
    get_log_selection,
    _fmt_from_dataset_format,
    _zenodo_download_url,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_METADATA_COLUMNS = [
    "Event Log ID",
    "Event Log Name",
    "Event Log Dataset File Name",
    "DOI Number",
    "Dataset Format",
    "Prominent Exhibited Behavior",
    "behavior_rank",
    "Number of Activities",
    "Number of Cases",
]


def _make_metadata_csv(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows, columns=_METADATA_COLUMNS)
    path = tmp_path / "Metadata.csv"
    df.to_csv(path, index=False)
    return path


def _make_row(
    log_id: str = "42",
    filename: str = "log.xes",
    doi: str = "10.5281/zenodo.9999999",
    fmt: str = "XES",
) -> pd.Series:
    return pd.Series({
        "Event Log ID": log_id,
        "Event Log Dataset File Name": filename,
        "DOI Number": doi,
        "Dataset Format": fmt,
    })


# ---------------------------------------------------------------------------
# get_log_selection
# ---------------------------------------------------------------------------

class TestGetLogSelection:
    def _sample_csv(self, tmp_path: Path) -> Path:
        rows = [
            {"Event Log ID": "1", "Event Log Name": "A", "Event Log Dataset File Name": "a.xes",
             "DOI Number": "10.5281/zenodo.1", "Dataset Format": "XES",
             "Prominent Exhibited Behavior": "Linear", "behavior_rank": 0,
             "Number of Activities": 5, "Number of Cases": 100},
            {"Event Log ID": "2", "Event Log Name": "B", "Event Log Dataset File Name": "b.xes",
             "DOI Number": "10.5281/zenodo.2", "Dataset Format": "XES",
             "Prominent Exhibited Behavior": "Spaghetti Behavior", "behavior_rank": 4,
             "Number of Activities": 30, "Number of Cases": 500},
            {"Event Log ID": "3", "Event Log Name": "C", "Event Log Dataset File Name": "c.csv",
             "DOI Number": "10.5281/zenodo.3", "Dataset Format": "CSV",
             "Prominent Exhibited Behavior": "Linear", "behavior_rank": 0,
             "Number of Activities": 8, "Number of Cases": 200},
        ]
        return _make_metadata_csv(tmp_path, rows)

    def test_returns_dataframe(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv)
        assert isinstance(result, pd.DataFrame)

    def test_no_filter_returns_all_rows(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv)
        assert len(result) == 3

    def test_filter_by_single_id(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv, log_ids=["2"])
        assert len(result) == 1
        assert str(result.iloc[0]["Event Log ID"]) == "2"

    def test_filter_by_multiple_ids(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv, log_ids=["1", "3"])
        assert len(result) == 2
        ids = set(result["Event Log ID"].astype(str))
        assert ids == {"1", "3"}

    def test_filter_with_empty_list_returns_all(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv, log_ids=[])
        assert len(result) == 3

    def test_filter_with_none_returns_all(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv, log_ids=None)
        assert len(result) == 3

    def test_unknown_id_returns_empty(self, tmp_path):
        csv = self._sample_csv(tmp_path)
        result = get_log_selection(csv, log_ids=["999"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# download_if_needed — cache behaviour
# ---------------------------------------------------------------------------

class TestDownloadIfNeeded:
    def test_returns_cached_file_without_download(self, tmp_path):
        row = _make_row(log_id="7", filename="log.xes")
        dest = tmp_path / "7" / "log.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"<log/>")

        with patch("evaluation.log_downloader.requests.get") as mock_get:
            path, fmt = download_if_needed(row, cache_dir=tmp_path)

        mock_get.assert_not_called()
        assert path == dest

    def test_downloads_when_cache_miss(self, tmp_path):
        row = _make_row(log_id="8", filename="new.xes")

        mock_resp = MagicMock()
        mock_resp.content = b"<log>data</log>"
        mock_resp.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_resp) as mock_get:
            path, fmt = download_if_needed(row, cache_dir=tmp_path)

        mock_get.assert_called_once()
        assert path.exists()
        assert path.read_bytes() == b"<log>data</log>"

    def test_force_redownloads_existing_file(self, tmp_path):
        row = _make_row(log_id="9", filename="old.xes")
        dest = tmp_path / "9" / "old.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"stale")

        mock_resp = MagicMock()
        mock_resp.content = b"fresh"
        mock_resp.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_resp) as mock_get:
            path, fmt = download_if_needed(row, cache_dir=tmp_path, force=True)

        mock_get.assert_called_once()
        assert path.read_bytes() == b"fresh"

    def test_cached_path_structure(self, tmp_path):
        row = _make_row(log_id="10", filename="trace.xes")
        dest = tmp_path / "10" / "trace.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x")

        with patch("evaluation.log_downloader.requests.get"):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path == dest

    def test_creates_parent_directories_on_download(self, tmp_path):
        row = _make_row(log_id="11", filename="deep.xes")

        mock_resp = MagicMock()
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_resp):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path.parent.is_dir()


# ---------------------------------------------------------------------------
# download_if_needed — format derivation
# ---------------------------------------------------------------------------

class TestFmtDerivation:
    def _download_with_fmt(self, tmp_path, fmt_value: str) -> str:
        row = _make_row(log_id="20", filename="log.x", fmt=fmt_value)
        mock_resp = MagicMock()
        mock_resp.content = b"x"
        mock_resp.raise_for_status = MagicMock()
        with patch("evaluation.log_downloader.requests.get", return_value=mock_resp):
            _, fmt = download_if_needed(row, cache_dir=tmp_path)
        return fmt

    def test_fmt_derived_from_dataset_format_xes(self, tmp_path):
        assert self._download_with_fmt(tmp_path, "XES") == "xes"

    def test_fmt_derived_from_dataset_format_csv(self, tmp_path):
        assert self._download_with_fmt(tmp_path, "CSV") == "csv"

    def test_fmt_lowercase_passthrough(self, tmp_path):
        assert self._download_with_fmt(tmp_path, "xes") == "xes"

    def test_fmt_strips_whitespace(self):
        assert _fmt_from_dataset_format("  XES  ") == "xes"


# ---------------------------------------------------------------------------
# _zenodo_download_url
# ---------------------------------------------------------------------------

class TestZenodoDownloadUrl:
    def test_doi_with_zenodo_record(self):
        url = _zenodo_download_url("10.5281/zenodo.1234567", "my_log.xes")
        assert url == "https://zenodo.org/records/1234567/files/my_log.xes?download=1"

    def test_doi_uppercase_zenodo(self):
        url = _zenodo_download_url("10.5281/Zenodo.9876543", "log.xes")
        assert "9876543" in url

    def test_plain_record_id_fallback(self):
        url = _zenodo_download_url("9999999", "file.xes")
        assert "9999999" in url
        assert "zenodo.org/records" in url
