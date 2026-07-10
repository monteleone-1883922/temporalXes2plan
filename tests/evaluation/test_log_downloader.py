"""Unit tests for evaluation.log_downloader."""
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from evaluation.log_downloader import (
    download_if_needed,
    download_metadata_if_needed,
    get_log_selection,
    _extract_from_zip,
    _fmt_from_dataset_format,
    _find_filename,
    _resolve_file_link,
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
    log_name: str = "TestLog",
    filename: str = "log.xes",
    doi: str = "10.5281/zenodo.9999999",
    fmt: str = "XES",
) -> pd.Series:
    return pd.Series({
        "Event Log ID": log_id,
        "Event Log Name": log_name,
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
# download_metadata_if_needed
# ---------------------------------------------------------------------------

class TestDownloadMetadataIfNeeded:
    def test_returns_existing_file_without_download(self, tmp_path):
        path = tmp_path / "Metadata.csv"
        path.write_text("existing content", encoding="utf-8")

        with patch("evaluation.log_downloader.requests.get") as mock_get:
            result = download_metadata_if_needed(path)

        mock_get.assert_not_called()
        assert result == path
        assert path.read_text(encoding="utf-8") == "existing content"

    def test_downloads_when_missing(self, tmp_path):
        path = tmp_path / "Metadata.csv"
        mock_response = MagicMock(content=b"col_a,col_b\n1,2\n")
        mock_response.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_response) as mock_get:
            result = download_metadata_if_needed(path)

        mock_get.assert_called_once()
        assert result == path
        assert path.read_bytes() == b"col_a,col_b\n1,2\n"

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "Metadata.csv"
        mock_response = MagicMock(content=b"data")
        mock_response.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_response):
            download_metadata_if_needed(path)

        assert path.exists()

    def test_force_redownloads_existing_file(self, tmp_path):
        path = tmp_path / "Metadata.csv"
        path.write_text("stale content", encoding="utf-8")
        mock_response = MagicMock(content=b"fresh content")
        mock_response.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_response) as mock_get:
            download_metadata_if_needed(path, force=True)

        mock_get.assert_called_once()
        assert path.read_bytes() == b"fresh content"

    def test_get_log_selection_downloads_metadata_when_missing(self, tmp_path):
        path = tmp_path / "Metadata.csv"
        rows = [{"Event Log ID": "1", "Event Log Name": "A", "Event Log Dataset File Name": "a.xes",
                 "DOI Number": "10.5281/zenodo.1", "Dataset Format": "XES",
                 "Prominent Exhibited Behavior": "Linear", "behavior_rank": 0,
                 "Number of Activities": 5, "Number of Cases": 100}]
        csv_bytes = pd.DataFrame(rows, columns=_METADATA_COLUMNS).to_csv(index=False).encode("utf-8")
        mock_response = MagicMock(content=csv_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch("evaluation.log_downloader.requests.get", return_value=mock_response) as mock_get:
            result = get_log_selection(path)

        mock_get.assert_called_once()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# download_if_needed — cache behaviour
#
# _resolve_file_link() is mocked directly here: it does its own HTML-scraping
# HTTP call (tested separately below in TestResolveFileLink) and is orthogonal
# to download_if_needed's caching/download logic.
# ---------------------------------------------------------------------------

class TestDownloadIfNeeded:
    def test_returns_cached_file_without_download(self, tmp_path):
        row = _make_row(log_id="7", log_name="Log7", filename="log.xes")
        dest = tmp_path / "7" / "Log7_log.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"<log/>")

        with (
            patch("evaluation.log_downloader._resolve_file_link") as mock_resolve,
            patch("evaluation.log_downloader.requests.get") as mock_get,
        ):
            path, fmt = download_if_needed(row, cache_dir=tmp_path)

        mock_resolve.assert_not_called()
        mock_get.assert_not_called()
        assert path == dest

    def test_downloads_when_cache_miss(self, tmp_path):
        row = _make_row(log_id="8", log_name="Log8", filename="new.xes")

        mock_resp = MagicMock()
        mock_resp.content = b"<log>data</log>"
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=("https://example.org/new.xes", "new.xes")) as mock_resolve,
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp) as mock_get,
        ):
            path, fmt = download_if_needed(row, cache_dir=tmp_path)

        mock_resolve.assert_called_once()
        mock_get.assert_called_once()
        assert path.exists()
        assert path.read_bytes() == b"<log>data</log>"

    def test_force_redownloads_existing_file(self, tmp_path):
        row = _make_row(log_id="9", log_name="Log9", filename="old.xes")
        dest = tmp_path / "9" / "Log9_old.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"stale")

        mock_resp = MagicMock()
        mock_resp.content = b"fresh"
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=("https://example.org/old.xes", "old.xes")),
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp) as mock_get,
        ):
            path, fmt = download_if_needed(row, cache_dir=tmp_path, force=True)

        mock_get.assert_called_once()
        assert path.read_bytes() == b"fresh"

    def test_cached_path_structure(self, tmp_path):
        row = _make_row(log_id="10", log_name="Log10", filename="trace.xes")
        dest = tmp_path / "10" / "Log10_trace.xes"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x")

        with (
            patch("evaluation.log_downloader._resolve_file_link"),
            patch("evaluation.log_downloader.requests.get"),
        ):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path == dest

    def test_creates_parent_directories_on_download(self, tmp_path):
        row = _make_row(log_id="11", log_name="Log11", filename="deep.xes")

        mock_resp = MagicMock()
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=("https://example.org/deep.xes", "deep.xes")),
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp),
        ):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path.parent.is_dir()

    def test_gz_response_is_decompressed(self, tmp_path):
        import gzip
        row = _make_row(log_id="12", log_name="Log12", filename="log.xes")

        mock_resp = MagicMock()
        mock_resp.content = gzip.compress(b"raw xes content")
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=("https://example.org/log.xes.gz", "log.xes.gz")),
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp),
        ):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path.read_bytes() == b"raw xes content"

    def test_zip_response_is_extracted(self, tmp_path):
        row = _make_row(log_id="13", log_name="Log13", filename="log.xes")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("log.xes", "<log>zipped</log>")
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=("https://example.org/log.zip", "log.zip")),
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp),
        ):
            path, _ = download_if_needed(row, cache_dir=tmp_path)

        assert path.read_bytes() == b"<log>zipped</log>"


# ---------------------------------------------------------------------------
# download_if_needed — format derivation
# ---------------------------------------------------------------------------

class TestFmtDerivation:
    def _download_with_fmt(self, tmp_path, fmt_value: str) -> str:
        row = _make_row(log_id="20", log_name="Log20", filename="log.xes", fmt=fmt_value)
        mock_resp = MagicMock()
        mock_resp.content = b"x"
        mock_resp.raise_for_status = MagicMock()
        ext = fmt_value.strip().lower()
        with (
            patch("evaluation.log_downloader._resolve_file_link",
                  return_value=(f"https://example.org/log.{ext}", f"log.{ext}")),
            patch("evaluation.log_downloader.requests.get", return_value=mock_resp),
        ):
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
# _extract_from_zip
# ---------------------------------------------------------------------------

class TestExtractFromZip:
    def _make_zip(self, entries: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in entries.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_extracts_matching_extension(self):
        content = self._make_zip({"data.xes": "<log/>"})
        result = _extract_from_zip(content, "xes")
        assert result == b"<log/>"

    def test_prefers_top_level_over_nested(self):
        content = self._make_zip({
            "nested/dir/data.xes": "<nested/>",
            "data.xes": "<top/>",
        })
        result = _extract_from_zip(content, "xes")
        assert result == b"<top/>"

    def test_falls_back_to_nested_if_no_top_level(self):
        content = self._make_zip({"nested/dir/data.xes": "<nested/>"})
        result = _extract_from_zip(content, "xes")
        assert result == b"<nested/>"

    def test_raises_when_extension_not_found(self):
        content = self._make_zip({"data.csv": "a,b\n1,2\n"})
        with pytest.raises(ValueError, match="No xes file found"):
            _extract_from_zip(content, "xes")


# ---------------------------------------------------------------------------
# _find_filename
# ---------------------------------------------------------------------------

class TestFindFilename:
    def test_exact_match(self):
        assert _find_filename("MyLog.xes", "MyLog.xes") is True

    def test_case_insensitive_match(self):
        assert _find_filename("mylog.xes", "MyLog.xes") is True

    def test_space_to_underscore_variant_matches(self):
        assert _find_filename("My_Log.xes", "My Log.xes") is True

    def test_no_match_returns_false(self):
        assert _find_filename("OtherLog.xes", "MyLog.xes") is False


# ---------------------------------------------------------------------------
# _resolve_file_link — DOI landing-page scraping
# ---------------------------------------------------------------------------

class TestResolveFileLink:
    def _mock_response(self, html: str, url: str = "https://data.4tu.nl/articles/abc"):
        resp = MagicMock()
        resp.text = html
        resp.url = url
        resp.raise_for_status = MagicMock()
        return resp

    def test_exact_text_match_inside_files_div(self):
        html = """
        <html><body>
          <div id="files">
            <a href="/ndownloader/files/111" id="dl-1">MyLog.xes</a>
          </div>
        </body></html>
        """
        with patch("evaluation.log_downloader.requests.get", return_value=self._mock_response(html)):
            url, actual_filename = _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])

        assert actual_filename == "MyLog.xes"
        assert url.endswith("/ndownloader/files/111")

    def test_extension_fallback_inside_files_div(self):
        html = """
        <html><body>
          <div id="files">
            <a href="/ndownloader/files/222" id="dl-1">MyLog.xes.gz</a>
          </div>
        </body></html>
        """
        with patch("evaluation.log_downloader.requests.get", return_value=self._mock_response(html)):
            url, actual_filename = _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])

        assert actual_filename == "MyLog.xes.gz"

    def test_generic_page_fallback_without_files_div(self):
        html = """
        <html><body>
          <a href="https://zenodo.org/records/123/files/MyLog.xes">MyLog.xes</a>
        </body></html>
        """
        with patch("evaluation.log_downloader.requests.get", return_value=self._mock_response(html)):
            url, actual_filename = _resolve_file_link("10.5281/zenodo.123", ["MyLog.xes"])

        assert url == "https://zenodo.org/records/123/files/MyLog.xes"

    def test_download_all_files_link_is_skipped(self):
        html = """
        <html><body>
          <div id="files">
            <a href="/download-all" id="download-all-files">Download all files</a>
            <a href="/ndownloader/files/333" id="dl-1">MyLog.xes</a>
          </div>
        </body></html>
        """
        with patch("evaluation.log_downloader.requests.get", return_value=self._mock_response(html)):
            url, actual_filename = _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])

        assert url.endswith("/ndownloader/files/333")

    def test_relative_href_resolved_to_absolute(self):
        html = """
        <html><body>
          <div id="files">
            <a href="/ndownloader/files/444">MyLog.xes</a>
          </div>
        </body></html>
        """
        with patch(
            "evaluation.log_downloader.requests.get",
            return_value=self._mock_response(html, url="https://data.4tu.nl/articles/abc"),
        ):
            url, _ = _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])

        assert url == "https://data.4tu.nl/ndownloader/files/444"

    def test_no_matching_link_raises_value_error(self):
        html = "<html><body><div id=\"files\"></div></body></html>"
        with patch("evaluation.log_downloader.requests.get", return_value=self._mock_response(html)):
            with pytest.raises(ValueError, match="not found on landing page"):
                _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])

    def test_http_error_propagates(self):
        resp = self._mock_response("<html></html>")
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("evaluation.log_downloader.requests.get", return_value=resp):
            with pytest.raises(requests.HTTPError):
                _resolve_file_link("10.4121/uuid:abc", ["MyLog.xes"])
