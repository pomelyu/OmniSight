import hashlib
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from omni_sight.utils.file_loader import FileLoader
from omni_sight.utils.file_loader import download_url_to_file

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "omnisight"

# ---------------------------------------------------------------------------
# FileLoader.__init__
# ---------------------------------------------------------------------------


def test_file_loader_default_folder() -> None:
    loader = FileLoader()
    assert loader.folder == _DEFAULT_CACHE_DIR


def test_file_loader_custom_folder(tmp_path: Path) -> None:
    loader = FileLoader(folder=str(tmp_path))
    assert loader.folder == tmp_path


# ---------------------------------------------------------------------------
# FileLoader.set_file / add
# ---------------------------------------------------------------------------


def test_set_file_stores_metadata() -> None:
    loader = FileLoader()
    loader.set_file(
        "model-abc12345.onnx",
        urls=["https://example.com/model.onnx"],
        identifier="m",
    )
    assert loader.file_names["m"] == "model-abc12345.onnx"
    assert loader.file_urls["m"] == ["https://example.com/model.onnx"]


def test_add_delegates_to_set_file() -> None:
    loader = FileLoader()
    loader.set_file(
        "model-abc12345.onnx",
        urls=["https://example.com/model.onnx"],
        identifier="m",
    )
    assert loader.file_names["m"] == "model-abc12345.onnx"


def test_set_file_raises_for_empty_name() -> None:
    loader = FileLoader()
    with pytest.raises(ValueError):
        loader.set_file("", urls=["https://example.com/model.onnx"])


def test_set_file_raises_for_empty_urls() -> None:
    loader = FileLoader()
    with pytest.raises(ValueError):
        loader.set_file("model-abc12345.onnx", urls=[])


# ---------------------------------------------------------------------------
# FileLoader.get_path
# ---------------------------------------------------------------------------


def test_get_path_raises_for_unknown_identifier() -> None:
    loader = FileLoader()
    with pytest.raises(KeyError):
        loader.get_path("unknown")


def test_get_path_returns_existing_file(tmp_path: Path) -> None:
    file_name = "model-abc12345.onnx"
    file_path = tmp_path / file_name
    file_path.write_bytes(b"fake model")

    loader = FileLoader(folder=str(tmp_path))
    loader.set_file(file_name, urls=["https://example.com/model.onnx"], identifier="m")

    assert loader.get_path("m") == file_path


def test_get_path_downloads_when_file_is_missing(tmp_path: Path) -> None:
    file_name = "model-abc12345.onnx"
    loader = FileLoader(folder=str(tmp_path))
    loader.set_file(file_name, urls=["https://example.com/model.onnx"], identifier="m")

    def fake_download(url: str, destination: Path, hash_prefix: Optional[str] = None) -> None:
        destination.write_bytes(b"fake model")

    with patch("omni_sight.utils.file_loader.download_url_to_file", side_effect=fake_download) as mock_dl:
        result = loader.get_path("m")

    assert result == tmp_path / file_name
    mock_dl.assert_called_once()


def test_get_path_tries_fallback_url_on_failure(tmp_path: Path) -> None:
    file_name = "model-abc12345.onnx"
    loader = FileLoader(folder=str(tmp_path))
    loader.set_file(
        file_name,
        urls=["https://bad.example.com/model.onnx", "https://good.example.com/model.onnx"],
        identifier="m",
    )

    call_count = 0

    def fake_download(url: str, destination: Path, hash_prefix: Optional[str] = None) -> None:
        nonlocal call_count
        call_count += 1
        if "bad" in url:
            raise RuntimeError("connection failed")
        destination.write_bytes(b"fake model")

    with patch("omni_sight.utils.file_loader.download_url_to_file", side_effect=fake_download):
        result = loader.get_path("m")

    assert result == tmp_path / file_name
    assert call_count == 2


def test_get_path_raises_when_all_urls_fail(tmp_path: Path) -> None:
    file_name = "model-abc12345.onnx"
    loader = FileLoader(folder=str(tmp_path))
    loader.set_file(file_name, urls=["https://bad.example.com/model.onnx"], identifier="m")

    with patch("omni_sight.utils.file_loader.download_url_to_file", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError, match="Unable to download"):
            loader.get_path("m")


# ---------------------------------------------------------------------------
# download_url_to_file
# ---------------------------------------------------------------------------


def _make_mock_response(content: bytes) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = content
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_download_writes_content(tmp_path: Path) -> None:
    content = b"fake model content"
    destination = tmp_path / "model.onnx"

    with patch("urllib.request.urlopen", return_value=_make_mock_response(content)):
        download_url_to_file("https://example.com/model.onnx", destination)

    assert destination.read_bytes() == content


def test_download_passes_hash_verification(tmp_path: Path) -> None:
    content = b"fake model content"
    destination = tmp_path / "model.onnx"
    correct_prefix = hashlib.sha256(content).hexdigest()[:8]

    with patch("urllib.request.urlopen", return_value=_make_mock_response(content)):
        download_url_to_file("https://example.com/model.onnx", destination, hash_prefix=correct_prefix)

    assert destination.exists()


def test_download_skips_hash_check_when_prefix_is_none(tmp_path: Path) -> None:
    content = b"fake model content"
    destination = tmp_path / "model.onnx"

    with patch("urllib.request.urlopen", return_value=_make_mock_response(content)):
        download_url_to_file("https://example.com/model.onnx", destination, hash_prefix=None)

    assert destination.exists()
