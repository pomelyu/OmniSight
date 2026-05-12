from pathlib import Path

import pytest

from omni_sight.utils.file_loader import FileLoader


def test_get_path_returns_existing_file_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return existing local file and skip download attempts."""
    model_name = "scrfd_1g_shape400x400-6a6ba473.onnx"
    existing_file = tmp_path / model_name
    existing_file.write_bytes(b"ready")

    file_loader = FileLoader(folder=str(tmp_path))
    file_loader.add(model_name, urls=["https://example.com/model.onnx"], identifier="scrfd")

    download_calls = []

    def _fake_download(*_args, **_kwargs) -> None:
        download_calls.append(1)

    monkeypatch.setattr("omni_sight.utils.file_loader.download_url_to_file", _fake_download)

    actual_path = file_loader.get_path(identifier="scrfd")

    assert actual_path == existing_file
    assert download_calls == []


def test_get_path_downloads_until_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Try candidate URLs in order and return path for first successful download."""
    model_name = "scrfd_1g_shape400x400-6a6ba473.onnx"
    urls = ["https://example.com/bad.onnx", "https://example.com/good.onnx"]

    file_loader = FileLoader(folder=str(tmp_path))
    file_loader.set_file(model_name, urls=urls, identifier="scrfd")

    called_urls = []

    def _fake_download(url: str, path: Path, hash_prefix: str = None) -> None:
        called_urls.append(url)
        if "bad" in url:
            raise RuntimeError("network error")

        assert hash_prefix == "6a6ba473"
        path.write_bytes(b"downloaded")

    monkeypatch.setattr("omni_sight.utils.file_loader.download_url_to_file", _fake_download)

    actual_path = file_loader.get_path(identifier="scrfd")

    assert actual_path == tmp_path / model_name
    assert actual_path.exists()
    assert called_urls == urls


def test_get_path_raises_for_unknown_identifier(tmp_path: Path) -> None:
    """Raise a clear error for unknown identifiers."""
    file_loader = FileLoader(folder=str(tmp_path))

    with pytest.raises(KeyError, match="Unknown identifier"):
        file_loader.get_path(identifier="missing")


def test_get_path_raises_when_all_downloads_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeError when all download candidates fail."""
    model_name = "scrfd_1g_shape400x400-6a6ba473.onnx"
    file_loader = FileLoader(folder=str(tmp_path))
    file_loader.add(model_name, urls=["https://example.com/fail.onnx"], identifier="scrfd")

    def _fake_download(_url: str, _path: Path, hash_prefix: str = None) -> None:
        raise RuntimeError("always failing")

    monkeypatch.setattr("omni_sight.utils.file_loader.download_url_to_file", _fake_download)

    with pytest.raises(RuntimeError, match="Unable to download"):
        file_loader.get_path(identifier="scrfd")


def test_set_file_validates_inputs(tmp_path: Path) -> None:
    """Validate required inputs before persisting file metadata."""
    file_loader = FileLoader(folder=str(tmp_path))

    with pytest.raises(ValueError, match="file_name"):
        file_loader.set_file("", urls=["https://example.com/model.onnx"])

    with pytest.raises(ValueError, match="urls"):
        file_loader.set_file("model.onnx", urls=[])
