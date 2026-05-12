
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from torch.hub import download_url_to_file as _torch_download_url_to_file
except ImportError:  # pragma: no cover - exercised only when torch is missing
    _torch_download_url_to_file = None

from omni_sight import logger

from .hash import parse_file_hash


def download_url_to_file(url: str, destination: Path, hash_prefix: Optional[str] = None) -> None:
    """Download a file to destination and optionally verify a SHA256 prefix."""
    if _torch_download_url_to_file is not None:
        _torch_download_url_to_file(url, destination, hash_prefix=hash_prefix)
        return

    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())

    if hash_prefix:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not digest.startswith(hash_prefix):
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file hash does not match expected prefix '{hash_prefix}'"
            )

# Usage:
# file_loader = FileLoader()
# file_loader.add("scrfd_1g_shape400x400-6a6ba473.onnx", urls=[http://], identifier="scrfd_1g_shape400x400")
# file_path = file_loader.get_path(identifier="scrfd_1g_shape400x400")

class FileLoader():
    """Load local files or download them from configured URLs."""

    def __init__(self, folder: str = None):
        self.file_urls: Dict[str, List[str]] = {}
        self.file_names: Dict[str, str] = {}

        if folder is not None:
            self.folder = Path(folder)
        else:
            self.folder = Path("")  # TODO: ~/.cache/omni_sight

    def add(self, file_name: str, urls: List[str], identifier: str = "default") -> None:
        """Register a file mapping for later retrieval."""
        self.set_file(file_name=file_name, urls=urls, identifier=identifier)

    def set_file(self, file_name: str, urls: List[str], identifier: str = "default") -> None:
        """Store file metadata keyed by identifier."""
        if not file_name:
            raise ValueError("file_name must be a non-empty string")
        if not urls:
            raise ValueError("urls must contain at least one candidate URL")

        self.file_names[identifier] = file_name
        self.file_urls[identifier] = urls

    def get_path(self, identifier: str = "default") -> Path:
        """Return local file path, downloading from configured URLs when needed."""
        if identifier not in self.file_names:
            raise KeyError(f"Unknown identifier: {identifier}")

        file_name = self.file_names[identifier]
        file_hash = parse_file_hash(file_name)
        file_path: Path = self.folder / file_name

        # 1. check if file exist in self.folder return the full path if exists
        if file_path.exists():
            return file_path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. download the file from url
        for candidate_url in self.file_urls[identifier]:
            try:
                download_url_to_file(candidate_url, file_path, hash_prefix=file_hash)
                return file_path
            except Exception:
                logger.info(f"Failed to download from {candidate_url}. Try next")

        # 3. raise error if all attempt fails
        raise RuntimeError(
            f"Unable to download '{file_name}' for identifier '{identifier}' from configured URLs"
        )
