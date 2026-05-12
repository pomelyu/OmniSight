import hashlib
import urllib.request
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from omni_sight import logger

from .hash import parse_file_hash

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "omnisight"


def download_url_to_file(url: str, destination: Path, hash_prefix: Optional[str] = None) -> None:
    """Download a URL to a local file and optionally verify its SHA-256 prefix.

    Args:
        url: Source URL to download.
        destination: Local path where the file will be written.
        hash_prefix: Expected SHA-256 hex prefix for integrity verification.
            Pass ``None`` to skip verification.

    Raises:
        RuntimeError: If the downloaded file's hash does not start with
            ``hash_prefix``.
    """
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())
        logger.info(f"Download file to {destination}")

    if hash_prefix:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not digest.startswith(hash_prefix):
            destination.unlink(missing_ok=True)
            logger.warning(
                f"Downloaded file hash does not match expected prefix '{hash_prefix}'"
            )


class FileLoader:
    """Load local files or download them from configured URLs.

    Example:
        >>> loader = FileLoader()
        >>> loader.set_file(
        ...     "scrfd_1g_shape400x400-6a6ba473.onnx",
        ...     urls=["https://example.com/scrfd_1g.onnx"],
        ...     identifier="scrfd_1g",
        ... )
        >>> path = loader.get_path("scrfd_1g")
    """

    def __init__(self, folder: Optional[str] = None) -> None:
        """Initialize the loader with a local cache directory.

        Args:
            folder: Directory where files are cached. Defaults to
                ``~/.cache/omnisight`` when ``None``.
        """
        self.file_urls: Dict[str, List[str]] = {}
        self.file_names: Dict[str, str] = {}
        self.folder: Path = Path(folder) if folder is not None else _DEFAULT_CACHE_DIR
        self.folder.mkdir(parents=True, exist_ok=True)

    def get_available_files(self) -> Dict[str, str]:
        return self.file_names

    def set_file(self, file_name: str, urls: List[str], identifier: str = "default") -> None:
        """Store file metadata keyed by identifier.

        Args:
            file_name: Local file name including the hash suffix.
            urls: Candidate download URLs tried in order.
            identifier: Key used to retrieve the file later. Defaults to
                ``"default"``.

        Raises:
            ValueError: If ``file_name`` is empty or ``urls`` is empty.
        """
        if not file_name:
            raise ValueError("file_name must be a non-empty string")
        if not urls:
            raise ValueError("urls must contain at least one candidate URL")

        self.file_names[identifier] = file_name
        self.file_urls[identifier] = urls

    def get_path(self, identifier: str = "default") -> Path:
        """Return the local path for an identifier, downloading if necessary.

        Args:
            identifier: Key previously registered with :meth:`add`. Defaults
                to ``"default"``.

        Returns:
            Absolute path to the local file.

        Raises:
            KeyError: If ``identifier`` has not been registered.
            RuntimeError: If the file cannot be downloaded from any configured URL.
        """
        if identifier not in self.file_names:
            raise KeyError(f"Unknown identifier: {identifier}")

        file_name = self.file_names[identifier]
        file_path: Path = self.folder / file_name

        if file_path.exists():
            return file_path

        file_hash = parse_file_hash(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        for candidate_url in self.file_urls[identifier]:
            try:
                download_url_to_file(candidate_url, file_path, hash_prefix=file_hash)
                return file_path
            except Exception:
                logger.info(f"Failed to download from {candidate_url}. Try next")

        raise RuntimeError(
            f"Unable to download '{file_name}' for identifier '{identifier}' from configured URLs"
        )
