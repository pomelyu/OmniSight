import hashlib
import os
import re
from typing import Optional


def get_sha256_hash(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Lowercase hexadecimal SHA-256 digest string (64 characters).

    Raises:
        ValueError: If ``file_path`` is empty.
        FileNotFoundError: If no file exists at ``file_path``.
    """
    if not file_path:
        raise ValueError("file_path must be a non-empty string")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        # Read in chunks to keep memory usage low and support very large files.
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


HASH_REGEX = re.compile(r"-([a-f0-9]+)\.")


def parse_file_hash(file_name: str) -> Optional[str]:
    """Extract the hash prefix embedded in a file name.

    Args:
        file_name: File name containing an optional hash suffix,
            e.g. ``resnet18-bfd8deac.pth``.

    Returns:
        The hex hash string (e.g. ``"bfd8deac"``), or ``None`` if no
        hash pattern is found.
    """
    match = HASH_REGEX.search(file_name)
    return match.group(1) if match else None
