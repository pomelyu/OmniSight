import hashlib
import os


def get_sha256_hash(file_path: str) -> str:
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
