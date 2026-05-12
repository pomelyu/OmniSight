import hashlib
import os
import re


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


# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
def parse_file_hash(file_name: str) -> str:
    r = HASH_REGEX.search(file_name)  # r is Optional[Match[str]]
    hash_prefix = r.group(1) if r else None
    return hash_prefix
