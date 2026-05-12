import argparse
from pathlib import Path

from omni_sight.utils.hash import get_sha256_hash


def _build_output_path(file_path: Path, hash_prefix: str) -> Path:
    """Construct the destination path by appending a hash prefix to the stem.

    Args:
        file_path: Original file path.
        hash_prefix: Short hash string to append (e.g. first 8 hex chars).

    Returns:
        New :class:`~pathlib.Path` with the hash prefix inserted before the
        file extension (e.g. ``model.onnx`` → ``model-abc12345.onnx``).
    """
    return file_path.with_name(f"{file_path.stem}-{hash_prefix}{file_path.suffix}")


def main() -> None:
    """Parse CLI arguments and rename the target file with a hash suffix.

    Raises:
        FileNotFoundError: If the source file does not exist (propagated from
            :func:`~omni_sight.utils.hash.get_sha256_hash`).
        FileExistsError: If a file with the target name already exists.
    """
    parser = argparse.ArgumentParser(
        description="Append first 8 chars of SHA-256 hash to a file name.",
    )
    parser.add_argument("file_path", help="Path to the file to rename")
    args = parser.parse_args()

    source_path = Path(args.file_path).resolve()
    hash_prefix = get_sha256_hash(str(source_path))[:8]
    target_path = _build_output_path(source_path, hash_prefix)

    if source_path == target_path:
        print(f"No rename needed: {source_path}")
        return

    if target_path.exists():
        raise FileExistsError(f"Target file already exists: {target_path}")

    source_path.rename(target_path)
    print(f"Renamed: {source_path} -> {target_path}")

if __name__ == "__main__":
    main()
