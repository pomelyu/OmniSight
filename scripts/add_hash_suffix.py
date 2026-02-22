import argparse
from pathlib import Path

from omni_sight.utils.hash import get_sha256_hash


def _build_output_path(file_path: Path, hash_prefix: str) -> Path:
    return file_path.with_name(f"{file_path.stem}-{hash_prefix}{file_path.suffix}")


def main() -> None:
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
