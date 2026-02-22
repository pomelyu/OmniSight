import hashlib
from pathlib import Path

import pytest

from omni_sight.utils.hash import get_sha256_hash


def test_get_sha256_hash_with_fixture_file() -> None:
	fixture_path = Path(__file__).resolve().parents[1] / "resources" / "one_girl.jpg"

	expected_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
	actual_hash = get_sha256_hash(str(fixture_path))

	assert actual_hash == expected_hash


def test_get_sha256_hash_raises_for_missing_file() -> None:
	missing_file_path = Path(__file__).resolve().parents[1] / "resources" / "not_found.jpg"

	with pytest.raises(FileNotFoundError):
		get_sha256_hash(str(missing_file_path))


def test_get_sha256_hash_raises_for_empty_path() -> None:
	with pytest.raises(ValueError):
		get_sha256_hash("")
