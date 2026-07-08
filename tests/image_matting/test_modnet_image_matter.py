from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest


def _make_mock_session() -> MagicMock:
    """Return a mock ort.InferenceSession that yields a constant 0.5 matte.

    The matte shape is derived from the input blob so the mock works for any
    image size fed through the pipeline.
    """
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_outputs.return_value = [MagicMock(name="output")]

    def _run(output_names, inputs):
        blob = next(iter(inputs.values()))
        n, _, h, w = blob.shape
        return [np.full((n, 1, h, w), 0.5, dtype=np.float32)]

    session.run.side_effect = _run
    return session


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """Return a synthetic RGB uint8 image for testing."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def matter():
    """Return a MODNetImageMatter with a mocked ONNX session."""
    with patch(
        "omni_sight.third_party.modnet.modnet_image_matter._modnet_loader.get_onnx_session",
        return_value=_make_mock_session(),
    ):
        from omni_sight.image_matting import MODNetImageMatter
        return MODNetImageMatter(device="cpu")


# ---------------------------------------------------------------------------
# _resolve_input_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        # already straddles 512 and divisible by 32 → unchanged
        (480, 640, (480, 640)),
        # straddles 512 but not divisible by 32 → floored to multiples of 32
        (500, 600, (480, 576)),
        # small image → shorter side scaled up to 512
        (200, 300, (512, 768)),
        # large image → shorter side scaled down to 512
        (1000, 2000, (512, 1024)),
        # extremely thin image → clamped to at least one stride (32)
        (20, 800, (32, 800)),
    ],
)
def test_resolve_input_size(matter, height: int, width: int, expected: tuple) -> None:
    assert matter._resolve_input_size(height, width) == expected


def test_resolve_input_size_is_stride_aligned(matter) -> None:
    """Resolved dims must always be positive multiples of 32."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        h, w = rng.integers(1, 4000, size=2)
        rh, rw = matter._resolve_input_size(int(h), int(w))
        assert rh > 0 and rw > 0
        assert rh % 32 == 0 and rw % 32 == 0


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_produces_nchw_blob(matter, rgb_image: np.ndarray) -> None:
    result = matter.preprocess(rgb_image)
    assert result["blob"].shape == (1, 3, 480, 640)


def test_preprocess_blob_is_float32(matter, rgb_image: np.ndarray) -> None:
    result = matter.preprocess(rgb_image)
    assert result["blob"].dtype == np.float32


def test_preprocess_normalizes_to_unit_range(matter, rgb_image: np.ndarray) -> None:
    blob = matter.preprocess(rgb_image)["blob"]
    assert blob.min() >= -1.0
    assert blob.max() <= 1.0


def test_preprocess_records_original_size(matter, rgb_image: np.ndarray) -> None:
    result = matter.preprocess(rgb_image)
    assert result["original_size"] == (rgb_image.shape[0], rgb_image.shape[1])


def test_preprocess_no_channel_swap(matter) -> None:
    """Confirm preprocess does not internally swap channels — RGB input stays RGB."""
    red_image = np.zeros((480, 640, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # R=255, G=0, B=0
    blob = matter.preprocess(red_image)["blob"][0]  # (3, H, W)
    assert blob[0].mean() > blob[1].mean(), "Channel 0 (R) should dominate for a red image"
    assert blob[0].mean() > blob[2].mean(), "Channel 0 (R) should dominate for a red image"


def test_preprocess_rejects_non_uint8(matter) -> None:
    with pytest.raises(ValueError, match="uint8"):
        matter.preprocess(np.zeros((64, 64, 3), dtype=np.float32))


def test_preprocess_rejects_wrong_channels(matter) -> None:
    with pytest.raises(ValueError):
        matter.preprocess(np.zeros((64, 64, 1), dtype=np.uint8))


def test_preprocess_rejects_grayscale(matter) -> None:
    with pytest.raises(ValueError):
        matter.preprocess(np.zeros((64, 64), dtype=np.uint8))


# ---------------------------------------------------------------------------
# run (end-to-end with mocked session)
# ---------------------------------------------------------------------------


def test_output_shape_matches_input(matter, rgb_image: np.ndarray) -> None:
    matte = matter.run(rgb_image)
    assert matte.shape == rgb_image.shape[:2]


def test_output_dtype_is_uint8(matter, rgb_image: np.ndarray) -> None:
    matte = matter.run(rgb_image)
    assert matte.dtype == np.uint8


def test_output_values_follow_model_matte(matter, rgb_image: np.ndarray) -> None:
    """A constant 0.5 model matte must decode to a constant 128 alpha map."""
    matte = matter.run(rgb_image)
    assert np.all(matte == 128)


def test_output_shape_for_small_image(matter) -> None:
    """Images below the reference size are upscaled internally but the matte
    must come back at the original resolution."""
    small_image = np.zeros((200, 300, 3), dtype=np.uint8)
    matte = matter.run(small_image)
    assert matte.shape == (200, 300)


def test_output_shape_for_large_image(matter) -> None:
    large_image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    matte = matter.run(large_image)
    assert matte.shape == (1000, 2000)
