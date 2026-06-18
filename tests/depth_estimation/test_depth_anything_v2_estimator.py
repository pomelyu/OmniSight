from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

_MODEL_OUTPUT_SHAPE = (1, 518, 518)  # batch, H, W — shape returned by the ONNX model


def _make_mock_session(output_shape: tuple = _MODEL_OUTPUT_SHAPE) -> MagicMock:
    """Return a mock ort.InferenceSession that yields a synthetic depth map."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="image")]
    session.get_outputs.return_value = [MagicMock(name="depth")]
    session.run.return_value = [np.random.rand(*output_shape).astype(np.float32)]
    return session


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """Return a synthetic RGB uint8 image for testing."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def estimator(rgb_image: np.ndarray):
    """Return a DepthAnythingV2Estimator with a mocked ONNX session."""
    mock_session = _make_mock_session()
    with patch(
        "omni_sight.third_party.depth_anything_v2.depth_anything_v2_estimator._dav2_loader.get_onnx_session",
        return_value=mock_session,
    ):
        from omni_sight.depth_estimation import DepthAnythingV2Estimator
        return DepthAnythingV2Estimator(device="cpu", model_name="depth_anything_v2_small")


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_produces_correct_blob_shape(estimator, rgb_image: np.ndarray) -> None:
    result = estimator.preprocess(rgb_image)
    assert result["blob"].shape == (1, 3, 518, 518)


def test_preprocess_blob_is_float32(estimator, rgb_image: np.ndarray) -> None:
    result = estimator.preprocess(rgb_image)
    assert result["blob"].dtype == np.float32


def test_preprocess_records_original_size(estimator, rgb_image: np.ndarray) -> None:
    result = estimator.preprocess(rgb_image)
    assert result["original_size"] == (rgb_image.shape[0], rgb_image.shape[1])


def test_preprocess_no_channel_swap(estimator) -> None:
    """Confirm preprocess does not internally swap channels — RGB input stays RGB."""
    # A pure-red RGB image should produce a blob whose first channel (R) has
    # a much higher mean than the other two after normalization.
    red_image = np.zeros((64, 64, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # R=255, G=0, B=0
    result = estimator.preprocess(red_image)
    blob = result["blob"][0]  # (3, H, W)
    assert blob[0].mean() > blob[1].mean(), "Channel 0 (R) should dominate for a red image"
    assert blob[0].mean() > blob[2].mean(), "Channel 0 (R) should dominate for a red image"


def test_preprocess_rejects_bgr_uint8_check(estimator) -> None:
    with pytest.raises(ValueError, match="uint8"):
        estimator.preprocess(np.zeros((64, 64, 3), dtype=np.float32))


def test_preprocess_rejects_wrong_channels(estimator) -> None:
    with pytest.raises(ValueError):
        estimator.preprocess(np.zeros((64, 64, 1), dtype=np.uint8))


# ---------------------------------------------------------------------------
# run (end-to-end with mocked session)
# ---------------------------------------------------------------------------


def test_output_shape_matches_input(estimator, rgb_image: np.ndarray) -> None:
    depth = estimator.run(rgb_image)
    assert depth.shape == rgb_image.shape[:2]


def test_output_dtype_is_float32(estimator, rgb_image: np.ndarray) -> None:
    depth = estimator.run(rgb_image)
    assert depth.dtype == np.float32


def test_output_is_finite(estimator, rgb_image: np.ndarray) -> None:
    depth = estimator.run(rgb_image)
    assert np.all(np.isfinite(depth))


def test_output_shape_for_non_square_image(estimator) -> None:
    """Verify the output is resized back to the original non-square dimensions."""
    wide_image = np.zeros((240, 640, 3), dtype=np.uint8)
    depth = estimator.run(wide_image)
    assert depth.shape == (240, 640)
