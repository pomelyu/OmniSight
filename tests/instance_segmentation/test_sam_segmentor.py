from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

_IMAGE_H, _IMAGE_W = 480, 640
_ENCODER_INPUT_H = 684
_ENCODER_INPUT_W = 1024
_ENC_EMBED_SHAPE = (1, 256, 64, 64)
_NUM_MASKS = 3
_NUM_MASKS_MULTI = 10  # multi-output decoder returns more than 3 candidates

_ALL_BASE_VARIANTS = ["mobile_sam", "sam_vit_b", "sam_vit_l", "sam_vit_h"]
_ALL_MULTI_VARIANTS = ["mobile_sam-m", "sam_vit_b-m", "sam_vit_l-m", "sam_vit_h-m"]
_ALL_VARIANTS = _ALL_BASE_VARIANTS + _ALL_MULTI_VARIANTS


def _make_mock_encoder(embed_shape: tuple = _ENC_EMBED_SHAPE) -> MagicMock:
    """Return a mock encoder session yielding a synthetic image embedding."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input_image")]
    session.get_outputs.return_value = [MagicMock(name="image_embeddings")]
    session.run.return_value = [np.random.rand(*embed_shape).astype(np.float32)]
    return session


def _make_mock_decoder(num_masks: int = _NUM_MASKS) -> MagicMock:
    """Return a mock decoder session yielding synthetic masks and IoU scores.

    Masks are returned at the encoder canvas size (684×1024) to match the
    samexporter convention where orig_im_size is fixed at
    (_ENCODER_INPUT_H, _ENCODER_INPUT_W).

    Args:
        num_masks: Number of mask candidates the decoder should output.
            Use ``_NUM_MASKS`` (3) for base variants and
            ``_NUM_MASKS_MULTI`` (10) for ``-m`` multi-output variants.
    """
    session = MagicMock()
    dec_inputs = [
        "image_embeddings",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size",
    ]
    session.get_inputs.return_value = [MagicMock(name=n) for n in dec_inputs]
    dec_outputs = ["masks", "iou_predictions", "low_res_logits"]
    session.get_outputs.return_value = [MagicMock(name=n) for n in dec_outputs]
    masks = np.random.rand(num_masks, _ENCODER_INPUT_H, _ENCODER_INPUT_W).astype(np.float32)
    masks = (masks - 0.3).astype(np.float32)
    session.run.return_value = [
        masks[np.newaxis],
        np.random.rand(1, num_masks).astype(np.float32),
        np.random.rand(1, num_masks, 256, 256).astype(np.float32),
    ]
    return session


def _make_segmentor(model_name: str, num_masks: int):
    """Return a SAMSegmentor with mocked sessions for the given model variant."""
    mock_encoder = _make_mock_encoder()
    mock_decoder = _make_mock_decoder(num_masks=num_masks)
    with (
        patch(
            "omni_sight.third_party.segment_anything.sam_segmentor._sam_encoder_loader.get_onnx_session",
            return_value=mock_encoder,
        ),
        patch(
            "omni_sight.third_party.segment_anything.sam_segmentor._sam_decoder_loader.get_onnx_session",
            return_value=mock_decoder,
        ),
    ):
        from omni_sight.instance_segmentation import SAMSegmentor
        return SAMSegmentor(device="cpu", model_name=model_name)


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """Return a synthetic RGB uint8 image for testing."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)


@pytest.fixture()
def segmentor():
    """Return a SAMSegmentor (mobile_sam base variant) with mocked sessions."""
    return _make_segmentor("mobile_sam", _NUM_MASKS)


@pytest.fixture()
def segmentor_multi():
    """Return a SAMSegmentor (sam_vit_b-m multi variant) with mocked sessions."""
    return _make_segmentor("sam_vit_b-m", _NUM_MASKS_MULTI)


# ---------------------------------------------------------------------------
# Constructor — model name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", _ALL_VARIANTS)
def test_constructor_accepts_all_variants(model_name: str) -> None:
    num_masks = _NUM_MASKS_MULTI if model_name.endswith("-m") else _NUM_MASKS
    seg = _make_segmentor(model_name, num_masks)
    assert seg is not None


def test_constructor_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model_name"):
        with (
            patch(
                "omni_sight.third_party.segment_anything.sam_segmentor._sam_encoder_loader.get_onnx_session",
            ),
            patch(
                "omni_sight.third_party.segment_anything.sam_segmentor._sam_decoder_loader.get_onnx_session",
            ),
        ):
            from omni_sight.instance_segmentation import SAMSegmentor
            SAMSegmentor(device="cpu", model_name="sam_vit_x")


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_blob_shape(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["blob"].shape == (_ENCODER_INPUT_H, _ENCODER_INPUT_W, 3)


def test_preprocess_blob_dtype(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["blob"].dtype == np.float32


def test_preprocess_records_original_size(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["original_size"] == (_IMAGE_H, _IMAGE_W)


def test_preprocess_transform_matrix_shape(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["transform_matrix"].shape == (3, 3)


def test_preprocess_transform_matrix_is_uniform_scale(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    m = result["transform_matrix"]
    assert m[0, 0] == m[1, 1]
    assert m[0, 0] > 0.0
    assert m[0, 2] == 0.0 and m[1, 2] == 0.0


def test_preprocess_rejects_non_uint8(segmentor) -> None:
    with pytest.raises(ValueError, match="uint8"):
        segmentor.preprocess(np.zeros((64, 64, 3), dtype=np.float32))


def test_preprocess_rejects_wrong_channels(segmentor) -> None:
    with pytest.raises(ValueError):
        segmentor.preprocess(np.zeros((64, 64, 1), dtype=np.uint8))


# ---------------------------------------------------------------------------
# run — base variant (3 masks)
# ---------------------------------------------------------------------------


def test_run_with_point_prompt_mask_shape(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)


def test_run_with_point_prompt_mask_dtype(segmentor, rgb_image: np.ndarray) -> None:
    masks, _ = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.dtype == bool


def test_run_with_point_prompt_scores_shape(segmentor, rgb_image: np.ndarray) -> None:
    _, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert scores.shape == (_NUM_MASKS,)
    assert scores.dtype == np.float32


def test_run_scores_are_sorted_descending(segmentor, rgb_image: np.ndarray) -> None:
    _, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert np.all(scores[:-1] >= scores[1:])


def test_run_with_box_prompt(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        box=np.array([100, 80, 400, 320], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)
    assert scores.shape == (_NUM_MASKS,)


def test_run_with_combined_prompt(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[200, 150]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
        box=np.array([100, 80, 400, 320], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)


# ---------------------------------------------------------------------------
# run — multi-output variant (N > 3 masks)
# ---------------------------------------------------------------------------


def test_run_multi_mask_count(segmentor_multi, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor_multi.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS_MULTI, _IMAGE_H, _IMAGE_W)
    assert scores.shape == (_NUM_MASKS_MULTI,)


def test_run_multi_mask_dtype(segmentor_multi, rgb_image: np.ndarray) -> None:
    masks, _ = segmentor_multi.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.dtype == bool


def test_run_multi_scores_sorted_descending(segmentor_multi, rgb_image: np.ndarray) -> None:
    _, scores = segmentor_multi.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert np.all(scores[:-1] >= scores[1:])


def test_run_multi_returns_more_masks_than_base(rgb_image: np.ndarray) -> None:
    seg_base = _make_segmentor("sam_vit_b", _NUM_MASKS)
    seg_multi = _make_segmentor("sam_vit_b-m", _NUM_MASKS_MULTI)
    masks_base, _ = seg_base.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    masks_multi, _ = seg_multi.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks_multi.shape[0] > masks_base.shape[0]


# ---------------------------------------------------------------------------
# run — input validation
# ---------------------------------------------------------------------------


def test_run_raises_without_prompt(segmentor, rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segmentor.run(rgb_image)


# ---------------------------------------------------------------------------
# draw_mask utility (tested here for convenience)
# ---------------------------------------------------------------------------


def test_draw_mask_modifies_image_in_place() -> None:
    from omni_sight.utils.visual import draw_mask

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    result = draw_mask(image, mask, color=(0, 255, 0), alpha=0.5)
    assert result is image
    assert image[50, 50, 1] > 0


def test_draw_mask_rejects_wrong_ndim() -> None:
    from omni_sight.utils.visual import draw_mask

    with pytest.raises(ValueError, match="2-D"):
        draw_mask(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 1), dtype=bool),
        )
