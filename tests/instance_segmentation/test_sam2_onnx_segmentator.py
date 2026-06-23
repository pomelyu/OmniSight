from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

_IMAGE_H, _IMAGE_W = 480, 640
_ENCODER_INPUT_SIZE = 1024
_MASK_INPUT_SIZE = 256
_NUM_MASKS = 3

_ENC_EMBED_SHAPE = (1, 256, 64, 64)
_HIGH_RES_0_SHAPE = (1, 32, 256, 256)
_HIGH_RES_1_SHAPE = (1, 64, 128, 128)

_ALL_MODEL_NAMES = [
    "sam2.1_tiny",
    "sam2.1_small",
    "sam2.1_base_plus",
    "sam2.1_large",
]


def _named(n: str) -> MagicMock:
    """Return a MagicMock whose ``.name`` attribute is the given string.

    ``MagicMock(name=n)`` sets only the repr name, not the ``.name`` attribute.
    Directly assigning ``m.name = n`` overrides the auto-generated attribute.
    """
    m = MagicMock()
    m.name = n
    return m


def _make_mock_encoder() -> MagicMock:
    """Return a mock SAM2 encoder session with 3 feature outputs."""
    session = MagicMock()
    session.get_inputs.return_value = [_named("image")]
    session.get_outputs.return_value = [
        _named("high_res_feats_0"),
        _named("high_res_feats_1"),
        _named("image_embed"),
    ]
    session.run.return_value = [
        np.random.rand(*_HIGH_RES_0_SHAPE).astype(np.float32),
        np.random.rand(*_HIGH_RES_1_SHAPE).astype(np.float32),
        np.random.rand(*_ENC_EMBED_SHAPE).astype(np.float32),
    ]
    return session


def _make_mock_decoder(num_masks: int = _NUM_MASKS) -> MagicMock:
    """Return a mock SAM2 decoder session.

    Masks are returned at the encoder canvas size (1024×1024) to match the
    samexporter SAM2 convention.

    Args:
        num_masks: Number of mask candidates the decoder should output.
    """
    session = MagicMock()
    dec_inputs = [
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    ]
    session.get_inputs.return_value = [_named(n) for n in dec_inputs]
    dec_outputs = ["masks", "iou_predictions"]
    session.get_outputs.return_value = [_named(n) for n in dec_outputs]
    masks = (np.random.rand(num_masks, _ENCODER_INPUT_SIZE, _ENCODER_INPUT_SIZE) - 0.3).astype(
        np.float32
    )
    session.run.return_value = [
        masks[np.newaxis],
        np.random.rand(1, num_masks).astype(np.float32),
    ]
    return session


def _make_segmentor(num_masks: int = _NUM_MASKS):
    """Return a SAM2OnnxSegmentator with mocked encoder and decoder sessions."""
    mock_encoder = _make_mock_encoder()
    mock_decoder = _make_mock_decoder(num_masks=num_masks)
    with (
        patch(
            "omni_sight.third_party.segment_anything.sam2_onnx_segmentator._sam2_encoder_loader.get_onnx_session",
            return_value=mock_encoder,
        ),
        patch(
            "omni_sight.third_party.segment_anything.sam2_onnx_segmentator._sam2_decoder_loader.get_onnx_session",
            return_value=mock_decoder,
        ),
    ):
        from omni_sight.instance_segmentation import SAM2OnnxSegmentator

        return SAM2OnnxSegmentator(
            device="cpu",
            encoder_path="fake_encoder.onnx",
            decoder_path="fake_decoder.onnx",
        )


@pytest.fixture()
def rgb_image() -> np.ndarray:
    """Return a synthetic RGB uint8 image for testing."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)


@pytest.fixture()
def segmentor():
    """Return a SAM2OnnxSegmentator with mocked sessions."""
    return _make_segmentor()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_requires_encoder_path() -> None:
    with pytest.raises(ValueError, match="encoder_path"):
        from omni_sight.instance_segmentation import SAM2OnnxSegmentator

        SAM2OnnxSegmentator(device="cpu", decoder_path="fake.onnx")


def test_constructor_requires_decoder_path() -> None:
    with pytest.raises(ValueError, match="decoder_path"):
        from omni_sight.instance_segmentation import SAM2OnnxSegmentator

        SAM2OnnxSegmentator(device="cpu", encoder_path="fake.onnx")


def test_constructor_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unknown model_name"):
        with (
            patch(
                "omni_sight.third_party.segment_anything.sam2_onnx_segmentator._sam2_encoder_loader.get_onnx_session",
                return_value=_make_mock_encoder(),
            ),
            patch(
                "omni_sight.third_party.segment_anything.sam2_onnx_segmentator._sam2_decoder_loader.get_onnx_session",
                return_value=_make_mock_decoder(),
            ),
        ):
            from omni_sight.instance_segmentation import SAM2OnnxSegmentator

            SAM2OnnxSegmentator(
                device="cpu",
                model_name="sam2.1_hiera_unknown",
                encoder_path="fake_encoder.onnx",
                decoder_path="fake_decoder.onnx",
            )


@pytest.mark.parametrize("model_name", _ALL_MODEL_NAMES)
def test_constructor_accepts_all_model_names(model_name: str) -> None:
    seg = _make_segmentor()
    assert seg is not None


def test_initial_state_is_none(segmentor) -> None:
    assert segmentor._prev_mask_logits is None


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_blob_shape(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["blob"].shape == (1, 3, _ENCODER_INPUT_SIZE, _ENCODER_INPUT_SIZE)


def test_preprocess_blob_dtype(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["blob"].dtype == np.float32


def test_preprocess_records_original_size(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["original_size"] == (_IMAGE_H, _IMAGE_W)


def test_preprocess_scale_factors(segmentor, rgb_image: np.ndarray) -> None:
    result = segmentor.preprocess(rgb_image)
    assert result["scale_x"] == pytest.approx(_ENCODER_INPUT_SIZE / _IMAGE_W)
    assert result["scale_y"] == pytest.approx(_ENCODER_INPUT_SIZE / _IMAGE_H)


def test_preprocess_rejects_non_uint8(segmentor) -> None:
    with pytest.raises(ValueError, match="uint8"):
        segmentor.preprocess(np.zeros((64, 64, 3), dtype=np.float32))


def test_preprocess_rejects_wrong_channels(segmentor) -> None:
    with pytest.raises(ValueError):
        segmentor.preprocess(np.zeros((64, 64, 1), dtype=np.uint8))


# ---------------------------------------------------------------------------
# run — stateless single-frame inference
# ---------------------------------------------------------------------------


def test_run_point_prompt_mask_shape(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)


def test_run_point_prompt_mask_dtype_bool(segmentor, rgb_image: np.ndarray) -> None:
    masks, _ = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.dtype == bool


def test_run_scores_shape(segmentor, rgb_image: np.ndarray) -> None:
    _, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert scores.shape == (_NUM_MASKS,)


def test_run_scores_sorted_descending(segmentor, rgb_image: np.ndarray) -> None:
    _, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert np.all(scores[:-1] >= scores[1:])


def test_run_box_prompt(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        box=np.array([100, 80, 400, 320], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)


def test_run_combined_prompt(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
        box=np.array([100, 80, 400, 320], dtype=np.float32),
    )
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)


def test_run_raises_without_prompt(segmentor, rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segmentor.run(rgb_image)


def test_run_does_not_mutate_state(segmentor, rgb_image: np.ndarray) -> None:
    """run() must not change _prev_mask_logits."""
    segmentor._prev_mask_logits = None
    segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert segmentor._prev_mask_logits is None


def test_run_preserves_existing_state(segmentor, rgb_image: np.ndarray) -> None:
    """run() must restore _prev_mask_logits that was set before the call."""
    sentinel = np.ones((1, 1, _MASK_INPUT_SIZE, _MASK_INPUT_SIZE), dtype=np.float32)
    segmentor._prev_mask_logits = sentinel
    segmentor.run(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert segmentor._prev_mask_logits is sentinel


# ---------------------------------------------------------------------------
# initialize — seeds state
# ---------------------------------------------------------------------------


def test_initialize_returns_masks_and_scores(segmentor, rgb_image: np.ndarray) -> None:
    masks, scores = segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert masks.shape[1:] == (_IMAGE_H, _IMAGE_W)
    assert scores.ndim == 1


def test_initialize_sets_prev_mask_logits(segmentor, rgb_image: np.ndarray) -> None:
    assert segmentor._prev_mask_logits is None
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert segmentor._prev_mask_logits is not None


def test_initialize_logits_shape(segmentor, rgb_image: np.ndarray) -> None:
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert segmentor._prev_mask_logits.shape == (1, 1, _MASK_INPUT_SIZE, _MASK_INPUT_SIZE)


def test_initialize_resets_prior_state(segmentor, rgb_image: np.ndarray) -> None:
    """initialize() must discard any existing _prev_mask_logits."""
    segmentor._prev_mask_logits = np.zeros((1, 1, _MASK_INPUT_SIZE, _MASK_INPUT_SIZE), dtype=np.float32)
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    # State should be updated (not the old zeros)
    assert segmentor._prev_mask_logits is not None


# ---------------------------------------------------------------------------
# propagate — uses state
# ---------------------------------------------------------------------------


def test_propagate_raises_before_initialize(segmentor, rgb_image: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="initialize"):
        segmentor.propagate(rgb_image)


def test_propagate_returns_masks_and_scores(segmentor, rgb_image: np.ndarray) -> None:
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    masks, scores = segmentor.propagate(rgb_image)
    assert masks.shape == (_NUM_MASKS, _IMAGE_H, _IMAGE_W)
    assert masks.dtype == bool
    assert scores.shape == (_NUM_MASKS,)


def test_propagate_updates_state(segmentor, rgb_image: np.ndarray) -> None:
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    logits_after_init = segmentor._prev_mask_logits.copy()
    segmentor.propagate(rgb_image)
    # State must be refreshed after propagate (different array object)
    assert segmentor._prev_mask_logits is not None


def test_propagate_passes_has_mask_input_true(segmentor, rgb_image: np.ndarray) -> None:
    """The decoder must receive has_mask_input=1 during propagation."""
    segmentor.initialize(
        rgb_image,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    segmentor.decoder_session.run.reset_mock()
    segmentor.propagate(rgb_image)
    call_args = segmentor.decoder_session.run.call_args
    inputs: dict = call_args[0][1]  # positional arg 1 is the feed dict
    assert inputs["has_mask_input"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# process_video — end-to-end
# ---------------------------------------------------------------------------


def test_process_video_list_of_frames(segmentor, rgb_image: np.ndarray) -> None:
    frames = [rgb_image.copy() for _ in range(4)]
    all_masks, all_scores = segmentor.process_video(
        frames,
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert all_masks.shape == (4, _IMAGE_H, _IMAGE_W)
    assert all_scores.shape == (4,)
    assert all_masks.dtype == bool


def test_process_video_empty_raises(segmentor) -> None:
    with pytest.raises(ValueError, match="empty"):
        segmentor.process_video(
            [],
            point_coords=np.array([[320, 240]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.float32),
        )


def test_process_video_single_frame(segmentor, rgb_image: np.ndarray) -> None:
    all_masks, all_scores = segmentor.process_video(
        [rgb_image],
        point_coords=np.array([[320, 240]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.float32),
    )
    assert all_masks.shape == (1, _IMAGE_H, _IMAGE_W)
    assert all_scores.shape == (1,)


# ---------------------------------------------------------------------------
# _encode_mask_as_logits
# ---------------------------------------------------------------------------


def test_encode_mask_as_logits_shape(segmentor) -> None:
    mask = np.zeros((_IMAGE_H, _IMAGE_W), dtype=bool)
    logits = segmentor._encode_mask_as_logits(mask)
    assert logits.shape == (1, 1, _MASK_INPUT_SIZE, _MASK_INPUT_SIZE)


def test_encode_mask_as_logits_true_maps_positive(segmentor) -> None:
    mask = np.ones((_IMAGE_H, _IMAGE_W), dtype=bool)
    logits = segmentor._encode_mask_as_logits(mask)
    assert np.all(logits > 0)


def test_encode_mask_as_logits_false_maps_negative(segmentor) -> None:
    mask = np.zeros((_IMAGE_H, _IMAGE_W), dtype=bool)
    logits = segmentor._encode_mask_as_logits(mask)
    assert np.all(logits < 0)
