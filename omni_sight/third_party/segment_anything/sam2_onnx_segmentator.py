"""SAM2.1 base class and ONNX wrapper for image and video segmentation."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np

from omni_sight.basic_processor import BasicProcessor
from omni_sight.onnx.onnx_loader import OnnxLoader

_ENCODER_INPUT_SIZE: int = 1024
_MASK_INPUT_SIZE: int = 256
_IMG_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_VALID_MODEL_NAMES: List[str] = [
    "sam2.1_tiny",
    "sam2.1_small",
    "sam2.1_base_plus",
    "sam2.1_large",
]

_sam2_encoder_loader = OnnxLoader()
_sam2_encoder_loader.set_file(
    file_name="sam2.1_hiera_tiny.encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_tiny.encoder.onnx?download=true"],
    identifier="sam2.1_tiny_encoder",
)
_sam2_encoder_loader.set_file(
    file_name="sam2.1_hiera_small.encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_small.encoder.onnx?download=true"],
    identifier="sam2.1_small_encoder",
)
_sam2_encoder_loader.set_file(
    file_name="sam2.1_hiera_base_plus.encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_base_plus.encoder.onnx?download=true"],
    identifier="sam2.1_base_plus_encoder",
)
_sam2_encoder_loader.set_file(
    file_name="sam2.1_hiera_large.encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_large.encoder.onnx?download=true"],
    identifier="sam2.1_large_encoder",
)

_sam2_decoder_loader = OnnxLoader()
_sam2_decoder_loader.set_file(
    file_name="sam2.1_hiera_tiny.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_tiny.decoder.onnx?download=true"],
    identifier="sam2.1_tiny_decoder",
)
_sam2_decoder_loader.set_file(
    file_name="sam2.1_hiera_small.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_small.decoder.onnx?download=true"],
    identifier="sam2.1_small_decoder",
)
_sam2_decoder_loader.set_file(
    file_name="sam2.1_hiera_base_plus.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_base_plus.decoder.onnx?download=true"],
    identifier="sam2.1_base_plus_decoder",
)
_sam2_decoder_loader.set_file(
    file_name="sam2.1_hiera_large.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam2.1/sam2.1_hiera_large.decoder.onnx?download=true"],
    identifier="sam2.1_large_decoder",
)


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _SAM2SegmentatorBase(BasicProcessor):
    """Shared image preprocessing, postprocessing, and frame utilities for SAM2."""

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        """Resize and normalise an RGB uint8 image for SAM2 inference.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"img"``: original image (passthrough, used by the PyTorch backend).
            - ``"blob"``: float32 NCHW array ``(1, 3, 1024, 1024)`` (used by ONNX backend).
            - ``"scale_x"``: ``1024 / original_W``.
            - ``"scale_y"``: ``1024 / original_H``.
            - ``"original_size"``: ``(H, W)`` of the input image.

        Raises:
            ValueError: If ``img`` is not ``(H, W, 3)`` uint8.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must have shape (H, W, 3).")
        if img.dtype != np.uint8:
            raise ValueError("img must be dtype uint8.")
        h, w = img.shape[:2]
        resized = cv2.resize(img, (_ENCODER_INPUT_SIZE, _ENCODER_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        blob = (resized.astype(np.float32) / 255.0 - _IMG_MEAN) / _IMG_STD
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
        return {
            "img": img,
            "blob": blob,
            "scale_x": _ENCODER_INPUT_SIZE / w,
            "scale_y": _ENCODER_INPUT_SIZE / h,
            "original_size": (h, w),
        }

    def postprocess(self, outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Sort mask candidates by IoU score and binarise.

        Expects ``model_infer`` to have already resized masks to the original
        image resolution.

        Args:
            outputs: Dict with:

            - ``"masks"``: float32 array of shape ``(N, H, W)``.
            - ``"iou_scores"``: float32 array of shape ``(N,)``.

        Returns:
            Tuple of bool masks ``(N, H, W)`` and float32 IoU scores ``(N,)``,
            both sorted from highest to lowest score.
        """
        masks: np.ndarray = outputs["masks"]
        iou: np.ndarray = outputs["iou_scores"]
        order = np.argsort(iou)[::-1]
        return (masks > 0.0)[order], iou[order].astype(np.float32)

    def run(
        self,
        img: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-image segmentation with geometric prompts.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 ``(x, y)`` click coordinates, or ``None``.
            point_labels: ``(N,)`` float32 — ``1`` foreground, ``0`` background.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` bounding-box prompt, or ``None``.

        Returns:
            Tuple of bool masks ``(N, H, W)`` and float32 IoU scores ``(N,)``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        if point_coords is None and box is None:
            raise ValueError("At least one of point_coords or box must be provided.")
        preprocessed = self.preprocess(img)
        preprocessed["point_coords"] = point_coords
        preprocessed["point_labels"] = point_labels
        preprocessed["box"] = box
        return self.postprocess(self.model_infer(preprocessed))

    @staticmethod
    def _iter_frames(frames: Union[List[np.ndarray], str]) -> Iterator[np.ndarray]:
        """Yield RGB uint8 frames from a list or a video file path.

        Args:
            frames: List of RGB uint8 arrays or a path to a video file.

        Yields:
            RGB uint8 arrays of shape ``(H, W, 3)``.

        Raises:
            ValueError: If ``frames`` is a string path that cannot be opened.
        """
        if isinstance(frames, (str, bytes)):
            cap = cv2.VideoCapture(str(frames))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {frames}")
            try:
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
        else:
            yield from frames


# ---------------------------------------------------------------------------
# ONNX implementation
# ---------------------------------------------------------------------------

class SAM2OnnxSegmentator(_SAM2SegmentatorBase):
    """SAM2.1 video segmentor backed by ONNX encoder and decoder.

    Supports both stateless single-frame inference (:meth:`run`) and stateful
    video inference (:meth:`initialize` / :meth:`propagate` /
    :meth:`process_video`).

    Video mode uses *mask propagation*: the best mask from frame N is used as
    ``mask_input`` to the decoder on frame N+1.  This uses the existing
    ``mask_input`` input of the ONNX decoder without requiring the
    ``memory_encoder`` or ``memory_attention`` modules (absent from the ONNX
    export).

    Note:
        This is lighter-weight than the official SAM2 PyTorch video predictor,
        which uses full temporal memory attention.  See
        :class:`~omni_sight.third_party.segment_anything.sam2_torch_segmentator.SAM2TorchSegmentator`
        for the PyTorch alternative.

    Supported ``model_name`` values:

    - ``"sam2.1_tiny"``
    - ``"sam2.1_small"``
    - ``"sam2.1_base_plus"``
    - ``"sam2.1_large"``

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from omni_sight.instance_segmentation import SAM2OnnxSegmentator
        >>> seg = SAM2OnnxSegmentator(device="cpu", model_name="sam2.1_tiny")
        >>> frame0 = cv2.cvtColor(cv2.imread("frame0.jpg"), cv2.COLOR_BGR2RGB)
        >>> masks, scores = seg.initialize(
        ...     frame0,
        ...     point_coords=np.array([[320, 240]], dtype=np.float32),
        ...     point_labels=np.array([1], dtype=np.float32),
        ... )
        >>> frame1 = cv2.cvtColor(cv2.imread("frame1.jpg"), cv2.COLOR_BGR2RGB)
        >>> masks, scores = seg.propagate(frame1)
    """

    def __init__(
        self,
        device: str,
        model_name: Optional[str] = None,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
    ) -> None:
        """Load ONNX encoder and decoder sessions.

        Args:
            device: Inference device (``"cpu"`` or ``"cuda"``).
            model_name: SAM2.1 variant name — used for auto-download when
                ``encoder_path`` / ``decoder_path`` are not given.
            encoder_path: Path to the ``.encoder.onnx`` file.
            decoder_path: Path to the ``.decoder.onnx`` file.

        Raises:
            ValueError: If ``model_name`` is not provided when paths are absent.
            ValueError: If ``model_name`` is not a recognised SAM2.1 variant.
        """
        super().__init__(device=device)
        if model_name is None and (encoder_path is None or decoder_path is None):
            raise ValueError(
                "model_name is required when encoder_path or decoder_path is not provided."
            )
        if model_name is not None and model_name not in _VALID_MODEL_NAMES:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Choose from: {', '.join(_VALID_MODEL_NAMES)}."
            )

        enc_id = f"{model_name}_encoder" if model_name else ""
        dec_id = f"{model_name}_decoder" if model_name else ""

        self.encoder_session = _sam2_encoder_loader.get_onnx_session(
            identifier=enc_id, model_path=encoder_path, device=device,
        )
        self.decoder_session = _sam2_decoder_loader.get_onnx_session(
            identifier=dec_id, model_path=decoder_path, device=device,
        )

        self.enc_input_names: List[str] = [inp.name for inp in self.encoder_session.get_inputs()]
        self.enc_output_names: List[str] = [out.name for out in self.encoder_session.get_outputs()]
        self.dec_input_names: List[str] = [inp.name for inp in self.decoder_session.get_inputs()]
        self.dec_output_names: List[str] = [out.name for out in self.decoder_session.get_outputs()]

        self._prev_mask_logits: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # BasicProcessor interface
    # ------------------------------------------------------------------

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM2 ONNX encoder then decoder on a preprocessed frame.

        Uses ``_prev_mask_logits`` from internal state (if set) to populate
        ``mask_input`` for the decoder.  Prompt keys accepted in ``preprocessed``:

        - ``"point_coords"``: ``(N, 2)`` float32 ``(x, y)`` in original-image pixels.
        - ``"point_labels"``: ``(N,)`` float32 — ``1`` fg, ``0`` bg.
        - ``"box"``: ``(4,)`` float32 ``[x1, y1, x2, y2]`` in original-image pixels.

        Args:
            preprocessed: Output of :meth:`preprocess` with optional prompt keys.

        Returns:
            Dict with:

            - ``"masks"``: float32 ``(N, H, W)`` at original resolution.
            - ``"iou_scores"``: float32 ``(N,)``.
            - ``"_dec_masks"``: raw decoder logits ``(1, N, H_dec, W_dec)`` used
              internally by :meth:`_extract_best_mask_logits` for video state.

        Raises:
            ValueError: If no geometric prompt is given and no mask state exists.
        """
        enc_out = self.encoder_session.run(None, {self.enc_input_names[0]: preprocessed["blob"]})
        enc_map = dict(zip(self.enc_output_names, enc_out))

        scale_x: float = preprocessed["scale_x"]
        scale_y: float = preprocessed["scale_y"]
        h, w = preprocessed["original_size"]

        point_coords = preprocessed.get("point_coords")
        point_labels = preprocessed.get("point_labels")
        box = preprocessed.get("box")

        if point_coords is None and box is None and self._prev_mask_logits is None:
            raise ValueError("At least one of point_coords or box must be provided.")

        coords_list: list = []
        labels_list: list = []

        if box is not None:
            box_arr = np.asarray(box, dtype=np.float32)
            coords_list += [[float(box_arr[0]) * scale_x, float(box_arr[1]) * scale_y],
                            [float(box_arr[2]) * scale_x, float(box_arr[3]) * scale_y]]
            labels_list += [2.0, 3.0]

        if point_coords is not None:
            for pt in np.asarray(point_coords, dtype=np.float32):
                coords_list.append([float(pt[0]) * scale_x, float(pt[1]) * scale_y])
            labels_list.extend(np.asarray(point_labels, dtype=np.float32).tolist())

        # Padding point required by the ONNX decoder contract
        coords_list.append([0.0, 0.0])
        labels_list.append(-1.0)

        coords_arr = np.array(coords_list, dtype=np.float32)[np.newaxis]  # (1, N, 2)
        labels_arr = np.array(labels_list, dtype=np.float32)[np.newaxis]  # (1, N)

        if self._prev_mask_logits is not None:
            mask_input = self._prev_mask_logits
            has_mask_input = np.ones((1,), dtype=np.float32)
        else:
            mask_input = np.zeros((1, 1, _MASK_INPUT_SIZE, _MASK_INPUT_SIZE), dtype=np.float32)
            has_mask_input = np.zeros((1,), dtype=np.float32)

        dec_inputs: Dict[str, np.ndarray] = {
            "image_embed": enc_map.get("image_embed", enc_out[2]),
            "high_res_feats_0": enc_map.get("high_res_feats_0", enc_out[0]),
            "high_res_feats_1": enc_map.get("high_res_feats_1", enc_out[1]),
            "point_coords": coords_arr,
            "point_labels": labels_arr,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
        }
        filtered = {k: v for k, v in dec_inputs.items() if k in self.dec_input_names}

        dec_out = self.decoder_session.run(None, filtered)
        dec_map = dict(zip(self.dec_output_names, dec_out))

        masks_key = next(
            (n for n in self.dec_output_names if "mask" in n.lower() and "low" not in n.lower()),
            self.dec_output_names[0],
        )
        iou_key = next(
            (n for n in self.dec_output_names if "iou" in n.lower()),
            self.dec_output_names[1] if len(self.dec_output_names) > 1 else self.dec_output_names[0],
        )

        dec_masks = dec_map[masks_key]   # (1, N, H_dec, W_dec)
        iou_scores = dec_map[iou_key][0]  # (N,)

        n = dec_masks.shape[1]
        masks_at_orig = np.stack(
            [cv2.resize(dec_masks[0, i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
             for i in range(n)],
            axis=0,
        )  # (N, H, W)

        return {
            "masks": masks_at_orig,
            "iou_scores": iou_scores,
            "_dec_masks": dec_masks,   # internal — kept for video propagation state
        }

    def run(
        self,
        img: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stateless single-frame segmentation.

        Temporarily clears internal video state so the result is independent of
        any ongoing :meth:`initialize` / :meth:`propagate` session.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 click coordinates, or ``None``.
            point_labels: ``(N,)`` float32 labels, or ``None``.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` prompt, or ``None``.

        Returns:
            Tuple of bool masks ``(N, H, W)`` and float32 IoU scores ``(N,)``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        saved, self._prev_mask_logits = self._prev_mask_logits, None
        try:
            return super().run(img, point_coords, point_labels, box)
        finally:
            self._prev_mask_logits = saved

    # ------------------------------------------------------------------
    # Video API
    # ------------------------------------------------------------------

    def _extract_best_mask_logits(self, raw_output: Dict[str, Any]) -> np.ndarray:
        """Return the raw decoder logits for the highest-IoU mask candidate.

        Args:
            raw_output: Dict returned by :meth:`model_infer`.

        Returns:
            float32 array of shape ``(1, 1, H_dec, W_dec)`` suitable for
            ``mask_input`` on the next decoder call.
        """
        best_idx = int(np.argmax(raw_output["iou_scores"]))
        return raw_output["_dec_masks"][:, best_idx : best_idx + 1]

    def initialize(
        self,
        frame: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the first frame and seed the mask propagation state.

        Must be called before :meth:`propagate`.  Discards any prior state.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 click coordinates, or ``None``.
            point_labels: ``(N,)`` float32 labels, or ``None``.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` prompt, or ``None``.

        Returns:
            Tuple of bool masks ``(N, H, W)`` and float32 IoU scores ``(N,)``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        self._prev_mask_logits = None
        preprocessed = self.preprocess(frame)
        preprocessed["point_coords"] = point_coords
        preprocessed["point_labels"] = point_labels
        preprocessed["box"] = box
        raw_output = self.model_infer(preprocessed)
        masks, scores = self.postprocess(raw_output)
        self._prev_mask_logits = self._extract_best_mask_logits(raw_output)
        return masks, scores

    def propagate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate the tracked mask to the next frame.

        Uses the best mask from the previous call as a spatial prior via
        ``mask_input``.  No geometric prompts are required.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Tuple of bool masks ``(N, H, W)`` and float32 IoU scores ``(N,)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self._prev_mask_logits is None:
            raise RuntimeError("Call initialize() with prompts on the first frame before propagate().")
        preprocessed = self.preprocess(frame)
        preprocessed["point_coords"] = np.array([[0, 0]])
        preprocessed["point_labels"] = np.array([0])
        preprocessed["box"] = None
        raw_output = self.model_infer(preprocessed)
        masks, scores = self.postprocess(raw_output)
        self._prev_mask_logits = self._extract_best_mask_logits(raw_output)
        return masks, scores

    def process_video(
        self,
        frames: Union[List[np.ndarray], str],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment an entire frame sequence via mask propagation.

        Calls :meth:`initialize` on the first frame then :meth:`propagate` on
        every subsequent frame.

        Args:
            frames: List of RGB uint8 arrays or a path to a video file.
            point_coords: ``(N, 2)`` float32 prompt for frame 0, or ``None``.
            point_labels: ``(N,)`` float32 labels for ``point_coords``.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` prompt for frame 0.

        Returns:
            Tuple of:

            - ``all_masks``: bool ``(T, H, W)`` — best mask per frame.
            - ``all_scores``: float32 ``(T,)`` — best IoU score per frame.

        Raises:
            ValueError: If ``frames`` is empty or the video cannot be opened.
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        frame_iter = self._iter_frames(frames)
        try:
            first_frame = next(frame_iter)
        except StopIteration:
            raise ValueError("frames is empty — cannot initialize.")

        first_masks, first_score = self.initialize(
            first_frame, point_coords=point_coords, point_labels=point_labels, box=box
        )
        all_masks = [first_masks[0]]
        all_scores = [float(first_score[0])]

        for frame in frame_iter:
            masks, scores = self.propagate(frame)
            all_masks.append(masks[0])
            all_scores.append(float(scores[0]))

        return np.stack(all_masks, axis=0), np.array(all_scores, dtype=np.float32)
