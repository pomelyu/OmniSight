"""SAM2.1 ONNX wrapper for image and video segmentation via mask propagation."""

from __future__ import annotations

from typing import Any
from typing import Dict
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

# ImageNet normalisation constants (RGB channel order)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

class SAM2OnnxSegmentator(BasicProcessor):
    """SAM2.1 video segmentor backed by ONNX encoder and decoder.

    Supports both stateless single-frame inference (:meth:`run`) and stateful
    video inference (:meth:`initialize` / :meth:`propagate` /
    :meth:`process_video`).

    Video mode uses *mask propagation*: the best mask from frame N is
    downsampled to 256×256, converted to logits, and fed as ``mask_input`` to
    the decoder on frame N+1.  This uses the existing ``mask_input`` input of
    the ONNX decoder without requiring the ``memory_encoder`` or
    ``memory_attention`` modules, which are absent from the ONNX export.

    Note:
        This is lighter-weight than the official SAM2 PyTorch video predictor,
        which uses full temporal memory attention.  A PyTorch-backed predictor
        will be added in a future version.

    Supported ``model_name`` values:

    - ``"sam2.1_hiera_tiny"``
    - ``"sam2.1_hiera_small"``
    - ``"sam2.1_hiera_base_plus"``
    - ``"sam2.1_hiera_large"``

    Preprocessing matches the samexporter convention: the image is resized to
    1024×1024 (non-uniform scale, i.e. aspect ratio is not preserved) and
    normalised with ImageNet mean/std before being passed to the encoder.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from omni_sight.instance_segmentation import SAM2OnnxSegmentator
        >>> seg = SAM2OnnxSegmentator(
        ...     device="cpu",
        ...     encoder_path="output_models/sam2.1_hiera_large.encoder.onnx",
        ...     decoder_path="output_models/sam2.1_hiera_large.decoder.onnx",
        ... )
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
            model_name: Optional SAM2.1 variant name — used for validation
                only; does not affect model loading when paths are given.
            encoder_path: Path to the ``.encoder.onnx`` file.
            decoder_path: Path to the ``.decoder.onnx`` file.

        Raises:
            ValueError: If ``encoder_path`` or ``decoder_path`` is not provided.
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
            identifier=enc_id,
            model_path=encoder_path,
            device=device,
        )
        self.decoder_session = _sam2_decoder_loader.get_onnx_session(
            identifier=dec_id,
            model_path=decoder_path,
            device=device,
        )

        self.enc_input_names: List[str] = [
            inp.name for inp in self.encoder_session.get_inputs()
        ]
        self.enc_output_names: List[str] = [
            out.name for out in self.encoder_session.get_outputs()
        ]
        self.dec_input_names: List[str] = [
            inp.name for inp in self.decoder_session.get_inputs()
        ]
        self.dec_output_names: List[str] = [
            out.name for out in self.decoder_session.get_outputs()
        ]

        # Video state: low-res logits (1, 1, 256, 256) carried across frames.
        self._prev_mask_logits: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # BasicProcessor interface
    # ------------------------------------------------------------------

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        """Resize and normalise an RGB uint8 image for the SAM2 encoder.

        The image is resized to 1024×1024 (non-uniform scale, matching the
        samexporter convention) and normalised with ImageNet mean/std.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"blob"``: float32 NCHW array of shape ``(1, 3, 1024, 1024)``.
            - ``"scale_x"``: horizontal scale factor ``1024 / original_W``.
            - ``"scale_y"``: vertical scale factor ``1024 / original_H``.
            - ``"original_size"``: ``(H, W)`` of the input image.

        Raises:
            ValueError: If ``img`` is not ``(H, W, 3)`` uint8.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must have shape (H, W, 3).")
        if img.dtype != np.uint8:
            raise ValueError("img must be dtype uint8.")

        h, w = img.shape[:2]
        resized = cv2.resize(
            img,
            (_ENCODER_INPUT_SIZE, _ENCODER_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = (resized.astype(np.float32) / 255.0 - _IMG_MEAN) / _IMG_STD
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW

        return {
            "blob": blob,
            "scale_x": _ENCODER_INPUT_SIZE / w,
            "scale_y": _ENCODER_INPUT_SIZE / h,
            "original_size": (h, w),
        }

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM2 encoder then decoder on a preprocessed frame.

        Uses ``_prev_mask_logits`` from internal state (if set) to populate
        ``mask_input`` and ``has_mask_input`` for the decoder.

        The dict ``preprocessed`` is the output of :meth:`preprocess` extended
        with optional prompt keys:

        - ``"point_coords"``: ``(N, 2)`` float32 ``(x, y)`` in original-image
          pixel space, or ``None``.
        - ``"point_labels"``: ``(N,)`` float32 — ``1`` foreground,
          ``0`` background. Required when ``point_coords`` is not ``None``.
        - ``"box"``: ``(4,)`` float32 ``[x1, y1, x2, y2]`` in original-image
          pixel space, or ``None``.

        If neither ``"point_coords"`` nor ``"box"`` is set **and** no previous
        mask state exists (i.e. :meth:`initialize` has not been called), a
        :exc:`ValueError` is raised.  During propagation the decoder is driven
        entirely by ``mask_input``.

        Args:
            preprocessed: Dict produced by :meth:`preprocess` with optional
                prompt keys added by the caller.

        Returns:
            Dict with keys ``"masks_logits"`` ``(1, N, 1024, 1024)``,
            ``"iou_predictions"`` ``(1, N)``, ``"original_size"``,
            ``"scale_x"``, ``"scale_y"``.

        Raises:
            ValueError: If no geometric prompt is given and no mask state exists.
        """
        enc_out = self.encoder_session.run(
            None, {self.enc_input_names[0]: preprocessed["blob"]}
        )
        enc_map = dict(zip(self.enc_output_names, enc_out))

        scale_x: float = preprocessed["scale_x"]
        scale_y: float = preprocessed["scale_y"]
        h, w = preprocessed["original_size"]

        point_coords = preprocessed.get("point_coords")
        point_labels = preprocessed.get("point_labels")
        box = preprocessed.get("box")

        has_geometric_prompt = point_coords is not None or box is not None
        if not has_geometric_prompt and self._prev_mask_logits is None:
            raise ValueError("At least one of point_coords or box must be provided.")

        coords_list: list = []
        labels_list: list = []

        if box is not None:
            box_arr = np.asarray(box, dtype=np.float32)
            coords_list.append([float(box_arr[0]) * scale_x, float(box_arr[1]) * scale_y])
            coords_list.append([float(box_arr[2]) * scale_x, float(box_arr[3]) * scale_y])
            labels_list.extend([2.0, 3.0])

        if point_coords is not None:
            pts = np.asarray(point_coords, dtype=np.float32)
            for pt in pts:
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
        # Only pass inputs the loaded model actually declares
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

        return {
            "masks_logits": dec_map[masks_key],      # (1, N, 256, 256)
            "iou_predictions": dec_map[iou_key],     # (1, N)
            "original_size": (h, w),
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

    def postprocess(
        self, inference_outputs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize mask logits to original resolution and sort by IoU score.

        Args:
            inference_outputs: Output dict from :meth:`model_infer`.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` — candidates sorted
              from highest to lowest IoU score.
            - ``iou_scores``: float32 array of shape ``(N,)``.
        """
        masks_logits: np.ndarray = inference_outputs["masks_logits"][0]  # (N, 1024, 1024)
        iou_scores: np.ndarray = inference_outputs["iou_predictions"][0]  # (N,)
        h, w = inference_outputs["original_size"]

        recovered = np.stack(
            [
                cv2.resize(masks_logits[i], (w, h), interpolation=cv2.INTER_LINEAR)
                for i in range(masks_logits.shape[0])
            ],
            axis=0,
        )
        masks_bool = recovered > 0.0
        order = np.argsort(iou_scores)[::-1]
        return masks_bool[order], iou_scores[order].astype(np.float32)

    def run(
        self,
        img: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stateless single-frame segmentation.

        Does not read or update the internal video propagation state.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 ``(x, y)`` click coordinates in
                image pixel space, or ``None``.
            point_labels: ``(N,)`` float32 labels — ``1`` foreground,
                ``0`` background. Required when ``point_coords`` is given.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` bounding-box prompt
                in image pixel space, or ``None``.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` — candidates sorted
              from highest to lowest IoU score.
            - ``iou_scores``: float32 array of shape ``(N,)`` in ``[0, 1]``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        # Temporarily clear state so model_infer treats this as a fresh frame
        saved = self._prev_mask_logits
        self._prev_mask_logits = None
        try:
            preprocessed = self.preprocess(img)
            preprocessed["point_coords"] = point_coords
            preprocessed["point_labels"] = point_labels
            preprocessed["box"] = box
            return self.postprocess(self.model_infer(preprocessed))
        finally:
            self._prev_mask_logits = saved

    # ------------------------------------------------------------------
    # Video API
    # ------------------------------------------------------------------

    def _encode_mask_as_logits(self, mask: np.ndarray) -> np.ndarray:
        """Downsample a binary mask to (1, 1, 256, 256) float32 logits.

        Args:
            mask: bool array of shape ``(H, W)`` in original image coordinates.

        Returns:
            float32 array of shape ``(1, 1, 256, 256)``.
        """
        small = cv2.resize(
            mask.astype(np.uint8),
            (_MASK_INPUT_SIZE, _MASK_INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        # True → +10.0, False → -10.0 (confident logits for the decoder)
        logits = small * 20.0 - 10.0
        return logits[np.newaxis, np.newaxis]  # (1, 1, 256, 256)

    def _extract_best_mask_logits(self, raw_output: Dict[str, Any]) -> np.ndarray:
        """Select the (1, 1, 256, 256) logits for the model output.

        Prefers the decoder's own ``masks`` output (continuous logits)
        over binarised re-encoding, since that is the format SAM2 was trained
        to receive as ``mask_input``.

        Args:
            raw_output: Dict returned by :meth:`model_infer`.

        Returns:
            float32 array of shape ``(1, 1, 256, 256)`` suitable for
            ``mask_input`` on the next decoder call.
        """
        iou_preds = raw_output["iou_predictions"][0]  # (N,)
        best_idx = int(np.argmax(iou_preds))
        return raw_output["masks_logits"][:, best_idx : best_idx + 1]  # (1, 1, 256, 256)

    def initialize(
        self,
        frame: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the first frame and seed the mask propagation state.

        Must be called before :meth:`propagate`.  Any existing propagation
        state is discarded.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 ``(x, y)`` click coordinates.
            point_labels: ``(N,)`` float32 labels — ``1`` foreground,
                ``0`` background. Required when ``point_coords`` is given.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` bounding-box prompt.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` for the first frame.
            - ``iou_scores``: float32 array of shape ``(N,)``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
        """
        self._prev_mask_logits = None  # reset any prior state
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

        Uses the best mask from the previous :meth:`initialize` or
        :meth:`propagate` call as a spatial prior via ``mask_input``.
        No geometric prompts are required.

        Args:
            frame: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` for this frame.
            - ``iou_scores``: float32 array of shape ``(N,)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self._prev_mask_logits is None:
            raise RuntimeError(
                "Call initialize() with prompts on the first frame before propagate()."
            )
        preprocessed = self.preprocess(frame)
        # No geometric prompts; mask_input drives the decoder,
        # assume (0, 0) is background
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
        """Segment an entire frame sequence, tracking the object across frames.

        Calls :meth:`initialize` on the first frame (with prompts) then
        :meth:`propagate` on every subsequent frame.

        Args:
            frames: Either a ``list`` of RGB uint8 arrays ``(H, W, 3)`` or a
                path to a video file that :class:`cv2.VideoCapture` can open.
            point_coords: ``(N, 2)`` float32 ``(x, y)`` prompt for frame 0.
            point_labels: ``(N,)`` float32 labels for ``point_coords``.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` prompt for frame 0.

        Returns:
            Tuple of:

            - ``all_masks``: bool array of shape ``(T, H, W)`` — best mask
              per frame, in input order.
            - ``all_scores``: float32 array of shape ``(T,)`` — best IoU
              score per frame.

        Raises:
            ValueError: If ``frames`` is a path and the video cannot be opened.
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
            ValueError: If ``frames`` is an empty list or the video has no frames.
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

    @staticmethod
    def _iter_frames(
        frames: Union[List[np.ndarray], str],
    ):
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
