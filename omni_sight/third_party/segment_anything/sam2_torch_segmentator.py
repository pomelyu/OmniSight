"""SAM2.1 PyTorch wrapper for image and video segmentation with full temporal memory."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import torch

from omni_sight.third_party.segment_anything.sam2_onnx_segmentator import _SAM2SegmentatorBase

# ---------------------------------------------------------------------------
# Vendor path: make "import sam2" resolve to reference/sam2 without pip install
# ---------------------------------------------------------------------------
_SAM2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "reference", "sam2")
)
if _SAM2_DIR not in sys.path:
    sys.path.insert(0, _SAM2_DIR)

# ---------------------------------------------------------------------------

_MODEL_CFG_MAP: Dict[str, str] = {
    "sam2.1_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class SAM2TorchSegmentator(_SAM2SegmentatorBase):
    """SAM2.1 segmentor backed by the official PyTorch implementation.

    Uses full temporal memory attention for video tracking — superior quality
    compared to :class:`~omni_sight.third_party.segment_anything.sam2_onnx_segmentator.SAM2OnnxSegmentator`.

    Shares :meth:`preprocess`, :meth:`postprocess`, and :meth:`run` with the
    ONNX backend via :class:`_SAM2SegmentatorBase`.  The main difference is
    :meth:`model_infer` (uses ``SAM2ImagePredictor``) and :meth:`process_video`
    (uses ``SAM2VideoPredictor`` with temporal memory).

    The SAM2 Python package is loaded from ``reference/sam2/`` in the
    OmniSight repo root — no separate ``pip install sam2`` is required.

    Supported ``model_name`` values:

    - ``"sam2.1_tiny"`` *(default)*
    - ``"sam2.1_small"``
    - ``"sam2.1_base_plus"``
    - ``"sam2.1_large"``

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from omni_sight.instance_segmentation import SAM2TorchSegmentator
        >>> seg = SAM2TorchSegmentator(
        ...     device="cpu",
        ...     model_name="sam2.1_tiny",
        ...     model_path="checkpoints/sam2.1_hiera_tiny.pt",
        ... )
        >>> image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        >>> masks, scores = seg.run(
        ...     image,
        ...     point_coords=np.array([[320, 240]], dtype=np.float32),
        ...     point_labels=np.array([1], dtype=np.float32),
        ... )
    """

    def __init__(
        self,
        device: str,
        model_name: Optional[str] = "sam2.1_tiny",
        model_cfg: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Load the SAM2 model and build image/video predictors.

        Args:
            device: Inference device (``"cpu"`` or ``"cuda"``).
            model_name: Model variant used to look up the Hydra config when
                ``model_cfg`` is not given. Defaults to ``"sam2.1_tiny"``.
            model_cfg: Explicit Hydra config name, e.g.
                ``"configs/sam2.1/sam2.1_hiera_t.yaml"``.  When set,
                ``model_name`` is only used for documentation.
            model_path: Path to the ``.pt`` checkpoint file.

        Raises:
            ValueError: If ``model_path`` is not supplied.
            ValueError: If ``model_name`` is not a recognised variant and
                ``model_cfg`` is not provided.
        """
        super().__init__(device=device, model_name=model_name, model_path=model_path)

        if model_path is None:
            raise ValueError("model_path must be provided for SAM2TorchSegmentator.")

        if model_cfg is None:
            if model_name not in _MODEL_CFG_MAP:
                raise ValueError(
                    f"Unknown model_name '{model_name}'. "
                    f"Provide model_cfg explicitly or use one of: "
                    f"{', '.join(_MODEL_CFG_MAP)}."
                )
            model_cfg = _MODEL_CFG_MAP[model_name]

        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # One model instance serves both image and video inference.
        self._video_predictor = build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=model_path,
            device=device,
        )
        self._image_predictor = SAM2ImagePredictor(self._video_predictor)

        self._autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if "cuda" in device
            else contextlib.nullcontext()
        )

    # ------------------------------------------------------------------
    # BasicProcessor interface
    # ------------------------------------------------------------------

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run the SAM2 image predictor on a preprocessed input dict.

        Uses ``preprocessed["img"]`` (the original RGB uint8 image) and passes
        it directly to ``SAM2ImagePredictor``, which handles its own resizing
        and normalisation.  Prompt keys accepted:

        - ``"point_coords"``: ``(N, 2)`` float32 ``(x, y)`` in pixel space.
        - ``"point_labels"``: ``(N,)`` float32 — ``1`` fg, ``0`` bg.
        - ``"box"``: ``(4,)`` float32 ``[x1, y1, x2, y2]`` in pixel space.

        Args:
            preprocessed: Output of :meth:`preprocess` with optional prompt keys.

        Returns:
            Dict with:

            - ``"masks"``: float32 ``(N, H, W)`` at original resolution.
            - ``"iou_scores"``: float32 ``(N,)``.

        Raises:
            ValueError: If neither ``"point_coords"`` nor ``"box"`` is set.
        """
        point_coords = preprocessed.get("point_coords")
        point_labels = preprocessed.get("point_labels")
        box = preprocessed.get("box")

        if point_coords is None and box is None:
            raise ValueError("At least one of point_coords or box must be provided.")

        with torch.inference_mode(), self._autocast_ctx:
            self._image_predictor.set_image(preprocessed["img"])
            masks, iou_scores, _ = self._image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        return {"masks": masks, "iou_scores": iou_scores}

    # ------------------------------------------------------------------
    # Video API
    # ------------------------------------------------------------------

    def process_video(
        self,
        frames: Union[List[np.ndarray], str],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment an object across a video using full temporal memory attention.

        Applies prompts on frame 0; the memory encoder propagates the mask to
        all subsequent frames without additional user input.

        Args:
            frames: List of RGB uint8 arrays ``(H, W, 3)`` or a path to a
                ``.mp4`` video file.
            point_coords: ``(N, 2)`` float32 prompt for frame 0, or ``None``.
            point_labels: ``(N,)`` float32 labels for ``point_coords``.
            box: ``(4,)`` float32 ``[x1, y1, x2, y2]`` prompt for frame 0.

        Returns:
            Tuple of:

            - ``all_masks``: bool ``(T, H, W)`` — best mask per frame.
            - ``all_scores``: float32 ``(T,)`` — placeholder scores (``1.0``
              each; the video predictor does not emit per-frame IoU).

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
            ValueError: If ``frames`` is an empty list.
        """
        if point_coords is None and box is None:
            raise ValueError("At least one of point_coords or box must be provided.")

        if isinstance(frames, (str, bytes)):
            with torch.inference_mode(), self._autocast_ctx:
                state = self._video_predictor.init_state(str(frames))
                return self._do_video_inference(state, point_coords, point_labels, box)

        frame_list = list(frames)
        if not frame_list:
            raise ValueError("frames is empty — cannot initialize.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, frame in enumerate(frame_list):
                cv2.imwrite(
                    os.path.join(tmp_dir, f"{i:05d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
            with torch.inference_mode(), self._autocast_ctx:
                state = self._video_predictor.init_state(tmp_dir)
                return self._do_video_inference(state, point_coords, point_labels, box)

    def _do_video_inference(
        self,
        state: Dict[str, Any],
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add prompts on frame 0 and propagate across the entire video.

        Must be called within a ``torch.inference_mode()`` context.

        Args:
            state: Inference state from ``SAM2VideoPredictor.init_state``.
            point_coords: ``(N, 2)`` float32 prompt or ``None``.
            point_labels: ``(N,)`` float32 labels or ``None``.
            box: ``(4,)`` float32 bounding-box prompt or ``None``.

        Returns:
            Tuple as described in :meth:`process_video`.
        """
        self._video_predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=point_coords, labels=point_labels, box=box,
        )

        num_frames: int = state["num_frames"]
        all_masks: List[Optional[np.ndarray]] = [None] * num_frames

        for frame_idx, _obj_ids, video_res_masks in self._video_predictor.propagate_in_video(state):
            # video_res_masks: (num_objects, 1, H, W) logits at original resolution
            all_masks[frame_idx] = video_res_masks[0, 0].cpu().numpy() > 0.0

        h: int = state["video_height"]
        w: int = state["video_width"]
        placeholder = np.zeros((h, w), dtype=bool)
        filled = [m if m is not None else placeholder for m in all_masks]
        return np.stack(filled, axis=0), np.ones(num_frames, dtype=np.float32)
