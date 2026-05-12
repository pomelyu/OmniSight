from __future__ import annotations

import os
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

from omni_sight.basic_processor import BasicProcessor
from omni_sight.utils.algo import nms
from omni_sight.onnx.onnx_loader import OnnxLoader

scrfd_model_loader = OnnxLoader()
scrfd_model_loader.set_file(
    file_name="scrfd_10g_bnkps-731cbbfd.onnx",
    urls=["https://media.githubusercontent.com/media/cysin/scrfd_onnx/refs/heads/main/scrfd_10g_bnkps.onnx"],
    identifier="scrfd_10g_bnkps",
)
scrfd_model_loader.set_file(
    file_name="scrfd_2.5g_bnkps-9c162a81.onnx",
    urls=["https://media.githubusercontent.com/media/cysin/scrfd_onnx/refs/heads/main/scrfd_2.5g_bnkps.onnx"],
    identifier="scrfd_2.5g_bnkps",
)

class SCRFDFaceDetector(BasicProcessor):
    """SCRFD ONNX face detector with OmniSight processor contract."""

    def __init__(
        self,
        device: str,
        model_name: Optional[str] = "scrfd_10g_bnkps",
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize detector and ONNX Runtime session.

        Args:
            device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
            model_name: Model identifier used as the file path when
                ``model_path`` is ``None``.
            model_path: Explicit path to the ONNX model file; takes
                precedence over ``model_name``.

        Raises:
            ValueError: If both ``model_name`` and ``model_path`` are ``None``.
            FileNotFoundError: If the resolved model file does not exist.
        """
        super().__init__(device=device, model_name=model_name, model_path=model_path)
        if model_path is None and model_name is None:
            raise ValueError("Either model_path or model_name must be provided.")

        self.center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.batched = False
        self.taskname = "detection"

        self.session = scrfd_model_loader.get_onnx_session(
            identifier=model_name,
            model_path=model_path,
            device=device,
        )
        self._init_vars()

    @staticmethod
    def _build_providers(device: str) -> List[str]:
        """Select ONNX Runtime providers from device string.

        Args:
            device: Device string (e.g. ``"cpu"`` or ``"cuda"``).

        Returns:
            List of ONNX Runtime provider strings ordered by preference.
        """
        normalized = (device or "cpu").lower()
        available = ort.get_available_providers()

        if normalized.startswith("cuda"):
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return [provider for provider in requested if provider in available]

        return ["CPUExecutionProvider"]

    def _init_vars(self) -> None:
        """Initialize model metadata and SCRFD feature map configuration."""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_size = self._resolve_input_size(input_shape)

        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.input_name = input_cfg.name
        self.output_names = [output.name for output in outputs]

        self.use_kps = False
        self._num_anchors = 1

        output_count = len(outputs)
        if output_count == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif output_count == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif output_count == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif output_count == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True
        else:
            raise ValueError(
                "Unsupported SCRFD output count: "
                f"{output_count}. Expected one of 6, 9, 10, 15."
            )

    def _resolve_input_size(self, input_shape: List[object]) -> Tuple[int, int]:
        """Resolve model input size from graph metadata or checkpoint name.

        Args:
            input_shape: Shape list from the ONNX graph input metadata
                (may contain symbolic strings for dynamic dimensions).

        Returns:
            ``(width, height)`` tuple for the model's fixed input size.

        Raises:
            ValueError: If the size cannot be determined from metadata or
                the model filename.
        """
        if not isinstance(input_shape[2], str) and not isinstance(input_shape[3], str):
            return int(input_shape[3]), int(input_shape[2])

        filename = os.path.basename(self.model_file)
        parsed_size = self._parse_input_size_from_filename(filename)
        if parsed_size is not None:
            return parsed_size

        raise ValueError(
            "Unable to resolve SCRFD input size from model metadata or filename. "
            "Expected static shape in ONNX graph or filename pattern like shape512x512."
        )

    @staticmethod
    def _parse_input_size_from_filename(filename: str) -> Optional[Tuple[int, int]]:
        """Parse input size from checkpoint filename.

        Args:
            filename: Checkpoint basename to parse (e.g.
                ``scrfd_10g_bnkps_shape512x512-237daff4.onnx``).

        Returns:
            ``(width, height)`` tuple if a ``WxH`` pattern is found,
            otherwise ``None``.
        """
        for pattern in (r"shape(\d+)x(\d+)", r"(\d+)x(\d+)"):
            match = re.search(pattern, filename, flags=re.IGNORECASE)
            if match is None:
                continue

            width = int(match.group(1))
            height = int(match.group(2))
            if width > 0 and height > 0:
                return width, height

        return None

    def preprocess(
        self,
        img: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Prepare image and metadata for SCRFD ONNX inference.

        Args:
            img: BGR input image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"blob"``: normalised NCHW float32 tensor ready for inference.
            - ``"det_scale"``: float32 array of shape ``(1,)`` with the
              resize scale factor.

        Raises:
            ValueError: If ``img`` does not have shape ``(H, W, 3)``.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must have shape (H, W, 3).")
        target_size = self.input_size

        im_ratio = float(img.shape[0]) / float(img.shape[1])
        model_ratio = float(target_size[1]) / float(target_size[0])

        if im_ratio > model_ratio:
            new_height = target_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = target_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / float(img.shape[0])
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128,
            size=tuple(det_img.shape[0:2][::-1]),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return {
            "blob": blob,
            "det_scale": np.array([det_scale], dtype=np.float32),
        }

    def model_infer(self, preprocessed: Dict[str, np.ndarray]) -> Dict[str, object]:
        """Run ONNX model inference with preprocessed input.

        Args:
            preprocessed: Output dict from :meth:`preprocess`.

        Returns:
            Dict with keys:

            - ``"blob"``: the input blob forwarded from ``preprocessed``.
            - ``"det_scale"``: float resize scale factor.
            - ``"net_outs"``: list of raw ONNX output arrays.
        """
        blob = preprocessed["blob"]
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        return {
            "blob": blob,
            "det_scale": float(preprocessed["det_scale"][0]),
            "net_outs": net_outs,
        }

    def postprocess(
        self,
        inference_outputs: Dict[str, object],
        thresh: float = 0.5,
        nms_thresh: float = 0.4,
        max_num: int = 0,
        metric: str = "center",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Decode model outputs to detections and optional keypoints.

        Args:
            inference_outputs: Output dict from :meth:`model_infer`.
            thresh: Confidence threshold for keeping a detection. Defaults
                to ``0.5``.
            nms_thresh: IoU threshold for NMS. Defaults to ``0.4``.
            max_num: Maximum number of detections to return. ``0`` means no
                limit. Defaults to ``0``.
            metric: Strategy for selecting top detections when
                ``max_num > 0``. ``"max"`` keeps the largest boxes;
                ``"center"`` prefers boxes closest to the image centre.
                Defaults to ``"center"``.

        Returns:
            Tuple of:

            - ``det``: float32 array of shape ``(K, 5)`` with columns
              ``[x1, y1, x2, y2, score]``.
            - ``kpss``: float32 array of shape ``(K, 5, 2)`` with keypoints,
              or ``None`` if the model does not predict keypoints.

        Raises:
            ValueError: If ``thresh`` is outside ``[0, 1]``, ``nms_thresh``
                is outside ``(0, 1]``, or ``metric`` is not ``"center"`` or
                ``"max"``.
        """
        if not 0.0 <= thresh <= 1.0:
            raise ValueError("thresh must be in [0, 1].")
        if not 0.0 < nms_thresh <= 1.0:
            raise ValueError("nms_thresh must be in (0, 1].")

        if metric not in ["center", "max"]:
            raise ValueError("metric should be either 'center' or 'max'")

        blob = inference_outputs["blob"]
        det_scale = float(inference_outputs["det_scale"])
        net_outs = inference_outputs["net_outs"]

        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1],
                    axis=-1,
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors,
                        axis=1,
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        if len(scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = nms(pre_det, nms_thresh)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = (input_height // 2, input_width // 2)
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
            values = area if metric == "max" else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :, :]

        return det, kpss

    def run(
        self,
        img: np.ndarray,
        thresh: float = 0.5,
        nms_thresh: float = 0.4,
        max_num: int = 0,
        metric: str = "center",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Execute preprocess, model_infer, and postprocess in sequence.

        Args:
            img: BGR input image of shape ``(H, W, 3)``.
            thresh: Confidence threshold. Defaults to ``0.5``.
            nms_thresh: IoU threshold for NMS. Defaults to ``0.4``.
            max_num: Maximum detections to return (``0`` = unlimited).
                Defaults to ``0``.
            metric: Detection-ranking strategy when ``max_num > 0``
                (``"center"`` or ``"max"``). Defaults to ``"center"``.

        Returns:
            Same as :meth:`postprocess`.
        """
        preprocessed = self.preprocess(img=img)
        raw_outputs = self.model_infer(preprocessed)
        return self.postprocess(
            inference_outputs=raw_outputs,
            thresh=thresh,
            nms_thresh=nms_thresh,
            max_num=max_num,
            metric=metric,
        )


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance predictions into bounding boxes.

    Args:
        points: Anchor centre coordinates of shape ``(N, 2)``.
        distance: Distance predictions of shape ``(N, 4)`` as
            ``[left, top, right, bottom]`` offsets from each anchor.
        max_shape: Optional ``(height, width)`` used to clip coordinates.

    Returns:
        Bounding boxes of shape ``(N, 4)`` as ``[x1, y1, x2, y2]``.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        max_h, max_w = max_shape
        x1 = np.clip(x1, 0, max_w)
        y1 = np.clip(y1, 0, max_h)
        x2 = np.clip(x2, 0, max_w)
        y2 = np.clip(y2, 0, max_h)

    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance predictions into keypoints.

    Args:
        points: Anchor centre coordinates of shape ``(N, 2)``.
        distance: Keypoint distance predictions of shape ``(N, 2*K)``
            where K is the number of keypoints, interleaved as
            ``[dx0, dy0, dx1, dy1, ...]``.
        max_shape: Optional ``(height, width)`` used to clip coordinates.

    Returns:
        Keypoint coordinates of shape ``(N, 2*K)`` interleaved as
        ``[x0, y0, x1, y1, ...]``.
    """
    preds: List[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]

        if max_shape is not None:
            max_h, max_w = max_shape
            px = np.clip(px, 0, max_w)
            py = np.clip(py, 0, max_h)

        preds.append(px)
        preds.append(py)

    return np.stack(preds, axis=-1)
