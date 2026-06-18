"""Depth Anything V2 ONNX wrapper following the OmniSight BasicProcessor contract."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from omni_sight.basic_processor import BasicProcessor
from omni_sight.onnx.onnx_loader import OnnxLoader

_INPUT_SIZE: int = 518  # must be divisible by 14 (ViT patch size); 518 = 37 * 14
_IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_dav2_loader = OnnxLoader()
_dav2_loader.set_file(
    file_name="depth_anything_v2_small.onnx",
    urls=["https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"],
    identifier="depth_anything_v2_small",
)
_dav2_loader.set_file(
    file_name="depth_anything_v2_base.onnx",
    urls=["https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx"],
    identifier="depth_anything_v2_base",
)
_dav2_loader.set_file(
    file_name="depth_anything_v2_large.onnx",
    urls=["https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx"],
    identifier="depth_anything_v2_large",
)


class DepthAnythingV2Estimator(BasicProcessor):
    """Monocular depth estimator backed by a Depth Anything V2 ONNX model.

    Accepts **RGB uint8** images and returns a float32 relative-depth map with
    the same spatial dimensions as the input. Larger depth values indicate
    points farther from the camera.

    Three model variants are available:

    - ``"depth_anything_v2_small"`` — 99 MB, Apache-2.0 *(default)*
    - ``"depth_anything_v2_base"`` — 389 MB, CC-BY-NC-4.0
    - ``"depth_anything_v2_large"`` — ~1.3 GB, CC-BY-NC-4.0

    ONNX model files are downloaded automatically to ``~/.cache/omnisight/``
    on first use.

    Example:
        >>> import cv2
        >>> from omni_sight.depth_estimation import DepthAnythingV2Estimator
        >>> image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        >>> estimator = DepthAnythingV2Estimator(device="cpu")
        >>> depth = estimator.run(image)   # shape (H, W), float32
    """

    def __init__(
        self,
        device: str,
        model_name: Optional[str] = "depth_anything_v2_small",
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the estimator and ONNX Runtime session.

        Args:
            device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
            model_name: Model variant identifier. Ignored when ``model_path``
                is provided. Defaults to ``"depth_anything_v2_small"``.
            model_path: Explicit path to a local ``.onnx`` file. When given,
                ``model_name`` is ignored for session creation.

        Raises:
            ValueError: If both ``model_name`` and ``model_path`` are ``None``.
        """
        super().__init__(device=device, model_name=model_name, model_path=model_path)
        if model_name is None and model_path is None:
            raise ValueError("Either model_name or model_path must be provided.")

        self.session = _dav2_loader.get_onnx_session(
            identifier=model_name,
            model_path=model_path,
            device=device,
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        """Resize and normalise an RGB uint8 image for model input.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"blob"``: float32 NCHW tensor of shape ``(1, 3, 518, 518)``.
            - ``"original_size"``: ``(H, W)`` tuple of the input image.

        Raises:
            ValueError: If ``img`` does not have shape ``(H, W, 3)`` or
                dtype ``uint8``.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must have shape (H, W, 3).")
        if img.dtype != np.uint8:
            raise ValueError("img must be dtype uint8.")

        original_size: Tuple[int, int] = (img.shape[0], img.shape[1])
        resized = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = normalized.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
        return {"blob": blob, "original_size": original_size}

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run ONNX inference on preprocessed input.

        Args:
            preprocessed: Output dict from :meth:`preprocess`.

        Returns:
            Dict with keys:

            - ``"net_out"``: raw depth output array from the model.
            - ``"original_size"``: forwarded from ``preprocessed``.
        """
        net_out = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed["blob"]},
        )[0]
        return {"net_out": net_out, "original_size": preprocessed["original_size"]}

    def postprocess(self, inference_outputs: Dict[str, Any]) -> np.ndarray:
        """Decode raw model output to a depth map at original resolution.

        Args:
            inference_outputs: Output dict from :meth:`model_infer`.

        Returns:
            float32 array of shape ``(H, W)`` with relative depth values.
            Larger values indicate points farther from the camera.
        """
        depth = inference_outputs["net_out"]
        # Remove batch dim; handle both (1, H, W) and (1, 1, H, W) output shapes.
        depth = depth[0]
        if depth.ndim == 3:
            depth = depth[0]

        h, w = inference_outputs["original_size"]
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_resized.astype(np.float32)

    def run(self, img: np.ndarray) -> np.ndarray:
        """Run the full depth estimation pipeline on an RGB uint8 image.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            float32 depth map of shape ``(H, W)``. Values are relative depth —
            larger means farther from the camera.
        """
        return self.postprocess(self.model_infer(self.preprocess(img)))
