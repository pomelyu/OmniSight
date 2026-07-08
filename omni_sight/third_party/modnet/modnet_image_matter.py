"""MODNet ONNX wrapper following the OmniSight BasicProcessor contract.

Reference: https://github.com/ZHKKKe/MODNet (onnx/inference_onnx.py)
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from omni_sight.basic_processor import BasicProcessor
from omni_sight.onnx.onnx_loader import OnnxLoader

_REF_SIZE: int = 512  # reference input size used by the official MODNet ONNX demo
_STRIDE: int = 32  # network downsamples by 32, so input dims must be multiples of it

_modnet_loader = OnnxLoader()
_modnet_loader.set_file(
    file_name="modnet_photographic_portrait_matting-07c308cf.onnx",
    urls=["https://drive.google.com/uc?id=1cgycTQlYXpTh26gB9FTnthE7AvruV8hd"],
    identifier="modnet",
)


class MODNetImageMatter(BasicProcessor):
    """Portrait alpha-matting processor backed by the MODNet ONNX model.

    Accepts **RGB uint8** images and returns a uint8 alpha matte with the
    same spatial dimensions as the input — ``255`` is foreground (person),
    ``0`` is background. The model is trained for photographic portraits
    (``modnet_photographic_portrait_matting``, Apache-2.0).

    The ONNX model file is downloaded automatically to ``~/.cache/omnisight/``
    on first use.

    Example:
        >>> import cv2
        >>> from omni_sight.image_matting import MODNetImageMatter
        >>> image = cv2.cvtColor(cv2.imread("portrait.jpg"), cv2.COLOR_BGR2RGB)
        >>> matter = MODNetImageMatter(device="cpu")
        >>> matte = matter.run(image)   # shape (H, W), uint8
    """

    MODEL_NAME: str = "modnet"

    def __init__(
        self,
        device: str,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the matting processor and ONNX Runtime session.

        Args:
            device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
            model_path: Explicit path to a local ``.onnx`` file. When given,
                the auto-downloaded default model is not used.
        """
        super().__init__(device=device, model_name=self.MODEL_NAME, model_path=model_path)

        self.session = _modnet_loader.get_onnx_session(
            identifier=self.MODEL_NAME,
            model_path=model_path,
            device=device,
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """Resize and normalise an RGB uint8 image for model input.

        Args:
            image: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"blob"``: float32 NCHW tensor with values in ``[-1, 1]`` and
              spatial dims resolved by :meth:`_resolve_input_size`.
            - ``"original_size"``: ``(H, W)`` tuple of the input image.

        Raises:
            ValueError: If ``image`` does not have shape ``(H, W, 3)`` or
                dtype ``uint8``.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3).")
        if image.dtype != np.uint8:
            raise ValueError("image must be dtype uint8.")

        original_size: Tuple[int, int] = (image.shape[0], image.shape[1])
        input_h, input_w = self._resolve_input_size(*original_size)
        resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        blob = normalized.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
        return {"blob": blob, "original_size": original_size}

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run ONNX inference on preprocessed input.

        Args:
            preprocessed: Output dict from :meth:`preprocess`.

        Returns:
            Dict with keys:

            - ``"matte"``: raw matte array from the model, values in ``[0, 1]``.
            - ``"original_size"``: forwarded from ``preprocessed``.
        """
        matte = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed["blob"]},
        )[0]
        return {"matte": matte, "original_size": preprocessed["original_size"]}

    def postprocess(self, inference_outputs: Dict[str, Any]) -> np.ndarray:
        """Decode the raw matte to a uint8 alpha map at original resolution.

        Args:
            inference_outputs: Output dict from :meth:`model_infer`.

        Returns:
            uint8 array of shape ``(H, W)``; ``255`` is foreground,
            ``0`` is background.
        """
        matte = np.squeeze(inference_outputs["matte"]).astype(np.float32)
        matte = np.clip(matte, 0.0, 1.0)

        h, w = inference_outputs["original_size"]
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_AREA)
        return np.round(matte * 255).astype(np.uint8)

    def run(self, image: np.ndarray) -> np.ndarray:
        """Run the full matting pipeline on an RGB uint8 image.

        Args:
            image: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            uint8 alpha matte of shape ``(H, W)``; ``255`` is foreground,
            ``0`` is background.
        """
        return self.postprocess(self.model_infer(self.preprocess(image)))

    @staticmethod
    def _resolve_input_size(height: int, width: int) -> Tuple[int, int]:
        """Compute the network input size for a given image size.

        Follows the official MODNet ONNX demo: if the image does not already
        straddle the reference size (shorter side ≤ 512 ≤ longer side), scale
        it so the shorter side becomes 512. Both dims are then floored to a
        multiple of the network stride (32), with a lower bound of one stride
        to avoid zero-sized inputs for extremely thin images.

        Args:
            height: Original image height in pixels.
            width: Original image width in pixels.

        Returns:
            ``(input_h, input_w)`` tuple to which the image should be resized.
        """
        if max(height, width) < _REF_SIZE or min(height, width) > _REF_SIZE:
            scale = _REF_SIZE / min(height, width)
            input_h, input_w = int(height * scale), int(width * scale)
        else:
            input_h, input_w = height, width

        input_h = max(input_h - input_h % _STRIDE, _STRIDE)
        input_w = max(input_w - input_w % _STRIDE, _STRIDE)
        return input_h, input_w
