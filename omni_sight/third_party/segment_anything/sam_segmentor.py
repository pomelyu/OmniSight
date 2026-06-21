"""Segment Anything Model (SAM) ONNX wrapper following the OmniSight BasicProcessor contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from omni_sight.basic_processor import BasicProcessor
from omni_sight.onnx.onnx_loader import OnnxLoader

_ENCODER_INPUT_H: int = 684
_ENCODER_INPUT_W: int = 1024

_sam_encoder_loader = OnnxLoader()
_sam_encoder_loader.set_file(
    file_name="mobile_sam_encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/mobile_sam.encoder.onnx?download=true"],
    identifier="mobile_sam_encoder",
)
_sam_encoder_loader.set_file(
    file_name="sam_vit_b_01ec64.encoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_b_01ec64.encoder.onnx?download=true"],
    identifier="sam_vit_b_encoder",
)
_sam_encoder_loader.set_file(
    file_name="sam_vit_l_0b3195.encoder.quant.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_l_0b3195.encoder.quant.onnx?download=true"],
    identifier="sam_vit_l_encoder",
)
_sam_encoder_loader.set_file(
    file_name="sam_vit_h_4b8939.encoder.quant.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_h_4b8939.encoder.quant.onnx?download=true"],
    identifier="sam_vit_h_encoder",
)

_sam_decoder_loader = OnnxLoader()
_sam_decoder_loader.set_file(
    file_name="sam_vit_b_01ec64.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_b_01ec64.decoder.onnx?download=true"],
    identifier="sam_vit_b_decoder",
)
_sam_decoder_loader.set_file(
    file_name="sam_vit_l_0b3195.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_l_0b3195.decoder.onnx?download=true"],
    identifier="sam_vit_l_decoder",
)
_sam_decoder_loader.set_file(
    file_name="sam_vit_h_4b8939.decoder.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_h_4b8939.decoder.onnx?download=true"],
    identifier="sam_vit_h_decoder",
)
_sam_decoder_loader.set_file(
    file_name="sam_vit_b_01ec64.decoder-multi.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_b_01ec64.decoder-multi.onnx?download=true"],
    identifier="sam_vit_b_decoder-multi",
)
_sam_decoder_loader.set_file(
    file_name="sam_vit_l_0b3195.decoder-multi.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_l_0b3195.decoder-multi.onnx?download=true"],
    identifier="sam_vit_l_decoder-multi",
)
_sam_decoder_loader.set_file(
    file_name="sam_vit_h_4b8939.decoder-multi.onnx",
    urls=["https://huggingface.co/pomelyu/model_zoo/resolve/main/sam/sam_vit_h_4b8939.decoder-multi.onnx?download=true"],
    identifier="sam_vit_h_decoder-multi",
)

_ENCODER_IDENTIFIER: Dict[str, str] = {
    "mobile_sam": "mobile_sam_encoder",
    "sam_vit_b": "sam_vit_b_encoder",
    "sam_vit_l": "sam_vit_l_encoder",
    "sam_vit_h": "sam_vit_h_encoder",
    "mobile_sam-m": "mobile_sam_encoder",
    "sam_vit_b-m": "sam_vit_b_encoder",
    "sam_vit_l-m": "sam_vit_l_encoder",
    "sam_vit_h-m": "sam_vit_h_encoder",
}
_DECODER_IDENTIFIER: Dict[str, str] = {
    "mobile_sam": "sam_vit_h_decoder",
    "sam_vit_b": "sam_vit_b_decoder",
    "sam_vit_l": "sam_vit_l_decoder",
    "sam_vit_h": "sam_vit_h_decoder",
    "mobile_sam-m": "sam_vit_h_decoder-multi",
    "sam_vit_b-m": "sam_vit_b_decoder-multi",
    "sam_vit_l-m": "sam_vit_l_decoder-multi",
    "sam_vit_h-m": "sam_vit_h_decoder-multi",
}


class SAMSegmentor(BasicProcessor):
    """Prompt-guided instance segmentor backed by Segment Anything Model (SAM) ONNX.

    Accepts an RGB uint8 image and prompt inputs (points, a bounding box, or
    both) and returns ranked binary mask candidates with quality scores.

    Eight model variants are available.  Append ``-m`` to any base name to use
    the multi-output decoder, which returns more than 3 mask candidates.

    Base variants (decoder outputs 3 mask candidates):

    - ``"mobile_sam"`` — MobileSAM encoder, Apache-2.0 *(default)*
    - ``"sam_vit_b"`` — SAM ViT-Base encoder, Apache-2.0
    - ``"sam_vit_l"`` — SAM ViT-Large encoder, Apache-2.0
    - ``"sam_vit_h"`` — SAM ViT-Huge encoder, Apache-2.0

    Multi-output variants (same encoder, decoder returns N > 3 candidates):

    - ``"mobile_sam-m"``, ``"sam_vit_b-m"``, ``"sam_vit_l-m"``, ``"sam_vit_h-m"``

    Preprocessing follows the samexporter convention: the image is warped to
    a 684×1024 (H×W) canvas via an affine scale transform (no normalization),
    and the same inverse transform is applied to recover masks in the original
    image coordinates.

    ONNX model files are downloaded automatically to ``~/.cache/omnisight/``
    on first use.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from omni_sight.instance_segmentation import SAMSegmentor
        >>> image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        >>> seg = SAMSegmentor(device="cpu")
        >>> masks, scores = seg.run(
        ...     image,
        ...     point_coords=np.array([[300, 200]], dtype=np.float32),
        ...     point_labels=np.array([1], dtype=np.float32),
        ... )
        >>> best_mask = masks[0]  # (H, W) bool, highest-scoring candidate
    """

    def __init__(
        self,
        device: str,
        model_name: Optional[str] = "mobile_sam",
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
    ) -> None:
        """Initialize encoder and decoder ONNX Runtime sessions.

        Args:
            device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
            model_name: Model variant identifier. One of ``"mobile_sam"``,
                ``"sam_vit_b"``, ``"sam_vit_l"``, ``"sam_vit_h"`` or any of
                those with a ``"-m"`` suffix for the multi-output decoder
                (e.g. ``"sam_vit_b-m"``). Ignored for individual sessions when
                the corresponding ``*_path`` argument is provided. Defaults to
                ``"mobile_sam"``.
            encoder_path: Explicit path to a local encoder ``.onnx`` file.
                When given, ``model_name`` is ignored for the encoder session.
            decoder_path: Explicit path to a local decoder ``.onnx`` file.
                When given, ``model_name`` is ignored for the decoder session.

        Raises:
            ValueError: If ``model_name`` is ``None`` and neither
                ``encoder_path`` nor ``decoder_path`` is provided.
            ValueError: If ``model_name`` is not a recognised variant.
        """
        super().__init__(device=device)
        if model_name is None and (encoder_path is None or decoder_path is None):
            raise ValueError(
                "model_name is required when encoder_path or decoder_path is not provided."
            )
        if model_name is not None and model_name not in _ENCODER_IDENTIFIER:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Choose from: {', '.join(_ENCODER_IDENTIFIER)}."
            )

        enc_id = _ENCODER_IDENTIFIER.get(model_name, "") if model_name else ""
        dec_id = _DECODER_IDENTIFIER.get(model_name, "") if model_name else ""

        self.encoder_session = _sam_encoder_loader.get_onnx_session(
            identifier=enc_id,
            model_path=encoder_path,
            device=device,
        )
        self.decoder_session = _sam_decoder_loader.get_onnx_session(
            identifier=dec_id,
            model_path=decoder_path,
            device=device,
        )

        self.enc_input_name: str = self.encoder_session.get_inputs()[0].name
        self.enc_output_name: str = self.encoder_session.get_outputs()[0].name
        self.dec_input_names: List[str] = [
            inp.name for inp in self.decoder_session.get_inputs()
        ]
        self.dec_output_names: List[str] = [
            out.name for out in self.decoder_session.get_outputs()
        ]

    def preprocess(self, img: np.ndarray) -> Dict[str, Any]:
        """Warp an RGB uint8 image into the 684×1024 (H×W) encoder canvas.

        The image is scaled uniformly so that it fits within the
        ``(_ENCODER_INPUT_H, _ENCODER_INPUT_W)`` = ``(684, 1024)`` canvas,
        matching the samexporter convention. No normalization is applied —
        the ONNX models handle it internally.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.

        Returns:
            Dict with keys:

            - ``"blob"``: float32 array of shape ``(684, 1024, 3)``.
            - ``"transform_matrix"``: ``(3, 3)`` float64 homogeneous scale
              matrix used to map original coordinates to encoder space.
            - ``"original_size"``: ``(H, W)`` tuple of the input image.

        Raises:
            ValueError: If ``img`` does not have shape ``(H, W, 3)`` or
                dtype ``uint8``.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("img must have shape (H, W, 3).")
        if img.dtype != np.uint8:
            raise ValueError("img must be dtype uint8.")

        h, w = img.shape[:2]
        scale = min(_ENCODER_INPUT_W / w, _ENCODER_INPUT_H / h)
        transform_matrix = np.array(
            [
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        warped = cv2.warpAffine(
            img,
            transform_matrix[:2],
            (_ENCODER_INPUT_W, _ENCODER_INPUT_H),
            flags=cv2.INTER_LINEAR,
        )

        return {
            "blob": warped.astype(np.float32),
            "transform_matrix": transform_matrix,
            "original_size": (h, w),
        }

    @staticmethod
    def _apply_transform(
        coords: np.ndarray, transform_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply a homogeneous transform matrix to a batch of 2-D coordinates.

        Args:
            coords: float32 array of shape ``(1, N, 2)`` in ``(x, y)`` order.
            transform_matrix: ``(3, 3)`` float64 homogeneous matrix.

        Returns:
            Transformed float32 array of shape ``(1, N, 2)``.
        """
        coords_f = deepcopy(coords).astype(np.float64)
        ones = np.ones((*coords_f.shape[:2], 1), dtype=np.float64)
        coords_h = np.concatenate([coords_f, ones], axis=2)
        transformed = np.matmul(coords_h, transform_matrix.T)
        return transformed[:, :, :2].astype(np.float32)

    def model_infer(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run encoder then decoder on preprocessed image and prompt data.

        The dict ``preprocessed`` must contain the output of :meth:`preprocess`
        and may additionally carry the following prompt keys set by
        :meth:`run` before invoking this method:

        - ``"point_coords"``: ``(N, 2)`` float32 array of ``(x, y)`` clicks in
          the original image coordinate space, or ``None``.
        - ``"point_labels"``: ``(N,)`` float32 array — ``1`` foreground,
          ``0`` background, or ``None`` when ``point_coords`` is ``None``.
        - ``"box"``: ``(4,)`` float32 array ``[x1, y1, x2, y2]`` in the
          original image coordinate space, or ``None``.

        At least one of ``"point_coords"`` or ``"box"`` must be non-``None``.

        Args:
            preprocessed: Output dict from :meth:`preprocess`, extended with
                prompt keys.

        Returns:
            Dict with keys:

            - ``"masks"``: raw float32 logits of shape ``(1, N, H, W)`` where
              N is 3 for base variants or more for ``-m`` multi-output variants.
            - ``"iou_predictions"``: float32 IoU scores of shape ``(1, N)``.
            - ``"original_size"``: ``(H, W)`` of the original input image.
            - ``"transform_matrix"``: ``(3, 3)`` float64 homogeneous matrix
              forwarded from ``preprocessed`` for use in :meth:`postprocess`.

        Raises:
            ValueError: If neither ``"point_coords"`` nor ``"box"`` is present.
        """
        image_embedding = self.encoder_session.run(
            [self.enc_output_name],
            {self.enc_input_name: preprocessed["blob"]},
        )[0]

        transform_matrix: np.ndarray = preprocessed["transform_matrix"]
        h, w = preprocessed["original_size"]
        point_coords = preprocessed.get("point_coords")
        point_labels = preprocessed.get("point_labels")
        box = preprocessed.get("box")

        if point_coords is None and box is None:
            raise ValueError("At least one of point_coords or box must be provided.")

        coords_list: list = []
        labels_list: list = []

        if box is not None:
            box_arr = np.asarray(box, dtype=np.float32)
            coords_list.append([float(box_arr[0]), float(box_arr[1])])
            coords_list.append([float(box_arr[2]), float(box_arr[3])])
            labels_list.extend([2.0, 3.0])

        if point_coords is not None:
            pts = np.asarray(point_coords, dtype=np.float32)
            coords_list.extend(pts.tolist())
            labels_list.extend(np.asarray(point_labels, dtype=np.float32).tolist())

        # Padding point required by the ONNX decoder contract.
        coords_list.append([0.0, 0.0])
        labels_list.append(-1.0)

        coords_arr = np.array(coords_list, dtype=np.float32)[np.newaxis]  # (1, N, 2)
        labels_arr = np.array(labels_list, dtype=np.float32)[np.newaxis]  # (1, N)

        coords_arr = self._apply_transform(coords_arr, transform_matrix)

        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        orig_im_size = np.array([_ENCODER_INPUT_H, _ENCODER_INPUT_W], dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coords_arr,
            "point_labels": labels_arr,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size,
        }
        filtered_inputs = {
            k: v for k, v in decoder_inputs.items() if k in self.dec_input_names
        }

        dec_outputs = self.decoder_session.run(None, filtered_inputs)
        output_map = dict(zip(self.dec_output_names, dec_outputs))

        masks_key = next(
            (
                name
                for name in self.dec_output_names
                if "mask" in name.lower() and "low" not in name.lower()
            ),
            self.dec_output_names[0],
        )
        iou_key = next(
            (name for name in self.dec_output_names if "iou" in name.lower()),
            self.dec_output_names[1]
            if len(self.dec_output_names) > 1
            else self.dec_output_names[0],
        )

        return {
            "masks": output_map[masks_key],
            "iou_predictions": output_map[iou_key],
            "original_size": (h, w),
            "transform_matrix": transform_matrix,
        }

    def postprocess(self, inference_outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse transform to logits and return ranked binary masks.

        The inverse of the preprocessing affine warp is applied to each mask
        logit plane to restore the original image resolution before thresholding.

        Args:
            inference_outputs: Output dict from :meth:`model_infer`.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` — N mask candidates
              sorted from highest to lowest IoU score. N is 3 for base variants
              and greater for ``-m`` multi-output variants.
            - ``iou_scores``: float32 array of shape ``(N,)`` — corresponding
              quality scores.
        """
        masks_logits: np.ndarray = inference_outputs["masks"][0]  # (N, H, W)
        iou_scores: np.ndarray = inference_outputs["iou_predictions"][0]
        h, w = inference_outputs["original_size"]
        transform_matrix: np.ndarray = inference_outputs["transform_matrix"]

        inv_transform = np.linalg.inv(transform_matrix)
        recovered = np.stack(
            [
                cv2.warpAffine(
                    masks_logits[i],
                    inv_transform[:2],
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                )
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
        """Run the full segmentation pipeline on an RGB uint8 image.

        Args:
            img: RGB uint8 image of shape ``(H, W, 3)``.
            point_coords: ``(N, 2)`` float32 array of ``(x, y)`` click
                coordinates in image pixel space, or ``None``.
            point_labels: ``(N,)`` float32 array of click labels — ``1`` for
                foreground, ``0`` for background. Required when
                ``point_coords`` is provided.
            box: ``(4,)`` float32 array ``[x1, y1, x2, y2]`` bounding-box
                prompt in image pixel space, or ``None``.

        Returns:
            Tuple of:

            - ``masks``: bool array of shape ``(N, H, W)`` — N mask candidates
              sorted from highest to lowest IoU score. N is 3 for base variants
              and greater for ``-m`` multi-output variants.
              Use ``masks[0]`` for the best single mask.
            - ``iou_scores``: float32 array of shape ``(N,)`` — corresponding
              quality scores in ``[0, 1]``.

        Raises:
            ValueError: If neither ``point_coords`` nor ``box`` is provided.
            ValueError: If ``img`` does not have shape ``(H, W, 3)`` or
                dtype ``uint8``.
        """
        preprocessed = self.preprocess(img)
        preprocessed["point_coords"] = point_coords
        preprocessed["point_labels"] = point_labels
        preprocessed["box"] = box
        return self.postprocess(self.model_infer(preprocessed))
