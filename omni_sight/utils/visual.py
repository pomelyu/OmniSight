from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

_DEPTH_COLORMAPS: Dict[str, Optional[int]] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "gray": None,
}


def visualize_depth(
    depth: np.ndarray,
    colormap: str = "gray",
) -> np.ndarray:
    """Convert a float32 depth map to a colorized BGR image.

    The depth map is min-max normalised to ``[0, 255]`` before the colormap
    is applied. Pass ``colormap="gray"`` for a plain grayscale output.

    Args:
        depth: float32 array of shape ``(H, W)``.
        colormap: Name of the colormap to apply. One of ``"inferno"``,
            ``"magma"``, ``"plasma"``, ``"viridis"``, ``"turbo"``, ``"gray"``.
            Defaults to ``"inferno"``.

    Returns:
        BGR uint8 image of shape ``(H, W, 3)``.

    Raises:
        ValueError: If ``depth`` is not a 2-D array or ``colormap`` is
            not a recognised name.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D array of shape (H, W).")
    if colormap not in _DEPTH_COLORMAPS:
        raise ValueError(
            f"Unknown colormap '{colormap}'. "
            f"Choose from: {', '.join(_DEPTH_COLORMAPS)}."
        )

    depth_min = float(depth.min())
    depth_max = float(depth.max())
    span = depth_max - depth_min
    if span < 1e-8:
        depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
    else:
        depth_uint8 = ((depth - depth_min) / span * 255).astype(np.uint8)

    cv_code = _DEPTH_COLORMAPS[colormap]
    if cv_code is None:
        return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(depth_uint8, cv_code)


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    size: int,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw a bounding box on an image and return the image.

    Args:
        image: BGR image array of shape ``(H, W, 3)``; modified in place.
        bbox: Bounding box in one of three formats:

            - ``[x1, y1, x2, y2]``
            - ``[x1, y1, x2, y2, score]``
            - ``[[x1, y1], [x2, y2]]``
        color: BGR colour tuple for the rectangle and score label.
        size: Thickness of the rectangle border in pixels.
        confidence: Score to display as a label. If ``None``, falls back to
            the score embedded in ``bbox`` (when present).

    Returns:
        The input ``image`` with the bounding box drawn on it.

    Raises:
        ValueError: If ``bbox`` is not one of the supported formats.
    """
    bbox_array = np.asarray(bbox)

    if bbox_array.shape == (2, 2):
        x1, y1 = bbox_array[0]
        x2, y2 = bbox_array[1]
        score = None
    elif bbox_array.ndim == 1 and bbox_array.size in (4, 5):
        x1, y1, x2, y2 = bbox_array[:4]
        score = float(bbox_array[4]) if bbox_array.size == 5 else None
    else:
        raise ValueError(
            "bbox must be [x1, y1, x2, y2], [x1, y1, x2, y2, c], "
            "or [[x1, y1], [x2, y2]]."
        )

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        int(size),
    )

    label_score = confidence if confidence is not None else score
    if label_score is not None:
        cv2.putText(
            image,
            f"{label_score:.3f}",
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return image


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a binary segmentation mask over an image and return the image.

    Args:
        image: BGR uint8 image of shape ``(H, W, 3)``; modified in place.
        mask: Bool or uint8 array of shape ``(H, W)``. Non-zero pixels are
            treated as foreground.
        color: BGR colour tuple for the mask overlay. Defaults to green.
        alpha: Opacity of the overlay in ``[0, 1]``. Defaults to ``0.4``.

    Returns:
        The input ``image`` with the mask blended on it.

    Raises:
        ValueError: If ``mask`` is not a 2-D array or ``alpha`` is not in
            ``[0, 1]``.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array of shape (H, W).")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    foreground = mask.astype(bool)
    overlay = np.zeros_like(image)
    overlay[foreground] = color
    cv2.addWeighted(overlay, alpha, image, 1.0, 0, dst=image)
    return image


def draw_keypoints(
    image: np.ndarray,
    kps: np.ndarray,
    color: Tuple[int, int, int],
    size: int,
) -> np.ndarray:
    """Draw keypoints on an image and return the image.

    Args:
        image: BGR image array of shape ``(H, W, 3)``; modified in place.
        kps: Keypoint array of shape ``(N, 2)`` with ``(x, y)`` coordinates.
        color: BGR colour tuple for the keypoint circles.
        size: Radius of each keypoint circle in pixels.

    Returns:
        The input ``image`` with the keypoints drawn on it.

    Raises:
        ValueError: If ``kps`` does not have shape ``(N, 2)``.
    """
    keypoints_array = np.asarray(kps)
    if keypoints_array.ndim != 2 or keypoints_array.shape[1] != 2:
        raise ValueError("kps must have shape (N, 2).")

    for point in keypoints_array:
        px = int(point[0])
        py = int(point[1])
        cv2.circle(image, (px, py), int(size), color, -1)

    return image
